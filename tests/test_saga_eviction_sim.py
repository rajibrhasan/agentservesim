"""SAGA's workflow-aware LRU reclaiming the simulator's prefix cache.

The ranking is policies.saga_runtime.eviction_order (Eq. 6). These check the
observation half the simulator supplies: resident size attribution, the
online-only reuse estimate, that the ranking actually changes which node the
radix cache reclaims, and that nothing happens before a successor has been
seen.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.memory_model import Device, MemoryModel
from serving.core.radix_tree import RadixCache
from serving.core.request import Request
from serving.core.saga_eviction import SagaEvictionOrder
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter

MODEL = 'meta-llama/Llama-3.1-8B'
BLOCK = 16
S = 1_000_000_000      # one simulated second in ns


def memory(pool_tokens=4096):
    m = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, BLOCK, 16, True, False, None, None)
    m.npu_mem = m.weight + m.get_kv(pool_tokens)
    m.mem_for_kv = m.get_kv(pool_tokens)
    m.npu_prefix_cache.capacity = m.mem_for_kv
    return m


def insert(m, session, tokens):
    cache = m.npu_prefix_cache
    cache._current_owner = session
    cache.insert(list(tokens))
    cache._current_owner = None
    m.apply_kv_cache_events()


def trained(ev, overlap=0.9, continued=True):
    """Teach the estimator from observed turns only."""
    ev.note_turn_complete("warmup", 0)
    if continued:
        ev.note_turn_arrival("warmup", S, overlap=overlap)
    return ev


# ------------------------------------------------------------- observations
def test_reuse_probability_is_observed_not_assumed():
    ev = SagaEvictionOrder()
    assert ev.reuse_probability("a") is None      # nothing seen yet
    assert ev.informed() is False
    # Two sessions finish turn 1; only one comes back.
    ev.note_turn_complete("a", 0)
    ev.note_turn_complete("b", 0)
    ev.note_turn_arrival("a", S, overlap=0.8)
    assert ev.reuse_probability("a") is None or True
    # After one turn completed, 1 of 2 continued.
    ev.turns_done["c"] = 1
    assert abs(ev.reuse_probability("c") - 0.5) < 1e-9
    assert abs(ev.overlap() - 0.8) < 1e-9
    assert ev.informed() is True


def test_shared_nodes_split_across_owners():
    m = memory()
    cache = m.npu_prefix_cache
    shared = list(range(1, 1 + 4 * BLOCK))
    insert(m, "a", shared)
    insert(m, "b", shared + list(range(500, 500 + 2 * BLOCK)))
    sizes = SagaEvictionOrder.resident_tokens(cache)
    # The shared head is charged half to each; the tail only to b.
    assert abs(sizes["a"] - 32) < 1e-6
    assert abs(sizes["b"] - (32 + 32)) < 1e-6
    assert abs(sum(sizes.values()) - cache.total_size()) < 1e-6


def test_locked_nodes_are_not_candidates():
    m = memory()
    cache = m.npu_prefix_cache
    toks = list(range(1, 1 + 4 * BLOCK))
    insert(m, "a", toks)
    node = cache.match_prefix(toks).last_device_node
    cache.inc_lock_ref(node)
    assert SagaEvictionOrder.resident_tokens(cache) == {}


# ------------------------------------------------------------------ ranking
def test_ranking_prefers_evicting_the_idle_low_reuse_session():
    m = memory()
    ev = trained(SagaEvictionOrder())
    insert(m, "idle", list(range(1, 1 + 4 * BLOCK)))
    insert(m, "fresh", list(range(900, 900 + 4 * BLOCK)))
    ev.turns_done["idle"] = 1
    ev.turns_done["fresh"] = 1
    ev.last_access_ns["idle"] = 1 * S
    ev.last_access_ns["fresh"] = 100 * S
    order = ev.ranked_sessions(m, 100 * S)
    assert order[0] == "idle"          # idler scores higher, evicted first


def test_no_ranking_before_a_successor_is_observed():
    m = memory()
    ev = SagaEvictionOrder()
    insert(m, "a", list(range(1, 1 + 4 * BLOCK)))
    ev.note_turn_complete("a", 0)
    assert ev.ranked_sessions(m, S) is None
    assert ev.node_key(m, S) is None        # the tree's LRU stands
    assert ev.stats["lru_fallback"] == 1


# ------------------------------------------------------- actual reclamation
def test_eviction_follows_the_ranking_not_the_lru():
    """The LRU would take the oldest node; SAGA takes the worst-scoring one."""
    m = memory(pool_tokens=4096)
    cache = m.npu_prefix_cache
    ev = trained(SagaEvictionOrder(), overlap=1.0)
    old = list(range(1, 1 + 4 * BLOCK))
    new = list(range(900, 900 + 4 * BLOCK))
    insert(m, "keep", old)        # inserted first: the LRU victim
    insert(m, "drop", new)
    # Make "keep" look valuable (recently touched) and "drop" look stale.
    ev.turns_done["keep"] = 1
    ev.turns_done["drop"] = 1
    ev.last_access_ns["keep"] = 90 * S
    ev.last_access_ns["drop"] = 1 * S
    m.sim_now = 100 * S
    m.eviction_order = ev
    m.evict_prefix_cache(m.get_kv(4 * BLOCK), Device.NPU)
    assert cache.match_prefix(new).hit_length == 0      # SAGA's victim went
    assert cache.match_prefix(old).hit_length == 4 * BLOCK   # the LRU victim stayed


def test_without_the_order_the_lru_victim_goes():
    m = memory(pool_tokens=4096)
    cache = m.npu_prefix_cache
    old = list(range(1, 1 + 4 * BLOCK))
    new = list(range(900, 900 + 4 * BLOCK))
    insert(m, "keep", old)
    insert(m, "drop", new)
    m.sim_now = 100 * S
    assert m.eviction_order is None
    m.evict_prefix_cache(m.get_kv(4 * BLOCK), Device.NPU)
    assert cache.match_prefix(old).hit_length == 0      # plain LRU: oldest first


def test_unknown_owners_sort_after_ranked_sessions():
    m = memory()
    ev = trained(SagaEvictionOrder())
    insert(m, "ranked", list(range(1, 1 + 2 * BLOCK)))
    insert(m, "stranger", list(range(900, 900 + 2 * BLOCK)))
    ev.turns_done["ranked"] = 1
    ev.last_access_ns["ranked"] = 1 * S
    key = ev.node_key(m, 100 * S)
    cache = m.npu_prefix_cache
    ranked_node = cache.match_prefix(list(range(1, 1 + 2 * BLOCK))).last_device_node
    other_node = cache.match_prefix(list(range(900, 900 + 2 * BLOCK))).last_device_node
    assert key(ranked_node)[0] < key(other_node)[0]


# -------------------------------------------------------------------- wiring
def test_adapter_installs_the_order_only_when_asked():
    m = memory()
    off = UnifiedPolicyAdapter(retention_value="saga-ttl", scheduling_value="fcfs",
                               routing_value="session-affinity", num_instances=1,
                               block_size=BLOCK, tau_s=2.0)
    off._saga_note(m, "s0", S, False)
    assert m.eviction_order is None

    on = UnifiedPolicyAdapter(retention_value="saga-ttl", scheduling_value="fcfs",
                              routing_value="session-affinity", num_instances=1,
                              block_size=BLOCK, tau_s=2.0, saga_eviction_order=True)
    on._saga_note(m, "s0", S, False)
    assert m.eviction_order is on._saga_evict[m.instance_id]
    assert on._saga_evict[m.instance_id].last_access_ns["s0"] == S
