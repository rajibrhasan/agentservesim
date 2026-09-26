"""Standalone unit test for the unified-policy mirror (unified_policy.py).

Exercises the adapter directly against a real RadixCache (no ASTRA-Sim
backend, no full Scheduler):
  1. TTL retention: protect on turn completion parks the prefix chain
     (lock_ref), release on next-turn arrival unparks it
  2. safety valve: breaking protections when the LRU cannot supply an
     eviction target, with forced/expired counters
  3. min-waste evict arm: targeted chain eviction
  4. priority ordering: harness stamps order the waiting queue
     (priority, arrival, id) in priority mode
  5. session-affinity routing pins programs to instances
  6. decision parity: the sim-side retention log equals the log the
     real harness executor produces from the same event stream
"""
import io
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.radix_tree import RadixCache
from serving.core.request import Request
from serving.core.scheduler import Scheduler
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter, import_harness

BLOCK = 16


def make_memory():
    cache = RadixCache(node_id=0, device='NPU', page_size=BLOCK,
                       capacity=10**12, kv_size=1024, instance_id=0)
    memory = types.SimpleNamespace(
        npu_prefix_cache=cache, npu_mem=10**12, weight=0, npu_used=0, npu_reserved=0,
        evictable_size=lambda device: cache.evictable_size() * cache.kv_size)
    memory.apply_kv_cache_events = lambda: setattr(
        memory, "npu_used", cache.total_size() * cache.kv_size)
    return memory


def cache_tokens(memory, n, start=0):
    """Insert an n-token chain (the state a finished request's prefix
    is in: cached, unlocked) and return the token ids."""
    tokens = list(range(start, start + n))
    memory.npu_prefix_cache.insert(tokens)
    memory.apply_kv_cache_events()
    res = memory.npu_prefix_cache.match_prefix(tokens)
    assert res.hit_length == n
    return tokens


def make_req(req_id, tokens, arrival_ns=0, latency_ns=2_000_000_000,
             queuing_ns=500_000_000):
    """Finished request whose hashed tokens name the cached chain:
    (input_hash_ids + output_hash_ids)[:-1] == tokens, matching what
    cache_finished_req inserted."""
    req = Request(req_id, "m", 64, 128, arrival_ns, 0,
                  input_hash_ids=list(tokens), output_hash_ids=[10**9])
    req.latency = latency_ns
    req.queuing_delay = queuing_ns
    return req


def make_adapter(**kw):
    defaults = dict(retention_value="ttl", scheduling_value="plas",
                    routing_value="session-affinity", num_instances=2,
                    block_size=BLOCK, tau_s=60.0)
    defaults.update(kw)
    return UnifiedPolicyAdapter(**defaults)


def row(session, idx, arrival_ns, req_index):
    return {"index": req_index, "session_id": session,
            "sub_request_index": idx, "arrival_time_ns": arrival_ns}


def test_ttl_protect_parks_and_arrival_releases():
    ad = make_adapter()
    mem = make_memory()
    node = cache_tokens(mem, 64)
    cache = mem.npu_prefix_cache

    r0 = row("s0", 0, 0, 100)
    ad.select_instance(r0, lambda: 0, 0)
    ad.on_turn_routed(r0)
    assert cache.evictable_size() == 64

    ad.on_turn_complete(make_req(100, node), "s0", 0, "pytest", 64,
                        mem, 2_000_000_000)
    dec = ad.retention_exec.decisions[-1]
    assert dec.action == "protect" and dec.blocks == 4
    assert cache.evictable_size() == 0  # parked: shielded from LRU
    assert cache.protected_size() == 64
    assert ad.kv_stats()["currently_protected"] == 1

    # Next turn arrives 10 s later: protection released, LRU-evictable again.
    r1 = row("s0", 1, 12_000_000_000, 101)
    ad.select_instance(r1, lambda: 0, 12_000_000_000)
    ad.on_turn_routed(r1)
    rel = ad.retention_exec.decisions[-1]
    assert rel.action == "release" and rel.blocks == 4
    assert cache.evictable_size() == 64
    assert ad.kv_stats()["currently_protected"] == 0


def test_safety_valve_breaks_unexpired_protection():
    ad = make_adapter()
    mem = make_memory()
    node = cache_tokens(mem, 64)
    ad.select_instance(row("s0", 0, 0, 100), lambda: 0, 0)
    ad.on_turn_complete(make_req(100, node), "s0", 0, None, 64,
                        mem, 1_000_000_000)
    assert mem.npu_prefix_cache.evictable_size() == 0

    # An eviction needs 32 tokens but everything is parked: the valve
    # breaks the (unexpired) protection block-granularly, tail first:
    # 2 of the 4 blocks come back, the prefix stays pinned and matchable.
    ad.ensure_evictable_tokens(mem, 32)
    assert mem.npu_prefix_cache.evictable_size() == 32
    assert mem.npu_prefix_cache.protected_size() == 32
    stats = ad.kv_stats()
    assert stats["reclaimed_forced"] == 2
    assert stats["reclaimed_expired"] == 0
    assert stats["currently_protected"] == 1
    assert mem.npu_prefix_cache.recount_sizes() == (32, 32)
    # The LRU now takes the tail; the pinned prefix still matches.
    mem.npu_prefix_cache.evict(32)
    assert mem.npu_prefix_cache.match_prefix(list(range(64))).hit_length == 32
    assert mem.npu_prefix_cache.recount_sizes() == (0, 32)
    # The remaining deficit takes the rest; the entry disappears.
    ad.ensure_evictable_tokens(mem, 32)
    assert mem.npu_prefix_cache.evictable_size() == 32
    stats = ad.kv_stats()
    assert stats["reclaimed_forced"] == 4
    assert stats["currently_protected"] == 0
    assert mem.npu_prefix_cache.recount_sizes() == (32, 0)


def test_safety_valve_takes_expired_first():
    ad = make_adapter(tau_s=1.0)
    mem = make_memory()
    n_a = cache_tokens(mem, 32, start=0)
    n_b = cache_tokens(mem, 32, start=1000)
    ad.select_instance(row("sA", 0, 0, 1), lambda: 0, 0)
    ad.on_turn_complete(make_req(1, n_a), "sA", 0, None, 32, mem, 1_000_000_000)
    ad.select_instance(row("sB", 0, 0, 2), lambda: 0, 0)
    ad.on_turn_complete(make_req(2, n_b), "sB", 0, None, 32, mem, 5_000_000_000)
    # At t=5s, sA's protection (deadline 2s) is expired, sB's (6s) is not.
    ad.ensure_evictable_tokens(mem, 16)
    stats = ad.kv_stats()
    assert stats["reclaimed_expired"] == 1   # one block off sA's tail
    assert stats["reclaimed_forced"] == 0
    assert stats["currently_protected"] == 2  # sA keeps its 16-token prefix
    assert mem.npu_prefix_cache.evictable_size() == 16


def test_min_waste_evict_removes_chain():
    _, _, _, waste, _ = import_harness()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"a": 0.0279, "c": 15.4, "S": 384}, f)
        profile = f.name
    # default_gap_s=10 with ctx=1000 makes preserve waste >> discard: evict.
    ad = make_adapter(retention_value="min-waste",
                      min_waste_profile=profile,
                      default_gap_s=10.0)
    mem = make_memory()
    node = cache_tokens(mem, 64)
    ad.select_instance(row("s0", 0, 0, 100), lambda: 0, 0)
    ad.on_turn_complete(make_req(100, node), "s0", 0, "pytest", 1000,
                        mem, 1_000_000_000)
    dec = ad.retention_exec.decisions[-1]
    assert dec.action == "evict"
    assert mem.npu_prefix_cache.evictable_size() == 0  # chain removed
    assert mem.npu_prefix_cache.total_size() == 0
    os.unlink(profile)


def test_priority_orders_waiting_queue():
    ad = make_adapter()
    # Program sA has attained 3 s of service; sB is new.
    ad.scheduling_exec.turn_complete("sA", 3.0)
    pa = ad.on_turn_routed(row("sA", 1, 10_000_000_000, 5))
    pb = ad.on_turn_routed(row("sB", 0, 11_000_000_000, 6))
    assert pa == 3000 and pb == 0

    # Unbound add_request against a stub: priority mode must run sB
    # (later arrival, lower priority) ahead of sA.
    stub = types.SimpleNamespace(scheduling_policy="priority", request=[], max_model_len=None, policy_hooks=None)
    Scheduler.add_request(stub, [5, "m", 64, 128, 10_000_000_000, 0],
                          priority=pa)
    Scheduler.add_request(stub, [6, "m", 64, 128, 11_000_000_000, 0],
                          priority=pb)
    assert [r.id for r in stub.request] == [6, 5]

    # FCFS mode keeps arrival order.
    stub2 = types.SimpleNamespace(scheduling_policy="fcfs", request=[], max_model_len=None, policy_hooks=None)
    Scheduler.add_request(stub2, [5, "m", 64, 128, 10_000_000_000, 0],
                          priority=pa)
    Scheduler.add_request(stub2, [6, "m", 64, 128, 11_000_000_000, 0],
                          priority=pb)
    assert [r.id for r in stub2.request] == [5, 6]


def test_session_affinity_pins_program():
    ad = make_adapter()
    mem = make_memory()
    assert ad.select_instance(row("sA", 0, 0, 1), lambda: 99, 0) == 0
    assert ad.select_instance(row("sB", 0, 0, 2), lambda: 99, 0) == 1
    # sA's next turn follows the pin even though instance 0 is loaded.
    assert ad.select_instance(row("sC", 0, 0, 3), lambda: 99, 0) == 0
    assert ad.select_instance(row("sA", 1, 0, 4), lambda: 99, 0) == 0
    # Flat request (no program identity): stock selection is used.
    flat = {"index": 7, "arrival_time_ns": 0}
    assert ad.select_instance(flat, lambda: 99, 0) == 99
    # Completion decrements the in-flight count.
    ad.on_turn_complete(make_req(1, cache_tokens(mem, 16)), "sA", 0, None,
                        16, mem, 1_000_000_000)
    assert ad.routing_exec.policy.inflight == [2, 1]


def test_decision_parity_with_real_harness_executor():
    """The Phase B statement in miniature: the sim mirror and the real
    harness executor, driven by the same event stream, produce
    position-identical retention decision logs."""
    retention, _, _, _, _ = import_harness()
    from policies.utils.kv_control import RecordingKVControl
    from policies.utils.parity import check_parity

    with tempfile.TemporaryDirectory() as td:
        ad = make_adapter(log_dir=td)
        mem = make_memory()
        node0 = cache_tokens(mem, 64, start=0)
        node1 = cache_tokens(mem, 128, start=5000)

        ad.select_instance(row("s0", 0, 0, 100), lambda: 0, 0)
        ad.on_turn_routed(row("s0", 0, 0, 100))
        ad.on_turn_complete(make_req(100, node0), "s0", 0, "grep", 64,
                            mem, 2_000_000_000)
        r1 = row("s0", 1, 12_000_000_000, 101)
        ad.select_instance(r1, lambda: 0, 12_000_000_000)
        ad.on_turn_routed(r1)
        ad.on_turn_complete(make_req(101, node1), "s0", 1, "pytest", 128,
                            mem, 15_000_000_000)
        ad.finish()
        with open(os.path.join(td, "retention.jsonl")) as f:
            sim_log = f.read().splitlines()

        # Real-harness side: same events, engine emulated by the
        # recording double reporting the same block counts.
        buf = io.StringIO()
        blocks = {"s0:0": 4, "s0:1": 8}
        kv = RecordingKVControl()
        kv.protect = lambda tag, dl: (kv.calls.append(("protect", tag, dl))
                                      or blocks[tag])
        kv.release = lambda tag: (kv.calls.append(("release", tag))
                                  or blocks[tag])
        ex = retention.RetentionExecutor(
            policy=retention.TTLRetention(tau_s=60.0), kv=kv, log_file=buf)
        ex.turn_complete("s0", 0, "s0:0", "grep", now=2.0)
        ex.turn_arrival("s0", now=12.0)
        ex.turn_complete("s0", 1, "s0:1", "pytest", now=15.0)
        ex.finish()

        rep = check_parity("retention", sim_log, buf.getvalue().splitlines())
        assert rep.ok, rep.summary()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                import traceback
                print(f"FAIL {name}: {e}")
                traceback.print_exc()
    print(("%d FAILED" % fails) if fails else "all passed")
    sys.exit(1 if fails else 0)


def test_one_object_can_decide_two_axes_on_the_request_plane():
    """Naming the same `module:Class` on two flags yields ONE instance.

    A policy that decides retention AND scheduling from one piece of state
    needs a single object; two instances of the class would coordinate only
    through module globals, which the search can corrupt with nothing to
    report. Routing such a policy to `--planes program` was the alternative,
    and it costs more than it buys: the request planes are the ones validated
    against real hardware.
    """
    from serving.core.unified_policy_adapter import _spec_or_raise
    cache = {}
    spec = "policies.stock:StockKV"
    a = _spec_or_raise("kv", spec, "retention", cache)
    b = _spec_or_raise("scheduling", spec, "scheduling", cache)
    assert a is b, "two axes built two instances of one class"

    # and a different spec is still a different object
    c = _spec_or_raise("kv", "policies.generic:TTLRetention", "retention", cache)
    assert c is not a
