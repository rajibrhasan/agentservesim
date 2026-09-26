"""Exercise the production pin planner and allocation paths under sharing."""
import copy

import pytest

from serving.core.memory_model import Device, KVCapacityError, MemoryModel
from serving.core.request import Request
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter

MODEL = 'meta-llama/Llama-3.1-8B'


def setup_shared_pins(capacity=48, shared=16, tail=16):
    memory = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, 16, 16,
                         True, False, None, None)
    memory.mem_for_kv = memory.get_kv(capacity)
    memory.npu_mem = memory.weight + memory.mem_for_kv
    memory.npu_prefix_cache.capacity = memory.mem_for_kv
    adapter = UnifiedPolicyAdapter('continuum', 'fcfs', 'session-affinity', 1, 16)
    memory.kv_protection = adapter
    for tag, start in [('a', 100), ('b', 200)]:
        ids = list(range(shared)) + list(range(start, start + tail))
        req = Request(tag, MODEL, len(ids), len(ids) + 1, 0, 0, ids, [999])
        req.num_computed_tokens = len(ids)
        memory.cache_unfinished_req(req, Device.NPU)
        node = req.npu_last_node
        memory.unlock_prefix(req, Device.NPU)
        adapter._ctx = (memory, node, len(ids))
        adapter.protect(tag, 100.0)
    return memory, adapter


def snapshot(memory, adapter):
    cache = memory.npu_prefix_cache
    nodes = []
    stack = [cache.root_node]
    while stack:
        node = stack.pop()
        nodes.append((id(node), tuple(node.key), node.lock_ref))
        stack.extend(node.children.values())
    return (sorted(nodes),
            [(tag, id(e.node), e.tokens, e.deadline_ns) for tag, e in adapter._parked.items()],
            copy.deepcopy(adapter.stats), memory.npu_used, memory.npu_reserved)


def incoming(length):
    return Request('new', MODEL, length, length + 1, 0, 0,
                   list(range(1000, 1000 + length)), [9999])


def test_shared_retention_prefix_is_counted_once_and_collectively_reclaimable():
    memory, adapter = setup_shared_pins()
    assert memory.npu_prefix_cache.total_size() == 48
    assert adapter.parked_tokens(memory) == 64
    assert adapter.reclaimable_parked_tokens(memory) == 48


@pytest.mark.parametrize('path', ['reserve', 'unfinished', 'finished'])
def test_full_allocation_can_reclaim_two_overlapping_pins(path):
    memory, adapter = setup_shared_pins()
    req = incoming(48)
    if path == 'reserve':
        memory.reserve_kv([req], {req.id: 48})
        assert req.kv_reserved == memory.get_kv(48)
    else:
        req.num_computed_tokens = 48
        publish = memory.cache_unfinished_req if path == 'unfinished' else memory.cache_finished_req
        publish(req, Device.NPU)
    assert not adapter._parked
    assert adapter.stats['reclaimed_forced'] == 3
    assert memory.npu_used + memory.npu_reserved <= memory.npu_mem


@pytest.mark.parametrize('path', ['valve', 'reserve', 'unfinished', 'finished'])
def test_impossible_allocation_changes_no_pin_or_account(path):
    memory, adapter = setup_shared_pins()
    before = snapshot(memory, adapter)
    req = incoming(64)
    if path == 'valve':
        assert adapter.ensure_evictable_tokens(memory, 64) is False
    else:
        with pytest.raises(KVCapacityError):
            if path == 'reserve':
                memory.reserve_kv([req], {req.id: 64})
            else:
                req.num_computed_tokens = 64
                publish = memory.cache_unfinished_req if path == 'unfinished' else memory.cache_finished_req
                publish(req, Device.NPU)
    assert snapshot(memory, adapter) == before


def test_running_reference_remains_unreclaimable():
    memory, adapter = setup_shared_pins()
    cache = memory.npu_prefix_cache
    shared_node = cache.match_prefix(list(range(16))).last_device_node
    cache.inc_lock_ref(shared_node)
    assert adapter.reclaimable_parked_tokens(memory) == 32
    before = snapshot(memory, adapter)
    assert adapter.ensure_evictable_tokens(memory, 48) is False
    assert snapshot(memory, adapter) == before
    assert adapter.ensure_evictable_tokens(memory, 32) is True
    assert cache.evictable_size() == 32
    # Release surviving retention references; the running prefix still holds.
    for tag in list(adapter._parked):
        adapter.release(tag)
    assert cache.protected_size() == 16
    cache.dec_lock_ref(shared_node)
    assert cache.evictable_size() == 48


@pytest.mark.parametrize('need', [16, 32, 48])
def test_partial_plan_matches_freed_space_and_preserves_remaining_chains(need):
    memory, adapter = setup_shared_pins()
    before = snapshot(memory, adapter)
    plan = adapter._plan_pin_reclaim(memory, need)
    assert sum(freed for _, _, freed in plan) == need
    assert snapshot(memory, adapter) == before
    assert adapter.ensure_evictable_tokens(memory, need) is True
    cache = memory.npu_prefix_cache
    assert cache.evictable_size() == need
    assert cache.protected_size() == 48 - need
    for entry in adapter._parked.values():
        assert entry.tokens == adapter._pin_chain_tokens(entry.node, cache.root_node)
    for tag in list(adapter._parked):
        adapter.release(tag)
    assert cache.protected_size() == 0
    assert cache.evictable_size() == 48
    assert adapter.stats['released'] + adapter.stats['reclaimed_forced'] == 3


def test_plan_survives_splitting_compressed_nodes():
    memory, adapter = setup_shared_pins(capacity=96, shared=32, tail=32)
    assert adapter.ensure_evictable_tokens(memory, 16) is True
    assert memory.npu_prefix_cache.evictable_size() == 16
    assert adapter._parked['a'].tokens == 48
    assert adapter.ensure_evictable_tokens(memory, 80) is True
    assert memory.npu_prefix_cache.evictable_size() == 80
    assert sum(e.tokens for e in adapter._parked.values()) == 16
    for tag in list(adapter._parked):
        adapter.release(tag)
    assert memory.npu_prefix_cache.evictable_size() == 96


def test_expired_pin_precedes_unexpired_pin():
    memory, adapter = setup_shared_pins()
    adapter._now_ns = 10
    adapter._parked['b'].deadline_ns = 5
    assert adapter.ensure_evictable_tokens(memory, 16) is True
    assert adapter.stats['reclaimed_expired'] == 1
    assert adapter.stats['reclaimed_forced'] == 0
    assert adapter._parked['a'].tokens == 32
    assert adapter._parked['b'].tokens == 16
