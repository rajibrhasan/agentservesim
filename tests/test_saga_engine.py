"""SAGA WA-LRU: the engine breaks pins in the order the policy publishes."""
import time
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from vllm.v1.request import RequestStatus

from test_vllm_policy_integration import add, make_native, step


def protected_pool(tmp_path, tags=('a', 'b', 'c'), deadlines=None):
    s, dma, _ = make_native(tmp_path)
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    manager = s.kv_cache_manager
    manager.enable_kv_protection = manager.block_pool.enable_kv_protection = True
    for i, tag in enumerate(tags):
        # Distinct prompts: shared prefixes would pin the same physical blocks.
        add(s, tag, [100 * (i + 1) + t for t in range(33)], output=1)
        while tag in s.requests:
            step(s, dma, runner)
    far = time.time() + 1000
    for i, tag in enumerate(tags):
        deadline = far if deadlines is None else deadlines[i]
        assert manager.kv_protect(tag, deadline) == 2, 'two full blocks per finished request'
    return s, manager, manager.block_pool


def parked(manager, pool, tag):
    return sum(bid in pool._protected for bid in manager._protected_requests.get(tag, ()))


def parked_ids(manager, pool, tag):
    # A finished request's partial tail block is tracked but never parked.
    return {bid for bid in manager._protected_requests.get(tag, ()) if bid in pool._protected}


def test_published_order_decides_which_unexpired_pins_break_first(tmp_path):
    s, manager, pool = protected_pool(tmp_path)
    assert s.kv_reclaim_order(['b', 'c', 'a']) == 6
    free = pool.free_block_queue.num_free_blocks
    pool.get_new_blocks(free + 2)          # the valve must break exactly two pins
    assert (parked(manager, pool, 'a'), parked(manager, pool, 'b'), parked(manager, pool, 'c')) == (2, 0, 2)
    pool.get_new_blocks(2)
    assert (parked(manager, pool, 'a'), parked(manager, pool, 'c')) == (2, 0)
    assert pool.protection_stats['reclaimed_forced'] == 4
    assert pool.protection_stats['reclaimed_expired'] == 0


def test_unranked_pins_keep_latest_deadline_first_and_expired_go_first(tmp_path):
    now = time.time()
    s, manager, pool = protected_pool(tmp_path, deadlines=(now + 100, now + 500, now - 5))
    assert s.kv_reclaim_order(['a']) == 2       # only a is ranked
    free = pool.free_block_queue.num_free_blocks
    pool.get_new_blocks(free + 2)
    # Expired protections are reclaimed before any ranking applies.
    assert parked(manager, pool, 'c') == 0 and parked(manager, pool, 'a') == 2
    pool.get_new_blocks(2)
    # Then the ranked request, before the unranked latest-deadline one.
    assert parked(manager, pool, 'a') == 0 and parked(manager, pool, 'b') == 2
    assert s.kv_reclaim_order([]) == 0 and not pool._reclaim_rank


def test_order_ignores_unknown_and_released_requests(tmp_path):
    s, manager, pool = protected_pool(tmp_path)
    assert manager.kv_release('b') == 2
    assert s.kv_reclaim_order(['ghost', 'b', 'a']) == 2
    assert set(pool._reclaim_rank) == parked_ids(manager, pool, 'a')
    assert all(rank == 2 for rank in pool._reclaim_rank.values())
    manager.kv_release('a')
    assert not pool._reclaim_rank, 'releasing a pin drops its rank'


def test_engine_utility_delegates_to_the_scheduler(tmp_path):
    from types import MethodType
    from vllm.v1.engine.core import EngineCore

    s, manager, pool = protected_pool(tmp_path)
    core = SimpleNamespace(scheduler=s)
    assert MethodType(EngineCore.kv_reclaim_order, core)(['c']) == 2
    assert set(pool._reclaim_rank) == parked_ids(manager, pool, 'c')
