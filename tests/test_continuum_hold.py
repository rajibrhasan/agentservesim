"""QueueHolds lives in the patched engine; see engine_files/kv_queue_holds.py."""
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from vllm.v1.core.kv_queue_holds import QueueHolds


def manager():
    pool = SimpleNamespace(_protected={1: 2.0},
                           blocks={1: SimpleNamespace(block_hash=b'first')})
    pool.set_protection_deadline = lambda ids, deadline: pool._protected.update(
        (i, deadline) for i in ids if i in pool._protected)
    return SimpleNamespace(block_pool=pool, _protected_requests={'p': [1]},
                           KV_HOLD_DEADLINE=1e18)


def test_multiple_waiters_keep_hold_until_last_cancel():
    m, holds = manager(), QueueHolds()
    holds.capture(m, 'p')
    m.block_pool._protected[1] = 1e18
    holds.capture(m, 'p')
    holds.finish(m, 'p', cancelled=True)
    assert m.block_pool._protected[1] == 1e18
    holds.finish(m, 'p', cancelled=True)
    assert m.block_pool._protected[1] == 2
    assert not holds.blocks and not holds.counts


def test_cancel_does_not_resurrect_pressure_reclaimed_block():
    m, holds = manager(), QueueHolds()
    holds.capture(m, 'p')
    del m.block_pool._protected[1]
    holds.finish(m, 'p', cancelled=True)
    assert not m.block_pool._protected


def test_cancel_does_not_change_reused_physical_block():
    m, holds = manager(), QueueHolds()
    holds.capture(m, 'p')
    m.block_pool._protected[1] = 1e18
    m.block_pool.blocks[1].block_hash = b'new-request'
    holds.finish(m, 'p', cancelled=True)
    assert m.block_pool._protected[1] == 1e18


def test_explicit_release_discards_hold_without_recreating_pin():
    m, holds = manager(), QueueHolds()
    holds.capture(m, 'p')
    m.block_pool._protected[1] = 1e18
    holds.finish(m, 'p')
    assert not holds.blocks and not holds.counts
    holds.finish(m, 'p', cancelled=True)
    assert m.block_pool._protected[1] == 1e18


def test_overlapping_tags_restore_only_after_last_hold():
    m, holds = manager(), QueueHolds()
    holds.capture(m, 'p')
    m.block_pool._protected[1] = 1e18
    m._protected_requests['q'] = [1]
    holds.capture(m, 'q')
    holds.finish(m, 'p', cancelled=True)
    assert m.block_pool._protected[1] == 1e18
    holds.finish(m, 'q', cancelled=True)
    assert m.block_pool._protected[1] == 2
