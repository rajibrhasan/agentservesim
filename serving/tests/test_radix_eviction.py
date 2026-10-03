"""Partial eviction: take the pages asked for, keep the prefix matchable."""

import pytest

from serving.core.radix_tree import RadixCache

PAGE = 16


def _cache(n_tokens=64):
    rc = RadixCache(node_id=0, device="NPU", page_size=PAGE, capacity=10 ** 9,
                    kv_size=1, instance_id=0)
    rc.insert(list(range(n_tokens)))
    return rc


@pytest.mark.parametrize("ask", [16, 32, 48, 64])
def test_takes_only_what_was_asked(ask):
    rc = _cache()
    before = rc.total_size()
    rc.evict(ask)
    assert before - rc.total_size() == ask


@pytest.mark.parametrize("ask", [16.0, 15.5, 16.5, 32.0])
def test_a_float_target_is_an_index(ask):
    """The caller's byte arithmetic produces one; slicing needs an int."""
    rc = _cache()
    before = rc.total_size()
    rc.evict(ask)                       # must not raise
    removed = before - rc.total_size()
    assert removed >= ask and removed % PAGE == 0


def test_the_surviving_prefix_still_matches():
    rc = _cache()
    rc.evict(16)
    assert rc.match_prefix(list(range(48))).hit_length == 48


def test_counters_agree_with_a_tree_walk():
    rc = _cache()
    rc.evict(16)
    evictable, locked = rc.recount_sizes()
    assert evictable == rc.evictable_size()
    assert locked == 0


def test_a_locked_node_is_not_taken():
    rc = _cache()
    node = rc.match_prefix(list(range(64))).last_device_node
    rc.inc_lock_ref(node)
    before = rc.total_size()
    rc.evict(16)
    assert rc.total_size() == before
