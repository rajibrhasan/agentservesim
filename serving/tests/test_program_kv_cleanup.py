"""A finished turn must leave the pool as it found it.

Block alignment (ebe03ef) introduced both of these: consecutive decode steps
began resolving to the SAME radix node, and the tail that does not fill a block
began being held as a reservation. Neither cleanup path had to exist before.
"""

import types

import pytest

from serving.core.program_kv import ProgramKVManager
from serving.core.radix_tree import RadixCache

BLOCK = 16


def _kv():
    rx = RadixCache(node_id=0, device="NPU", page_size=BLOCK, capacity=10 ** 9,
                    kv_size=1, instance_id=0)
    orch = types.SimpleNamespace(attribution=types.SimpleNamespace(),
                                 pinned=lambda: [])
    return rx, ProgramKVManager(instance=0, capacity_tokens=10 ** 6,
                                orchestrator=orch, radix=rx, attribution=None,
                                block_size=BLOCK)


def _run_turn(kv, computed):
    """Commit once per decode step, as the scheduler does."""
    ids = list(range(computed))
    for n in range(BLOCK, computed + 1):
        kv.allocate("p", 1, 0.0, owner="p")
        kv.commit("p", ids[:n], owner="p")
    return ids


@pytest.mark.parametrize("computed", [32, 33, 48, 49])
def test_a_finished_turn_leaves_nothing_locked(computed):
    """Commits that resolve to the same node must not stack lock references.

    Steps 33..47 all floor to the 32-token node; taking a reference each time
    left fifteen the single release at completion could not give back, and a
    locked node is reclaimable by nothing -- not pressure, not a policy.
    """
    rx, kv = _kv()
    ids = _run_turn(kv, computed)
    kv.release(ids, owner="p")
    _, locked = rx.recount_sizes()
    assert locked == 0


@pytest.mark.parametrize("computed", [32, 33, 48, 49])
def test_a_finished_turn_leaves_nothing_reserved(computed):
    rx, kv = _kv()
    ids = _run_turn(kv, computed)
    kv.release(ids, owner="p")
    assert sum(kv._inflight.values()) == 0


@pytest.mark.parametrize("computed", [16, 17, 32, 33, 48])
def test_while_running_the_whole_context_is_charged(computed):
    """Published blocks plus the unfilled tail account for every token."""
    rx, kv = _kv()
    _run_turn(kv, computed)
    assert rx.total_size() + sum(kv._inflight.values()) == computed
    assert rx.total_size() % BLOCK == 0


def test_only_whole_blocks_are_published():
    rx, kv = _kv()
    _run_turn(kv, 48)
    assert rx.total_size() == 48       # not 288, as the unaligned insert gave
