import pytest

from policies.saga_runtime import (
    CacheObservation, SagaPlacement, Successor, TaskEstimate,
    WorkerObservation, eviction_order, fair_shares,
)


def test_eviction_keeps_likely_successor_over_recent_terminal_session():
    paused = CacheObservation('paused', 0, 100, (Successor(1, 1),))
    terminal = CacheObservation('done', 9, 100, ())
    assert eviction_order([paused, terminal], 10, 10) == (terminal, paused)


def test_eviction_size_and_prefix_overlap_affect_order():
    small = CacheObservation('small', 1, 10, (Successor(1, 1),))
    large = CacheObservation('large', 1, 100, (Successor(1, 1),))
    unlikely = CacheObservation('unlikely', 1, 10, (Successor(0.1, 0.5),))
    assert eviction_order([small, large, unlikely], 1, 0) == (unlikely, large, small)
    with pytest.raises(ValueError, match='exclusive'):
        CacheObservation('invalid', 0, 1, (Successor(0.9, 1), Successor(0.9, 1)))


def test_fair_share_aggregates_tenants_and_handles_deadlines_explicitly():
    tasks = [TaskEstimate('a', 'one', 10, 20), TaskEstimate('b', 'one', 5, 20),
             TaskEstimate('c', 'two', 5, 20)]
    assert fair_shares(tasks, 10, 0.1) == {'one': 0.75, 'two': 0.25}
    assert fair_shares(tasks, 30, 0.1) == {'one': 0.75, 'two': 0.25}
    with pytest.raises(ValueError, match='exactly once'):
        fair_shares(tasks + tasks[:1], 10, 0.1)


def test_affinity_requires_actual_cached_state_and_load_headroom():
    placement = SagaPlacement()
    placement.home['p'] = 0
    workers = [WorkerObservation(0, 0.7), WorkerObservation(1, 0.1)]
    assert placement.route('p', workers, {0}) == 0
    assert placement.route('p', workers, set()) == 1
    assert placement.route('p', [WorkerObservation(0, 0.8), workers[1]], {0}) == 1


def test_steal_requires_idle_window_and_excess_then_updates_only_after_ack():
    placement = SagaPlacement()
    placement.home['old'] = 0
    source = WorkerObservation(0, 0.9, (('new', 2), ('old', 1)))
    target = WorkerObservation(1, 0.1, empty_since_s=3)
    assert placement.propose_steal(1, [source, target], 3.05) is None
    balanced = WorkerObservation(0, 0.2, source.queued_sessions)
    assert placement.propose_steal(1, [balanced, target], 4) is None
    proposal = placement.propose_steal(1, [source, target], 4)
    assert proposal.session == 'old'
    assert placement.home['old'] == 0
    assert placement.propose_steal(1, [source, target], 5) is None
    placement.complete_steal(proposal, published=False)
    assert placement.home['old'] == 0
    proposal = placement.propose_steal(1, [source, target], 5)
    placement.complete_steal(proposal, published=True)
    assert placement.home['old'] == 1
    with pytest.raises(ValueError, match='stale'):
        placement.complete_steal(proposal, published=True)


def test_zero_load_target_can_steal_but_only_from_positive_load():
    placement = SagaPlacement()
    target = WorkerObservation(1, 0, empty_since_s=0)
    idle = WorkerObservation(0, 0, (('p', 0),))
    assert placement.propose_steal(1, [idle, target], 1) is None
    busy = WorkerObservation(0, 0.5, idle.queued_sessions)
    assert placement.propose_steal(1, [busy, target], 1).source == 0
