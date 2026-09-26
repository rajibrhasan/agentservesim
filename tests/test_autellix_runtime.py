import pytest

from policies.autellix_runtime import AutellixRuntime, QueueConfig


def runtime(beta=1000):
    return AutellixRuntime(QueueConfig((1.0, 4.0), (1.0, 2.0, 4.0), beta))


def one_slot(rid, selected):
    return not selected


def test_quantum_exhaustion_preempts_long_call_without_memory_pressure():
    r = runtime()
    r.arrive('long', 'a', 0)
    r.arrive('short', 'b', 0)
    r.start_batch(['long'], 0)
    r.finish_batch(1)
    plan = r.plan(1, ['long'], one_slot)
    assert plan.selected == ('short',)
    assert plan.preempt == ('long',)
    assert r.calls['long'].queue == 1


def test_wait_and_tool_gap_are_not_execution_service():
    r = runtime()
    r.arrive('a0', 'a', 0)
    r.start_batch(['a0'], 10)
    r.finish_batch(11, ['a0'])
    assert r.processes['a'].service_s == 1
    assert r.processes['a'].wait_s == 10
    r.arrive('a1', 'a', 111)
    assert r.calls['a1'].inherited_service_s == 1
    assert r.calls['a1'].queue == 1


def test_parallel_calls_update_critical_path_not_sum():
    r = runtime()
    r.arrive('left', 'p', 0)
    r.arrive('right', 'p', 0)
    r.start_batch(['left', 'right'], 0)
    r.finish_batch(1, ['left'])
    r.start_batch(['right'], 1)
    r.finish_batch(3, ['right'])
    assert r.processes['p'].service_s == 3
    r.arrive('join', 'p', 3)
    assert r.calls['join'].inherited_service_s == 3


def test_promotion_preserves_lifetime_execution_for_completion():
    r = runtime(beta=2)
    r.arrive('a', 'p', 0)
    r.start_batch(['a'], 0)
    r.finish_batch(1)
    assert r.calls['a'].queue == 1
    assert r.plan(4, [], one_slot).selected == ('a',)
    assert r.calls['a'].queue == 0
    assert r.calls['a'].fairness_service_s == 0
    assert r.calls['a'].execution_s == 1
    r.start_batch(['a'], 4)
    r.finish_batch(4.5, ['a'])
    assert r.processes['p'].service_s == 1.5


def test_bottom_queue_rotates_and_waiter_does_not_gain_service():
    r = runtime()
    r.arrive('a', 'p', 0, inherited_service_s=5)
    r.arrive('b', 'q', 0, inherited_service_s=5)
    r.start_batch(['a'], 0)
    r.finish_batch(4)
    assert r.plan(4, ['a'], one_slot).selected == ('b',)
    assert r.calls['b'].execution_s == 0
    assert r.calls['b'].wait_s == 4


def test_arrival_during_batch_waits_only_since_arrival():
    r = runtime()
    r.arrive('a', 'p', 0)
    r.start_batch(['a'], 0)
    r.arrive('b', 'q', 0.75)
    r.finish_batch(1, ['a'])
    r.plan(1, [], one_slot)
    assert r.calls['b'].wait_s == 0.25


def test_batch_completion_is_exactly_once_and_invalid_events_do_not_commit():
    r = runtime()
    r.arrive('a', 'p', 0)
    r.start_batch(['a'], 0)
    with pytest.raises(ValueError):
        r.finish_batch(1, ['unknown'])
    assert r.calls['a'].execution_s == 0
    r.finish_batch(1)
    with pytest.raises(RuntimeError):
        r.finish_batch(1)
    assert r.calls['a'].execution_s == 1


def test_no_overlapping_batches_or_clock_reversal():
    r = runtime()
    r.arrive('a', 'p', 1)
    with pytest.raises(ValueError):
        r.start_batch(['a'], 0)
    r.start_batch(['a'], 1)
    with pytest.raises(RuntimeError):
        r.start_batch(['a'], 1)
    with pytest.raises(RuntimeError):
        r.plan(1, [], one_slot)


@pytest.mark.parametrize('bounds,quanta,beta', [
    ((1, 1), (1, 2, 3), 2), ((1,), (1,), 2),
    ((1,), (1, 0), 2), ((1,), (1, 2), float('nan')),
])
def test_invalid_configuration(bounds, quanta, beta):
    with pytest.raises(ValueError):
        QueueConfig(bounds, quanta, beta)


def test_overprovisioning_keeps_the_next_calls_queued_beyond_the_fit():
    r = runtime()
    for rid in ('a', 'b', 'c', 'd'):
        r.arrive(rid, rid, 0)
    plan = r.plan(0, [], one_slot, overprovision=2)
    assert plan.selected == ('a',)
    assert plan.overprovisioned == ('b', 'c'), 'queue order past the first non-fit, bounded'
    # Overprovisioned residents are not preemption victims.
    plan = r.plan(0, ['a', 'b', 'd'], one_slot, overprovision=1)
    assert plan.selected == ('a',) and plan.overprovisioned == ('b',) and plan.preempt == ('d',)
    assert r.plan(0, [], one_slot).overprovisioned == ()
    with pytest.raises(ValueError):
        r.plan(0, [], one_slot, overprovision=-1)


def test_first_nonfit_stops_selection_even_without_overprovision():
    r = runtime()
    for rid in ('large', 'small'):
        r.arrive(rid, rid, 0)
    fits = lambda rid, selected: rid == 'small'
    assert r.plan(0, [], fits).selected == ()
    plan = r.plan(0, [], fits, overprovision=1)
    assert plan.selected == ()
    assert plan.overprovisioned == ('large',)
