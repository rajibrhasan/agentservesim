"""Regress the two-instance false stall with production request schedulers."""
from serving.core.idle_sweep import IdleScheduleSweep
from serving.tests.test_request_cache_recovery import scheduler, request, memory


def test_empty_instance_does_not_declare_other_runnable_instance_stuck():
    schedulers = [scheduler(), scheduler()]
    req = request(1, 0, 16)
    req.num_computed_tokens = 0
    schedulers[1].request.append(req)
    assert schedulers[0].schedule(0, 0) is None
    assert not any(s.inflight for s in schedulers)
    pending = [i for i, s in enumerate(schedulers) if not s.is_request_empty()]
    # The former guard raised here before giving instance 1 a turn.
    assert pending == [1]
    assert not IdleScheduleSweep().record_failure(0, pending)
    assert schedulers[1].schedule(0, 0) is not None


def test_genuine_capacity_stall_requires_both_instances_to_fail():
    sweep = IdleScheduleSweep()
    schedulers = [scheduler(), scheduler()]
    for i, s in enumerate(schedulers):
        s.memory = memory(16)
        req = request(i, 0, 32)
        req.num_computed_tokens = 0
        s.request.append(req)
        assert s.schedule(0, 0) is None
        assert sweep.record_failure(i, [0, 1]) == (i == 1)


def test_completion_or_arrival_invalidates_old_failed_attempts():
    sweep = IdleScheduleSweep()
    assert not sweep.record_failure(1, [0, 1])
    sweep.reset()
    assert not sweep.record_failure(0, [0, 1])
    assert sweep.record_failure(1, [0, 1])


def test_no_pending_work_allows_tool_gap_fast_forward():
    assert IdleScheduleSweep().record_failure(0, [])
