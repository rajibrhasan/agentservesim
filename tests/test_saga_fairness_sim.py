"""SAGA's adaptive fair share on the simulator.

The calculation is policies.saga_runtime.fair_shares (Eqs. 8, 9). These check
the inputs the simulator supplies: that the two it cannot guess are demanded
rather than invented, that remaining work is estimated from observed history,
and that the shares reach the queue as a weighted-service stamp.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.saga_fairness import SagaFairness
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
from policies.program import ProgramTable

NS = 1_000_000_000


def session(sid, tenant=None, deadline_ns=None):
    row = {"session_id": sid, "arrival_time_ns": 0, "sub_requests": []}
    if tenant is not None:
        row["tenant"] = tenant
    if deadline_ns is not None:
        row["deadline_ns"] = deadline_ns
    return row


# --------------------------------------------------------------- refusals
def test_missing_tenant_or_deadline_is_refused_by_name():
    f = SagaFairness()
    for row, missing in ((session("s0"), "tenant"),
                         (session("s1", tenant="t"), "deadline_ns"),
                         (session("s2", deadline_ns=NS), "tenant")):
        try:
            f.register(row["session_id"], row)
        except ValueError as e:
            assert missing in str(e) and row["session_id"] in str(e)
        else:
            raise AssertionError(f"accepted {row}")


def test_slack_must_be_positive():
    try:
        SagaFairness(overdue_slack_s=0)
    except ValueError as e:
        assert "recorded port parameter" in str(e)
    else:
        raise AssertionError("accepted a zero slack")


# -------------------------------------------------------------- estimates
def test_remaining_work_is_estimated_from_observed_history():
    f = SagaFairness()
    assert f.remaining_gpu_s(1) is None          # nothing seen yet
    f.note_turn_service(2.0)
    f.note_turn_service(4.0)
    assert f.mean_service_s() == 3.0
    # Four programs reached turn 1; three continued -> p=0.75, tail 3 turns.
    for _ in range(4):
        f.note_turn_complete(1)
    for _ in range(3):
        f.note_turn_arrival(1)
    assert abs(f.expected_further_turns(1) - 3.0) < 1e-9
    assert abs(f.remaining_gpu_s(1) - 9.0) < 1e-9


def test_shares_favour_the_tenant_with_more_work_per_unit_slack():
    f = SagaFairness(overdue_slack_s=1.0)
    f.register("a", session("a", tenant="urgent", deadline_ns=10 * NS))
    f.register("b", session("b", tenant="relaxed", deadline_ns=1000 * NS))
    # Two programs reached one completed turn; one of them continued.
    f.note_turn_service(1.0)
    f.note_turn_complete(1)
    f.note_turn_complete(1)
    f.note_turn_arrival(1)
    programs = ProgramTable()
    for pid in ("a", "b"):
        programs.on_turn_release(pid, 0, now=0.0)
        programs.on_turn_complete(pid, 0, now=1.0, tool_name="t", context_tokens=8)
    shares = f.compute(programs, 0)
    assert shares is not None
    assert shares["urgent"] > shares["relaxed"]     # less slack, larger share
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_no_shares_before_anything_is_observed():
    f = SagaFairness()
    f.register("a", session("a", tenant="t", deadline_ns=10 * NS))
    programs = ProgramTable()
    programs.on_turn_release("a", 0, now=0.0)
    assert f.compute(programs, 0) is None      # no service, no continuation


# ----------------------------------------------------------------- wiring
def adapter(**kw):
    return UnifiedPolicyAdapter(retention_value="saga-tool-ttl",
                                scheduling_value="fcfs",
                                routing_value="session-affinity",
                                num_instances=1, block_size=16, tau_s=2.0, **kw)


def test_fairness_turns_on_priority_scheduling():
    assert adapter().priority_scheduling is False
    assert adapter(saga_fairness=True).priority_scheduling is True


def test_share_weighted_stamp_lets_a_larger_share_accrue_more_service():
    ad = adapter(saga_fairness=True)
    ad.register_session(session("a", tenant="big", deadline_ns=10 * NS))
    ad.register_session(session("b", tenant="small", deadline_ns=10_000 * NS))
    f = ad._fairness
    f.note_turn_service(1.0)
    f.note_turn_complete(1)
    f.note_turn_complete(1)
    f.note_turn_arrival(1)
    for pid in ("a", "b"):
        ad.programs.on_turn_release(pid, 0, now=0.0)
        ad.programs.on_turn_complete(pid, 0, now=1.0, tool_name="t", context_tokens=8)
    ad.scheduling_exec.turn_complete("a", 5.0)
    ad.scheduling_exec.turn_complete("b", 5.0)
    pa = ad._fair_priority("a", 0)
    pb = ad._fair_priority("b", 0)
    assert pa is not None and pb is not None
    # Equal service, but the bigger share divides further: it runs first.
    assert pa < pb
    assert ad.stats["saga_fair_stamps"] == 2


def test_register_is_inert_when_fairness_is_off():
    ad = adapter()
    ad.register_session(session("a"))      # no tenant, and no complaint
    assert ad._fairness is None
    assert ad._fair_priority("a", 0) is None
