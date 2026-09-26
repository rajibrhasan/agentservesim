"""SAGA prefetch: recompute a paused context before its tool returns.

The engine integration prefetches by recomputation (a one-token generation
over the session's prompt, pinned until the result is due); the simulator does
the same. These check the timing decision and that the synthetic request is
submitted to the right instance and never counted as a program turn.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.request import Request
from serving.core.saga_prefetch import SagaPrefetcher
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
from policies.program import ProgramTable

MODEL = 'meta-llama/Llama-3.1-8B'
NS = 1_000_000_000


def finished(rid, tokens):
    r = Request(rid, MODEL, len(tokens), 1, 0, 0,
                input_hash_ids=list(tokens), output_hash_ids=[10 ** 9])
    r.num_computed_tokens = len(tokens)
    return r


def never_resident(*a):
    return False


def gapped(programs, pid, tool="grep", started=0.0):
    programs.on_turn_release(pid, 0, now=started)
    programs.on_turn_complete(pid, 0, now=started, tool_name=tool, context_tokens=64)
    return programs


# ---------------------------------------------------------------- refusals
def test_prefetch_needs_the_learned_gap_estimator():
    try:
        SagaPrefetcher(None)
    except ValueError as e:
        assert "saga-tool-ttl" in str(e)
    else:
        raise AssertionError("built without a gap estimate")


# ------------------------------------------------------------------ timing
def test_nothing_is_due_before_the_margin():
    p = SagaPrefetcher(lambda tool: 10.0, margin_s=0.5)
    progs = gapped(ProgramTable(), "s0")
    p.note_gap_start("s0", finished(1, range(64)), 0, 0)
    assert p.due(progs, 5 * NS, never_resident) == []      # 5 s into a 10 s gap
    due = p.due(progs, int(9.6 * NS), never_resident)
    assert [d[0] for d in due] == ["s0"]                   # inside the margin


def test_a_resident_context_is_not_recomputed():
    p = SagaPrefetcher(lambda tool: 1.0, margin_s=0.5)
    progs = gapped(ProgramTable(), "s0")
    p.note_gap_start("s0", finished(1, range(64)), 0, 0)
    assert p.due(progs, NS, lambda *a: True) == []
    assert p.stats["skipped_resident"] == 1
    assert "s0" not in p.parked          # and it is not retried


def test_the_real_successor_cancels_the_prefetch():
    p = SagaPrefetcher(lambda tool: 1.0)
    p.note_gap_start("s0", finished(1, range(64)), 0, 0)
    p.note_issued("s0", 123)
    assert p.in_flight == {"s0": 123}
    p.note_turn_arrival("s0")
    assert p.in_flight == {} and p.parked == {}


def test_one_prefetch_in_flight_at_a_time():
    p = SagaPrefetcher(lambda tool: 1.0, margin_s=10.0, max_in_flight=1)
    progs = ProgramTable()
    for pid in ("s0", "s1"):
        gapped(progs, pid)
        p.note_gap_start(pid, finished(1, range(64)), 0, 0)
    assert len(p.due(progs, NS, never_resident)) == 1
    p.note_issued("s0", 1)
    assert p.due(progs, NS, never_resident) == []


def test_the_last_token_is_excluded_like_the_cache_insert():
    p = SagaPrefetcher(lambda tool: 1.0)
    p.note_gap_start("s0", finished(1, range(64)), 0, 0)
    ids, _, _ = p.parked["s0"]
    assert len(ids) == 64      # 64 input + 1 output, minus the trailing token


# ----------------------------------------------------------------- wiring
def adapter(**kw):
    return UnifiedPolicyAdapter(retention_value="saga-tool-ttl",
                                scheduling_value="fcfs",
                                routing_value="session-affinity",
                                num_instances=1, block_size=16, tau_s=2.0, **kw)


def test_prefetch_is_off_unless_asked():
    ad = adapter()
    assert ad._prefetcher is None
    assert ad.prefetch_tick(NS) == ()


def test_submitted_to_the_instance_that_last_served_the_program():
    from serving.core.scheduler import Scheduler
    ad = adapter(saga_prefetch=True, saga_prefetch_margin_s=10.0)
    scheds = [Scheduler(MODEL, i, i, 128, 64, 1, 1, 1, 80, 80, 0, None, 16, 16,
                        0, False, True, False, None, None, True) for i in (0, 1)]
    ad.attach_schedulers(scheds)
    ad._instance_of["s0"] = 1
    gapped(ad.programs, "s0")
    ad._prefetcher.note_gap_start("s0", finished(5, range(64)), 1, 0)
    issued = ad.prefetch_tick(NS)
    assert len(issued) == 1 and issued[0][2] == 1
    assert scheds[0].request == []
    req = scheds[1].request[0]
    assert req.prefetch_of == "s0" and req.output == 1
    assert ad.prefetch_stats()["issued"] == 1


def test_a_prefetch_is_never_counted_as_a_program_turn():
    """__main__ filters them out of finished_reqs; the marker is what it uses."""
    r = Request(1, MODEL, 8, 1, 0, 0, input_hash_ids=[1], output_hash_ids=[2])
    assert r.prefetch_of is None
    r.prefetch_of = "s0"
    assert [x for x in [r] if x.prefetch_of is None] == []
