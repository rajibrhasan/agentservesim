"""Program state: the invariants the planes are allowed to rely on.

Pure state, no simulator, no container. Everything here runs in milliseconds.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from serving.core.program_orchestrator import (        # noqa: E402
    Attribution, PlannedTurn, ProgramOrchestrator, ProgramState)


def chain(*specs):
    """Build a path graph the way the loader does: turn i parented on i-1 with
    that turn's tool gap as the edge delay. `specs` are (input, output, gap_ns).
    """
    from serving.core.program_orchestrator import PlannedTurn
    return tuple(
        PlannedTurn(node_id=i, input_toks=inp, output_toks=out,
                    parents=() if i == 0 else ((i - 1, specs[i - 1][2]),))
        for i, (inp, out, _) in enumerate(specs))


@pytest.fixture
def orch():
    return ProgramOrchestrator()


# ------------------------------------------------------------- lifecycle

def test_admit_is_idempotent_and_keeps_the_original_arrival(orch):
    """JCT is measured from first arrival; a re-admission must not reset it."""
    orch.admit_program("p", now=10.0)
    again = orch.admit_program("p", now=99.0)
    assert again.arrival_ts == 10.0
    assert len(orch) == 1


def test_phase_transitions(orch):
    orch.on_turn_arrival("p", 0, now=1.0)
    assert orch.get("p").turn_idx == 0 and not orch.get("p").in_gap
    orch.on_turn_scheduled("p", now=2.0)
    assert not orch.get("p").in_gap      # scheduled, still no gap
    orch.on_turn_complete("p", now=5.0, service_s=3.0, has_more_turns=True)
    assert orch.get("p").in_gap


def test_last_turn_ends_done_not_in_gap(orch):
    orch.on_turn_arrival("p", 0, now=1.0)
    orch.on_turn_complete("p", now=5.0, service_s=3.0, has_more_turns=False)
    p = orch.get("p")
    assert p.end_ts is not None and not p.in_gap
    assert p.in_gap is False and p.gap_started_ts is None


def test_service_accrues_across_turns(orch):
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.on_turn_complete("p", now=2.0, service_s=2.0, has_more_turns=True)
    orch.on_turn_arrival("p", 1, now=3.0)
    orch.on_turn_complete("p", now=6.0, service_s=1.5, has_more_turns=True)
    assert orch.get("p").attained_service_s == pytest.approx(3.5)


def test_records_are_frozen(orch):
    p = orch.admit_program("p", now=0.0)
    with pytest.raises(Exception):
        p.turn_idx = 7           # a holder must not be able to write through


def test_every_transition_replaces_rather_than_mutates(orch):
    before = orch.on_turn_arrival("p", 0, now=1.0)
    after = orch.on_turn_complete("p", now=2.0, service_s=1.0)
    assert before is not after
    assert before.turns_completed == 0        # the old record still reads old
    assert after.turns_completed == 1


def test_admission_records_nothing_on_the_program(orch):
    """Whether a turn is running is the scheduler's `running` list. A copy of
    it here would be a second record of one fact, updated by different code --
    which is the arrangement these planes exist to remove. The EVENT still
    matters: a queue-persistent retention policy releases at admission rather
    than at arrival, and the difference is the whole queue wait."""
    orch.on_turn_arrival("p", 0, now=1.0)
    before = orch.get("p")
    after = orch.on_turn_scheduled("p", now=2.0)
    assert after == before


# ------------------------------------------------------------ gap history

def test_gap_history_folds_in_the_engine_not_a_policy(orch):
    """The bug this prevents: history kept inside a policy depends on every
    host calling an observation hook, and one host did not."""
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.on_turn_complete("p", now=1.0, service_s=1.0,
                          has_more_turns=True)             # gap opens at 1.0
    orch.on_turn_arrival("p", 1, now=4.0)                  # 3.0 s gap
    p = orch.get("p")
    assert p.gap_count == 1
    assert p.mean_gap_s == pytest.approx(3.0)


def test_mean_gap_is_none_before_any_gap(orch):
    """None (no history) must be distinguishable from 0.0 (instant tools):
    a cold-start rule keys on exactly this."""
    orch.on_turn_arrival("p", 0, now=0.0)
    assert orch.get("p").mean_gap_s is None


def test_mean_gap_averages_over_several(orch):
    orch.on_turn_arrival("p", 0, now=0.0)
    for i, (end, nxt) in enumerate([(1.0, 3.0), (4.0, 10.0)]):
        orch.on_turn_complete("p", now=end, service_s=0.5, has_more_turns=True)
        orch.on_turn_arrival("p", i + 1, now=nxt)
    assert orch.get("p").gap_count == 2
    assert orch.get("p").mean_gap_s == pytest.approx((2.0 + 6.0) / 2)


def test_arrival_without_an_open_gap_does_not_count_one(orch):
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.on_turn_arrival("p", 1, now=5.0)
    assert orch.get("p").gap_count == 0


def test_gap_history_survives_across_turns(orch):
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.on_turn_complete("p", now=1.0, service_s=1.0, has_more_turns=True)
    orch.on_turn_arrival("p", 1, now=2.0)
    orch.on_turn_scheduled("p", now=2.5)
    orch.on_turn_complete("p", now=3.0, service_s=0.5, has_more_turns=True)
    assert orch.get("p").gap_count == 1


# ------------------------------------------------------------- residency

def test_place_records_history_and_live_instance(orch):
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.place("p", 0)
    p = orch.get("p")
    assert p.live_instance == 0 and p.instance_history == frozenset({0})


def test_moving_instances_clears_the_stale_footprint(orch):
    """Live context is single-instance. Blocks left behind are ordinary cache
    with no live owner; charging them to the program would double-count it."""
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.place("p", 0)
    orch.set_footprint("p", context_tokens=900, charged_tokens=450)
    orch.grant_pin("p", "r0", now=1.0)
    orch.place("p", 1)
    p = orch.get("p")
    assert p.live_instance == 1
    assert p.instance_history == frozenset({0, 1})
    assert p.context_tokens == 0 and p.charged_tokens == 0
    assert p.pins == ()


def test_replacing_the_same_instance_keeps_the_footprint(orch):
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.place("p", 0)
    orch.set_footprint("p", context_tokens=900, charged_tokens=450)
    orch.place("p", 0)
    assert orch.get("p").context_tokens == 900


def test_context_and_charged_are_recorded_separately(orch):
    """They differ exactly when prefixes are shared: recompute cost wants the
    first, pressure wants the second."""
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.set_footprint("p", context_tokens=1000, charged_tokens=500)
    p = orch.get("p")
    assert (p.context_tokens, p.charged_tokens) == (1000, 500)


def test_on_instance_selects_only_live_residents(orch):
    for name, inst in (("a", 0), ("b", 0), ("c", 1)):
        orch.on_turn_arrival(name, 0, now=0.0)
        orch.place(name, inst)
    assert {p.program_id for p in orch.on_instance(0)} == {"a", "b"}


# ------------------------------------------------------------------ pins

def test_grant_and_release(orch):
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.grant_pin("p", "r0", now=1.0, deadline_ts=3.0)
    assert orch.get("p").is_pinned
    orch.release_pin("p", "r0")
    assert not orch.get("p").is_pinned


def test_regranting_the_same_request_does_not_stack(orch):
    """Otherwise a re-protect leaks a pin that nothing will ever release."""
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.grant_pin("p", "r0", now=1.0)
    orch.grant_pin("p", "r0", now=2.0)
    assert len(orch.get("p").pins) == 1
    assert orch.get("p").pins[0].granted_ts == 2.0


def test_oldest_pin_orders_pressure_victims(orch):
    """A bool and a deadline cannot answer 'which program loses its pin'."""
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.grant_pin("p", "r1", now=5.0)
    orch.grant_pin("p", "r0", now=2.0)
    assert orch.get("p").oldest_pin().request_id == "r0"


def test_expired_pins_are_found_by_deadline(orch):
    orch.on_turn_arrival("a", 0, now=0.0)
    orch.on_turn_arrival("b", 0, now=0.0)
    orch.grant_pin("a", "r0", now=1.0, deadline_ts=2.0)
    orch.grant_pin("b", "r1", now=1.0, deadline_ts=9.0)
    assert [pid for pid, _ in orch.expired_pins(now=5.0)] == ["a"]


def test_a_pin_without_a_deadline_never_expires(orch):
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.grant_pin("p", "r0", now=1.0)
    assert list(orch.expired_pins(now=1e9)) == []


def test_pinned_lists_the_pressure_candidate_set(orch):
    for name in ("a", "b", "c"):
        orch.on_turn_arrival(name, 0, now=0.0)
    orch.grant_pin("a", "r", now=1.0)
    orch.grant_pin("c", "r", now=1.0)
    assert {p.program_id for p in orch.pinned()} == {"a", "c"}


# ------------------------------------------------------------- boundary

def test_no_instance_state_leaked_into_the_record():
    """Queue depth, free blocks and in-flight counts belong to no one program.
    Letting them in turns the record into a global scratchpad."""
    forbidden = {"queue_depth", "free_blocks", "n_inflight", "running",
                 "waiting", "npu_used", "kv_utilization"}
    assert not (forbidden & set(ProgramState.__dataclass_fields__))


def test_no_future_information_in_the_record():
    """Admissible: past-derived aggregates. Inadmissible: anything a deployed
    system could not observe at decision time."""
    forbidden = {"output_toks", "output_len", "gap_duration_s", "true_gap_s",
                 "remaining_turns", "total_turns"}
    assert not (forbidden & set(ProgramState.__dataclass_fields__))


def test_attribution_rule_is_explicit():
    """Shared-prefix charging is part of observable semantics, not a default
    buried in an implementation."""
    assert ProgramOrchestrator().attribution is Attribution.SPLIT
    assert ProgramOrchestrator(Attribution.HOLDER).attribution is Attribution.HOLDER


def test_counters_report_pin_state(orch):
    orch.on_turn_arrival("a", 0, now=0.0)
    orch.on_turn_arrival("b", 0, now=0.0)
    orch.grant_pin("a", "r0", now=1.0)
    orch.grant_pin("a", "r1", now=1.0)
    c = orch.counters()
    assert c == {"programs": 2, "pinned_programs": 1, "pins_held": 2}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------- metrics

def test_jct_is_measured_from_arrival_to_the_last_turn(orch):
    """The benchmark's headline number, from state the orchestrator already
    holds -- it lived in the router only because the router happened to notice
    a chain ending."""
    from serving.core.program_orchestrator import PlannedTurn
    orch.define_program("a", chain((10, 20, 5), (10, 20, 0)), arrival_ts=100.0)
    orch.take_next("a", now=100.0)
    orch.on_turn_complete("a", now=150.0, service_s=1.0, node_id=0)
    orch.take_next("a", now=155.0)
    orch.on_turn_complete("a", now=200.0, service_s=1.0, node_id=1)
    m = orch.workflow_metrics()
    assert m == [{"workflow_id": "a", "arrival_ns": 100, "end_ns": 200,
                  "jct_ns": 100, "num_nodes": 2}]


def test_an_unfinished_program_has_no_metric(orch):
    from serving.core.program_orchestrator import PlannedTurn
    orch.define_program("a", chain((10, 20, 5), (10, 20, 0)), arrival_ts=0.0)
    orch.take_next("a", now=0.0)
    orch.on_turn_complete("a", now=1.0, service_s=1.0, node_id=0)
    assert orch.workflow_metrics() == []      # still owes a turn


def test_metrics_are_ordered_by_arrival(orch):
    from serving.core.program_orchestrator import PlannedTurn
    for name, arr in (("late", 50.0), ("early", 10.0)):
        orch.define_program(name, chain((10, 20, 0)), arrival_ts=arr)
        orch.take_next(name, now=arr)
        orch.on_turn_complete(name, now=arr + 5, service_s=1.0)
    assert [m["workflow_id"] for m in orch.workflow_metrics()] == ["early", "late"]


def test_the_csv_matches_the_format_the_arena_reads(orch, tmp_path):
    """The arena parses this file; the header is a contract, not a detail."""
    from serving.core.program_orchestrator import PlannedTurn
    orch.define_program("a", chain((10, 20, 0)), arrival_ts=0.0)
    orch.take_next("a", now=0.0)
    orch.on_turn_complete("a", now=42.0, service_s=1.0)
    out = tmp_path / "run_workflows.csv"
    assert orch.save_workflow_metrics(str(out)) == 1
    lines = out.read_text().splitlines()
    assert lines[0] == "workflow_id,arrival_ns,end_ns,jct_ns,num_nodes"
    assert lines[1] == "a,0,42,42,1"


def test_summary_matches_the_old_router_shape(orch):
    """Same keys, same percentile interpolation: these numbers have been
    published, so a different formula would silently restate them."""
    for i, (arr, end) in enumerate([(0.0, 10.0), (0.0, 20.0), (0.0, 60.0)]):
        orch.define_program(f"p{i}", chain((10, 20, 0)), arrival_ts=arr)
        orch.take_next(f"p{i}", now=arr)
        orch.on_turn_complete(f"p{i}", now=end, service_s=1.0, node_id=0)
    s = orch.workflow_metrics_summary()
    assert set(s) == {"num_workflows", "jct_mean_ns", "jct_p50_ns",
                      "jct_p90_ns", "jct_p99_ns", "jct_min_ns", "jct_max_ns",
                      "makespan_ns", "workflow_throughput_per_s"}
    assert s["num_workflows"] == 3 and s["jct_p50_ns"] == 20
    assert s["jct_min_ns"] == 10 and s["jct_max_ns"] == 60


def test_summary_is_none_before_anything_completes(orch):
    assert orch.workflow_metrics_summary() is None


def test_first_arrival_floors_at_one(orch):
    """The loop sets its starting clock from this, and zero means 'unset'
    elsewhere -- the old router's floor, kept deliberately."""
    assert orch.first_arrival_ts() == 1
    orch.define_program("a", chain((10, 20, 0)), arrival_ts=0.0)
    assert orch.first_arrival_ts() == 1
    orch.define_program("b", chain((10, 20, 0)), arrival_ts=500.0)
    assert orch.first_arrival_ts() == 1


# -------------------------------------------------- turn release timing

def test_the_next_turn_becomes_runnable_after_the_tool_gap():
    """Release is the orchestrator's, not the router's: when a turn is runnable
    is a fact about the program."""
    from serving.core.program_orchestrator import PlannedTurn
    orch = ProgramOrchestrator()
    orch.define_program("a", chain((10, 20, 100), (30, 40, 0)), arrival_ts=0.0)
    assert [pid for pid, _ in orch.due(0.0)] == ["a"]
    orch.take_next("a", now=0.0)
    assert orch.due(0.0) == []                       # turn in flight
    orch.on_turn_complete("a", now=10.0, service_s=1.0,
                          tool_name="pip", node_id=0)
    assert orch.due(50.0) == []                      # gap still running
    assert [pid for pid, _ in orch.due(110.0)] == ["a"]


def test_next_arrival_is_what_the_loop_fast_forwards_to():
    """Stepping toward a tool gap instead of jumping it is how a five-request
    trace took six million polls."""
    from serving.core.program_orchestrator import PlannedTurn
    orch = ProgramOrchestrator()
    orch.define_program("a", chain((10, 20, 0)), arrival_ts=500.0)
    assert orch.next_arrival_ts() == 500.0


def test_has_pending_keeps_the_run_alive_through_a_gap():
    from serving.core.program_orchestrator import PlannedTurn
    orch = ProgramOrchestrator()
    orch.define_program("a", chain((10, 20, 5), (10, 20, 0)), arrival_ts=0.0)
    orch.take_next("a", now=0.0)
    assert orch.has_pending()                        # one turn still owed
    orch.on_turn_complete("a", now=1.0, service_s=1.0, node_id=0)
    orch.take_next("a", now=6.0)
    assert not orch.has_pending()


def test_completion_infers_whether_more_turns_follow():
    """The trace already said; the caller should not have to assert it."""
    from serving.core.program_orchestrator import PlannedTurn
    orch = ProgramOrchestrator()
    orch.define_program("a", chain((10, 20, 5)), arrival_ts=0.0)
    orch.take_next("a", now=0.0)
    orch.on_turn_complete("a", now=1.0, service_s=1.0)
    assert orch.get("a").end_ts is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_the_in_flight_map_does_not_leak():
    """It is the only state here that is not derived, so whatever it keeps it
    keeps for the length of the run."""
    from serving.core.program_orchestrator import PlannedTurn
    orch = ProgramOrchestrator()
    orch.define_program("p", (PlannedTurn(0, 10, 20, tool="bash"),), arrival_ts=0.0)
    orch.take_next("p", now=0.0)
    orch.on_turn_complete("p", now=1.0, service_s=1.0)       # node_id omitted
    assert orch._in_flight == {}
    assert orch.get("p").tool_name == "bash"


def test_an_ambiguous_fan_out_is_not_guessed():
    """Two turns in flight and no node named: attributing either tool would be
    a coin flip recorded as a measurement."""
    from serving.core.program_orchestrator import PlannedTurn
    orch = ProgramOrchestrator()
    orch.define_program("p", (PlannedTurn(0, 10, 20, tool="bash"),
                              PlannedTurn(1, 10, 20, tool="grep")),
                        arrival_ts=0.0)
    orch.take_next("p", now=0.0, node_id=0)
    orch.take_next("p", now=0.0, node_id=1)
    orch.on_turn_complete("p", now=1.0, service_s=1.0)
    assert orch.get("p").tool_name is None
    orch.on_turn_complete("p", now=2.0, service_s=1.0, node_id=1)
    assert orch.get("p").tool_name == "grep"


# ------------------------------------------------- cluster-wide tool gaps

def test_tool_gaps_are_keyed_by_tool_across_every_program():
    """Continuum keys its rule on the TOOL, not the program. On the board
    trace `pip` averages 19.9 s against `sed` at 0.12 s, so a per-program mean
    over a program's mix of tools is a different policy entirely."""
    from serving.core.program_orchestrator import PlannedTurn
    orch = ProgramOrchestrator()
    for pid, tool, dur in (("a", "pip", 20.0), ("b", "pip", 20.0),
                           ("c", "sed", 0.1)):
        orch.define_program(pid, (PlannedTurn(0, 10, 20, tool=tool),
                                  PlannedTurn(1, 10, 20)), arrival_ts=0.0)
        orch.take_next(pid, now=0.0)
        orch.on_turn_complete(pid, now=0.0, service_s=0.0, node_id=0)
        orch.on_turn_arrival(pid, 1, now=dur)
    assert orch.tool_mean_gap_s("pip") == pytest.approx(20.0)
    assert orch.tool_mean_gap_s("sed") == pytest.approx(0.1)
    assert orch.tool_mean_gap_s("never-run") is None


def test_a_tool_never_seen_is_none_not_zero():
    """A cold-start rule keys on the difference."""
    orch = ProgramOrchestrator()
    assert orch.tool_mean_gap_s("pip") is None
    assert orch.tool_mean_gap_s(None) is None


def test_a_turns_generated_ids_come_from_its_successors_prompt():
    """Under SIM_DERIVE_OUTPUT_IDS=1, a turn's generated ids are read from its
    successor's prompt.

    A turn's output is not invented text: the next prompt is this prompt plus
    what this turn generated plus the tool result, so the ids are already in the
    trace at the offset where this prompt ends. That is right for modelling a
    DEPLOYMENT, and it is why this path exists.

    It is not the default, and must not be used for a run compared against a
    replay: the replay fixes only the output LENGTH and lets vLLM generate its
    own content while the next prompt is replayed from the trace, so the real
    engine gets no such reuse. Measured over 2,823 turns of
    rtx6000_70b_swebench_gate__jps0.02_engine_pinrel/stock, cached_tokens tracks
    the previous turn's PROMPT in 38.3% of turns and prompt + generated in 2.6%.
    See test_generated_ids_are_private_by_default."""
    orch = ProgramOrchestrator()
    prompt = list(range(100, 140))          # turn 0's prompt, 40 tokens
    generated = [900, 901, 902, 903]        # what turn 0 produces
    tool_result = list(range(500, 520))
    turns = (
        PlannedTurn(node_id=0, input_toks=40, output_toks=44,
                    input_hash_ids=tuple(prompt)),
        PlannedTurn(node_id=1, input_toks=64, output_toks=70, parents=((0, 0),),
                    input_hash_ids=tuple(prompt + generated + tool_result)),
    )
    orch.define_program("p", turns, arrival_ts=0.0)

    import serving.core.router as _router
    was = _router._DERIVE_FROM_SUCCESSOR
    _router._DERIVE_FROM_SUCCESSOR = True
    try:
        assert list(orch.generated_ids("p", 0, 4)) == generated
        assert list(orch.generated_ids("p", 0, 2)) == generated[:2]
        # the last turn has no successor: nothing is derivable, so the ids are
        # private -- they still occupy the pool, they just match nobody.
        assert list(orch.generated_ids("p", 1, 4)) != []
        assert orch.generated_ids("p", 0, 0) == ()
    finally:
        _router._DERIVE_FROM_SUCCESSOR = was


def test_generated_ids_are_private_by_default():
    """Default: ids unique to the turn, matchable by nobody.

    The replay we validate against gives the real engine no cross-turn reuse of
    generated content, so deriving would hand the simulator hits its reference
    does not get. The tokens are still numbered and still occupy KV -- only
    their matchability changes."""
    orch = ProgramOrchestrator()
    prompt = list(range(100, 140))
    generated = [900, 901, 902, 903]
    tool_result = list(range(500, 520))
    turns = (
        PlannedTurn(node_id=0, input_toks=40, output_toks=44,
                    input_hash_ids=tuple(prompt)),
        PlannedTurn(node_id=1, input_toks=64, output_toks=70, parents=((0, 0),),
                    input_hash_ids=tuple(prompt + generated + tool_result)),
    )
    orch.define_program("p", turns, arrival_ts=0.0)

    ids = list(orch.generated_ids("p", 0, 4))
    assert len(ids) == 4, "the tokens exist and occupy the pool"
    assert ids != generated, "but they are not the successor's prompt"
    assert not set(ids) & set(prompt + generated + tool_result), "and match nothing"
    assert orch.generated_ids("p", 0, 0) == ()
    assert orch.generated_ids("nosuch", 0, 4) == ()
