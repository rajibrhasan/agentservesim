"""The scheduling plane over the engine's own Request and Batch types.

These build real `Request` objects and assert the plane emits a real `Batch`,
because a parallel type would pass its own tests and then fail at the first
boundary -- `generate_trace` reads q/k lists off the batch to index the profiled
latency tables.

The preemption tests carry the most weight. A simulator that preempts far more
often than the engine it models still produces plausible JCTs; the divergence
shows only in a counter.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from serving.core.program_kv import ProgramKVManager                 # noqa: E402
from serving.core.program_orchestrator import (                      # noqa: E402
    ProgramOrchestrator)
from serving.core.program_scheduler import (                         # noqa: E402
    ProgramBatchScheduler, program_of, turn_of)
from serving.core.request import Batch, Request                      # noqa: E402


def build(capacity=100_000, budget=16384, seqs=128, **kw):
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, capacity, orch)
    sched = ProgramBatchScheduler(0, orch, kv, model="m",
                                  max_num_batched_tokens=budget,
                                  max_num_seqs=seqs, **kw)
    return orch, kv, sched


def req(pid, idx=0, prompt=100, gen=1, arrival=0.0, ids=None):
    """A Request tagged with program identity the way the engine tags it.

    `gen` is how many tokens the turn should produce. The Request field is
    `output`, which in the traces is a CUMULATIVE target -- prompt plus
    generated -- not a count of generated tokens. A five-request trace showed
    input=5714 with output=5734 for twenty tokens. Building requests with
    output=<small count> here is what let a wrong completion rule pass its
    tests and then demand 5,733 decode steps on a real trace.
    """
    r = Request(f"{pid}:{idx}", "m", prompt, prompt + gen, arrival, 0)
    r.session_id = pid
    r.sub_request_index = idx
    if ids is not None:
        r.input_hash_ids = list(ids)
    return r



def add(s, pid, idx=0, prompt=100, gen=1, arrival=0.0, ids=None):
    """Enqueue a request the way the router does: a field list plus identity
    keywords, with the scheduler constructing the Request. Returns it."""
    fields = [f"{pid}:{idx}", "m", prompt, prompt + gen, arrival, 0,
              list(ids) if ids is not None else [], []]
    return s.add_request(fields, session_id=pid, sub_request_index=idx)

def insert(kv, program, tokens):
    kv.radix._current_owner = program
    try:
        kv.radix.insert(list(tokens))
    finally:
        kv.radix._current_owner = None


# --------------------------------------------------- engine types, not new ones

def test_the_plane_defines_no_turn_or_batch_type_of_its_own():
    """Request already carries what a turn needs and everything downstream
    speaks these types; a parallel type drifts at the first field added."""
    import serving.core.program_scheduler as ps
    assert not hasattr(ps, "Turn")
    assert ps.Batch is Batch

def test_program_identity_is_read_off_the_request():
    r = req("prog-a", idx=3)
    assert program_of(r) == "prog-a" and turn_of(r) == 3


def test_a_flat_request_belongs_to_no_program():
    r = Request("x", "m", 10, 1, 0.0, 0)
    assert program_of(r) is None


def test_schedule_emits_the_engine_batch():
    _, _, s = build()
    add(s, "a", prompt=100)
    batch = s.schedule(now=1.0)
    assert isinstance(batch, Batch)
    assert batch.num_prefill == 1 and batch.total_len == 100
    assert batch.q_list == [100] and batch.prefill_q_list == [100]
    assert batch.requests and batch.requests[0].id == "a:0"


def test_batch_is_none_when_nothing_runs():
    _, _, s = build()
    assert s.schedule(now=1.0) is None


def test_decode_fills_the_decode_lists():
    _, _, s = build()
    r = add(s, "a", prompt=100, gen=5)
    b = s.schedule(now=1.0)
    s.add_done(b.batch_id + 1, 0, 1.0)          # completes the prefill chunk
    assert r.num_computed_tokens == 100
    batch = s.schedule(now=2.0)
    assert batch.num_decode == 1 and batch.decode_k_list == [100]
    assert batch.kv_len == 100 and batch.total_len == 1


def test_first_chunk_stamps_the_scheduling_timestamps():
    """first_sched_ts is what the parity work uses instead of queuing_delay,
    which is overwritten per chunk."""
    _, _, s = build()
    r = add(s, "a", prompt=100)
    s.schedule(now=7.0)
    assert r.first_sched_ts == 7.0 and r.queuing_delay == 7.0


# ------------------------------------------------------------- ordering

def test_default_order_is_fcfs_by_arrival():
    _, _, s = build()
    add(s, "b", arrival=5.0)
    add(s, "a", arrival=1.0)
    batch = s.schedule(now=6.0)
    assert [r.session_id for r in batch.requests] == ["a", "b"]


def test_priority_policy_reorders_the_queue():
    _, _, s = build(priority_fn=lambda st, now: 0 if st.program_id == "b" else 9)
    add(s, "a", arrival=1.0)
    add(s, "b", arrival=5.0)
    batch = s.schedule(now=6.0)
    assert [r.session_id for r in batch.requests] == ["b", "a"]


def test_a_partial_ranking_leaves_unstamped_requests_in_arrival_order():
    _, _, s = build(priority_fn=lambda st, now: 0 if st.program_id == "c" else None)
    for name, t in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        add(s, name, arrival=t)
    batch = s.schedule(now=4.0)
    assert [r.session_id for r in batch.requests] == ["c", "a", "b"]


def test_stamps_are_counted():
    _, _, s = build(priority_fn=lambda st, now: 1)
    add(s, "a")
    s.schedule(now=1.0)
    assert s.counters["priority_stamps"] >= 1


# ------------------------------------------------------------ admission

def test_admission_hold_keeps_a_request_waiting_and_counts_it():
    _, _, s = build(admit_fn=lambda st, pressure, now: False)
    add(s, "a")
    assert s.schedule(now=1.0) is None
    assert s.counters["admission_holds"] == 1
    assert s.running == [] and len(s.last_held) == 1


def test_admit_sees_pressure_not_raw_occupancy():
    """vLLM counts cached blocks as free; a policy keyed on utilization must be
    handed the same quantity on both hosts."""
    seen = []
    orch, kv, s = build(
        capacity=1000,
        admit_fn=lambda st, snap, now: seen.append(snap.kv_utilization) or True)
    orch.on_turn_arrival("x", 0, now=0.0)
    orch.place("x", 0)
    insert(kv, "x", range(500))
    add(s, "a")
    s.schedule(now=1.0)
    assert seen and seen[0] == pytest.approx(0.0)


def test_a_raising_admit_policy_does_not_stall_the_queue():
    def boom(*a):
        raise ValueError("bad candidate")

    _, _, s = build(admit_fn=boom)
    add(s, "a")
    assert s.schedule(now=1.0) is not None


def test_scheduled_event_fires_on_admission():
    """Continuum release-at-admission needs one defined moment."""
    orch, _, s = build()
    r = add(s, "a")
    assert r in s.waiting and r not in s.running
    s.schedule(now=1.0)
    assert r in s.running and r not in s.waiting


# ------------------------------------------------------------- batching

def test_running_requests_are_served_before_waiting_ones():
    _, _, s = build(budget=150)
    first = add(s, "a", prompt=100, gen=5)
    b = s.schedule(now=1.0)
    s.add_done(b.batch_id + 1, 0, 1.0)          # prefill completes
    add(s, "b", prompt=100)
    batch = s.schedule(now=3.0)
    assert batch.num_decode == 1                # the running turn goes first


def test_token_budget_is_respected():
    _, _, s = build(budget=150)
    add(s, "a", prompt=100)
    add(s, "b", prompt=100)
    assert s.schedule(now=1.0).total_len <= 150


def test_long_prompt_is_chunked_across_steps():
    _, _, s = build(budget=10_000, long_prefill_token_threshold=64)
    r = add(s, "a", prompt=200)
    b = s.schedule(now=1.0)
    assert b.q_list == [64]
    s.add_done(b.batch_id + 1, 0, 1.0)          # a batch must finish first:
    assert r.num_computed_tokens == 64          # pp_size caps batches in flight
    assert s.schedule(now=2.0).q_list == [64]


def test_sequence_cap_limits_concurrency():
    _, _, s = build(seqs=2)
    for i in range(5):
        add(s, f"p{i}")
    s.schedule(now=1.0)
    assert len(s.running) == 2


# ------------------------------------------------------------ acquire

def test_admission_stops_when_the_pool_is_full_without_preempting():
    """Preempting running work to start NEW work manufactures preemptions the
    real engine never performs."""
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 300, orch)
    s = ProgramBatchScheduler(0, orch, kv, model="m",
                              max_num_batched_tokens=10_000)
    for i in range(6):
        add(s, f"p{i}", prompt=100)
    s.schedule(now=1.0)
    assert s.counters["preemptions"] == 0
    assert len(s.running) < 6 and s.waiting


def test_allocation_locks_so_the_valve_cannot_take_a_live_request():
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 1000, orch)
    s = ProgramBatchScheduler(0, orch, kv, model="m")
    add(s, "a", prompt=200)
    s.schedule(now=1.0)
    assert kv.occupancy()["locked"] > 0


def test_two_requests_in_one_step_cannot_share_an_uncomputed_prefix():
    """Reservation is deliberately conservative within a step.

    Both requests are admitted before either has computed anything, so neither
    prefix exists yet and each must reserve its own space. Letting the second
    "hit" on the first would be a cache hit on tokens nobody has produced --
    the premature sharing the reserve/commit split exists to prevent. Erring
    this way is also the safe direction: under-counting would let the pool
    over-admit.
    """
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 10_000, orch)
    s = ProgramBatchScheduler(0, orch, kv, model="m")
    shared = list(range(500))
    add(s, "a", prompt=500, ids=shared)
    add(s, "b", prompt=500, ids=shared)
    s.schedule(now=1.0)
    assert kv.inflight_tokens() == 1000


def test_a_shared_prefix_is_charged_once_after_the_step_confirms():
    """Sharing begins when the tokens exist. After commit the tree holds one
    copy owned by both programs, not one copy each."""
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 10_000, orch)
    s = ProgramBatchScheduler(0, orch, kv, model="m")
    shared = list(range(500))
    # output > 1 so both are still decoding when the assertion looks: a
    # one-token turn finishes at prefill completion and releases its KV, which
    # is correct but not what this test is about.
    add(s, "a", prompt=500, gen=20, ids=shared)
    add(s, "b", prompt=500, gen=20, ids=shared)
    batch = s.schedule(now=1.0)
    s.add_done(batch.batch_id + 1, 0, 1.0)
    # Whole pages are shared; each turn owns its unpublished four-token tail.
    assert kv.inflight_tokens() == 8
    assert kv.radix.protected_size() == 496
    assert kv.occupancy()["locked"] == 496 + 8
    assert kv.context_tokens("a") == 496 and kv.context_tokens("b") == 496


def test_add_done_advances_progress_so_the_next_step_moves_on():
    """Without this the scheduler re-issues the same chunk forever."""
    _, _, s = build(budget=10_000, long_prefill_token_threshold=64)
    r = add(s, "a", prompt=200)
    batch = s.schedule(now=1.0)
    assert r.num_computed_tokens == 0
    s.add_done(batch.batch_id + 1, 0, 1.0)
    assert r.num_computed_tokens == 64
    assert s.schedule(now=2.0).q_list == [64]    # the NEXT chunk, not a repeat


# ----------------------------------------------------------- preemption

def test_kv_reclaim_is_tried_before_preempting_anything():
    orch, kv, s = build(capacity=1000)
    orch.on_turn_arrival("cold", 0, now=0.0)
    orch.place("cold", 0)
    insert(kv, "cold", range(500))
    add(s, "a")
    s.schedule(now=1.0)
    assert s.ensure_capacity(300, now=2.0) == []
    assert s.counters["preemptions"] == 0


def test_preemption_is_recompute_and_returns_the_request_to_waiting():
    _, _, s = build(capacity=1000)
    r = add(s, "a", prompt=500)
    s.schedule(now=1.0)
    r.num_computed_tokens = 500
    s.ensure_capacity(900, now=2.0)
    assert r in s.waiting and r not in s.running
    assert r.num_computed_tokens == 0
    assert r.n_preempted >= 1


def test_a_preempted_request_goes_to_the_front_not_the_back():
    _, _, s = build(capacity=1000)
    victim = add(s, "a", arrival=0.0)
    s.schedule(now=1.0)
    add(s, "b", arrival=5.0)
    s.ensure_capacity(900, now=6.0)
    assert s.waiting[0] is victim


def test_victim_policy_is_honoured_and_counted():
    _, _, s = build(capacity=1000,
                    victim_fn=lambda cands, now: cands[0].request_id)
    a = add(s, "a", arrival=0.0)
    add(s, "b", arrival=1.0)
    s.schedule(now=2.0)
    s.ensure_capacity(900, now=3.0)
    assert s.counters["victim_overrides"] >= 1 and a in s.waiting


def test_a_raising_victim_policy_falls_back_to_the_default():
    def boom(*a):
        raise ValueError("bad candidate")

    _, _, s = build(capacity=1000, victim_fn=boom)
    add(s, "a")
    s.schedule(now=1.0)
    s.ensure_capacity(900, now=2.0)
    assert s.counters["preemptions"] >= 1
    assert s.counters["victim_overrides"] == 0


# ---------------------------------------------------------- completion

def test_turn_complete_opens_a_gap():
    """A finished request is a finished turn, and nothing has to be told so."""
    orch, _, s = build()
    from serving.core.program_orchestrator import PlannedTurn
    orch.define_program("a", (PlannedTurn(0, 100, 1), PlannedTurn(1, 100, 1)),
                        arrival_ts=0.0)
    r = add(s, "a")
    s.schedule(now=1.0)
    r.end_time = 5_000_000_000
    s.turn_complete(r, now=5.0)
    assert r not in s.running
    p = orch.get("a")
    assert p.in_gap
    assert p.turns_completed == 1


def test_service_is_first_schedule_to_last_token():
    """Prefill included -- the measure the real driver reports. Excluding it
    under-charges long prompts, which is the whole population a service-ranking
    policy is trying to see."""
    orch, _, s = build()
    from serving.core.program_orchestrator import PlannedTurn
    orch.define_program("a", (PlannedTurn(0, 100, 1),), arrival_ts=0.0)
    r = add(s, "a")
    s.schedule(now=1_000_000_000)
    assert r.first_sched_ts == 1_000_000_000
    r.end_time = 5_000_000_000
    s.turn_complete(r, now=5_000_000_000)
    assert orch.get("a").attained_service_s == pytest.approx(4.0)


def test_completion_recovers_the_tool_from_the_turn_it_ran():
    """The engine never carries the tool name, so it cannot hand it back. The
    orchestrator looks up the turn it handed out."""
    orch, _, s = build()
    from serving.core.program_orchestrator import PlannedTurn
    orch.define_program("a", (PlannedTurn(0, 100, 1, tool="bash"),
                              PlannedTurn(1, 100, 1)), arrival_ts=0.0)
    turn = orch.take_next("a", now=0.0)
    assert turn.tool == "bash"
    r = add(s, "a")
    s.schedule(now=1.0)
    r.end_time = 5.0
    s.turn_complete(r, now=5.0)
    assert orch.get("a").tool_name == "bash"


def test_the_plane_keeps_no_program_state():
    _, _, s = build()
    own = {k for k in vars(s) if not k.startswith("_")}
    assert "programs" not in own and "program_table" not in own


def test_counters_use_the_arena_vocabulary():
    _, _, s = build()
    assert {"priority_stamps", "admission_holds", "victim_overrides",
            "preemptions"} <= set(s.counters)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_a_workload_supplied_priority_is_not_mistaken_for_a_policy_stamp():
    """Request.priority defaults to 0 and main can set it from the workload.
    Neither means the scheduling policy has an opinion about this request."""
    _, _, s = build(priority_fn=lambda st, now: None)
    early = add(s, "a", arrival=1.0)
    late = add(s, "b", arrival=2.0)
    late.priority = -100                      # set by the workload, not a policy
    batch = s.schedule(now=3.0)
    assert [r.session_id for r in batch.requests] == ["a", "b"]


def test_completion_waits_for_every_npu():
    """A batch fires on num_npus NPUs and each reports separately. Advancing on
    the first would count every chunk num_npus times -- on every TP>1 cell,
    which is all of them."""
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 100_000, orch)
    s = ProgramBatchScheduler(0, orch, kv, model="m", start_npu=0, num_npus=2)
    r = add(s, "a", prompt=100)
    batch = s.schedule(now=1.0)
    s.add_done(batch.batch_id + 1, 0, 1.0)          # first NPU only
    assert r.num_computed_tokens == 0
    s.add_done(batch.batch_id + 1, 1, 1.0)          # now the last
    assert r.num_computed_tokens == 100


def test_a_repeat_report_from_the_same_npu_is_ignored():
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 100_000, orch)
    s = ProgramBatchScheduler(0, orch, kv, model="m", start_npu=0, num_npus=2)
    r = add(s, "a", prompt=100)
    batch = s.schedule(now=1.0)
    for _ in range(3):
        s.add_done(batch.batch_id + 1, 0, 1.0)
    assert r.num_computed_tokens == 0


def test_add_done_returns_finished_requests():
    _, _, s = build()
    r = add(s, "a", prompt=100, gen=1)
    b = s.schedule(now=1.0)
    _, _, ended = s.add_done(b.batch_id + 1, 0, 5.0)
    assert [x.id for x in ended] == ["a:0"]


def test_prompt_throughput_counts_cached_tokens():
    """vLLM reports prompt throughput including prefix-cache hits; counting only
    computed tokens would understate it on exactly the cache-heavy workloads
    the suite is about."""
    _, _, s = build()
    r = add(s, "a", prompt=100)
    r.prefix_cache_hit = 400
    b = s.schedule(now=1.0)
    prompt_t, _, _ = s.add_done(b.batch_id + 1, 0, 2.0)
    assert prompt_t == 500


def test_preemption_folds_generated_tokens_into_the_prompt():
    """Recompute re-prefills the prompt PLUS what was generated -- those tokens
    are context now and their KV is gone. Zeroing without folding would
    silently shorten the turn."""
    _, _, s = build(capacity=1000)
    r = add(s, "a", prompt=500, gen=50)
    s.schedule(now=1.0)
    r.num_computed_tokens = 520          # prefill done + 20 decoded
    s.ensure_capacity(900, now=2.0)
    assert r.original_input == 521      # includes the already sampled token
    assert r.num_computed_tokens == 0


def test_a_single_token_turn_finishes_at_prefill_and_releases_its_kv():
    """One token is produced by the lm_head pass at prefill completion, so such
    a turn never decodes. Its KV becomes cache, not a leak."""
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 10_000, orch)
    s = ProgramBatchScheduler(0, orch, kv, model="m")
    add(s, "a", prompt=300, gen=1)
    b = s.schedule(now=1.0)
    _, gen_t, ended = s.add_done(b.batch_id + 1, 0, 2.0)
    assert gen_t == 1 and [r.id for r in ended] == ["a:0"]
    occ = kv.occupancy()
    assert occ["locked"] == 0 and occ["cached"] == 288
    assert kv.inflight_tokens() == 0


def test_a_turn_decodes_output_minus_one_times():
    """Measured on a real cell: every turn recorded exactly output-1
    inter-token latencies."""
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 10_000, orch)
    s = ProgramBatchScheduler(0, orch, kv, model="m")
    r = add(s, "a", prompt=100, gen=5)
    steps = 0
    while True:
        b = s.schedule(now=float(steps + 1))
        if b is None:
            break
        _, _, ended = s.add_done(b.batch_id + 1, 0, float(steps + 1))
        steps += 1
        if ended:
            break
    assert steps == 5          # 1 prefill step + 4 decode steps


# -------------------------------------------------------- prefix reuse

def test_a_cached_prefix_is_not_recomputed():
    """Memory is half of what a prefix cache buys. If the cached tokens are
    still prefilled, retaining a prefix costs pool space and saves nothing, and
    every retention policy scores the same."""
    orch, kv, s = build(enable_prefix_caching=True)
    ids = list(range(1, 501))
    a = add(s, "a", prompt=500, ids=ids)
    s.schedule(now=0.0)
    s.add_done(s.inflight[0].batch_id + 1, 0, 1.0)          # a's prefill lands

    b = add(s, "b", prompt=500, ids=ids)                    # same prompt
    s.schedule(now=2.0)
    assert b.prefix_cache_hit > 0
    assert b.chunk_len < 500, "b re-prefilled a prompt that was already cached"


def test_the_last_token_is_always_recomputed():
    """vLLM caps the hit at input-1: a fully-cached prompt still needs one
    token computed to get logits. Without the cap is_prefill() goes false and
    the request is never scheduled at all."""
    orch, kv, s = build(enable_prefix_caching=True)
    ids = list(range(1, 321))
    a = add(s, "a", prompt=320, ids=ids)
    s.schedule(now=0.0)
    s.add_done(s.inflight[0].batch_id + 1, 0, 1.0)

    b = add(s, "b", prompt=320, ids=ids)
    s.schedule(now=2.0)
    assert b.prefix_cache_hit <= 319
    assert b.chunk_len >= 1
    assert b in s.running


def test_no_hit_without_prefix_caching():
    orch, kv, s = build(enable_prefix_caching=False)
    ids = list(range(1, 501))
    add(s, "a", prompt=500, ids=ids)
    s.schedule(now=0.0)
    s.add_done(s.inflight[0].batch_id + 1, 0, 1.0)
    b = add(s, "b", prompt=500, ids=ids)
    s.schedule(now=2.0)
    assert b.prefix_cache_hit == 0 and b.chunk_len == 500


def test_hit_rate_is_counted_once_per_request():
    """A chunked prefill re-matches on every attempt. Counting each one
    inflates the denominator by how often a request happened to be re-ranked,
    which is not a property of the cache at all."""
    orch, kv, s = build(enable_prefix_caching=True)
    ids = list(range(1, 501))
    a = add(s, "a", prompt=500, ids=ids)
    for _ in range(4):
        s._prefix_match(a)
    assert s.return_prefix_info() == ((500, 0), (0, 0))


def test_the_report_is_requested_and_hit():
    orch, kv, s = build(enable_prefix_caching=True)
    ids = list(range(1, 501))
    add(s, "a", prompt=500, ids=ids)
    s.schedule(now=0.0)
    s.add_done(s.inflight[0].batch_id + 1, 0, 1.0)
    b = add(s, "b", prompt=500, ids=ids)
    s.schedule(now=2.0)
    (req_toks, hit_toks), cpu = s.return_prefix_info()
    assert req_toks == 1000 and 0 < hit_toks <= 499 and cpu == (0, 0)


# ------------------------------------------------------------ the artifact

def _rows(s, tmp):
    import csv
    s.save_output(str(tmp), is_append=False)
    return list(csv.DictReader(open(str(tmp))))


def test_the_output_column_is_generated_tokens(tmp_path):
    """`output` is cumulative internally -- prompt plus generated. Writing it
    raw reports 209 where 20 is meant, and the file is what the arena reads."""
    orch, kv, s = build()
    r = add(s, "a", prompt=189, gen=20)
    s.schedule(now=0.0)
    for _ in range(40):
        if not s.inflight:
            s.schedule(now=1.0)
        if not s.inflight:
            break
        s.add_done(s.inflight[0].batch_id + 1, 0, 1.0)
    assert r in s.done
    row = _rows(s, tmp_path / "out.csv")[0]
    assert row["input"] == "189" and row["output"] == "20"


def test_times_are_written_as_whole_nanoseconds(tmp_path):
    """`17001004787.0` and `17001004787` are not the same nine bytes, and
    anything parsing the column with int() sees only one of them."""
    orch, kv, s = build()
    r = add(s, "a", prompt=16, gen=1, arrival=17001004787.0)
    s.schedule(now=17001004787.0)
    s.add_done(s.inflight[0].batch_id + 1, 0, 17001004800.0)
    row = _rows(s, tmp_path / "out.csv")[0]
    assert row["arrival"] == "17001004787"
    assert "." not in row["end_time"]


def test_inter_token_latency_is_recorded(tmp_path):
    """n generated tokens give n-1 intervals; an empty list is a plane that
    never measured them."""
    orch, kv, s = build()
    r = add(s, "a", prompt=16, gen=5)
    t = 0.0
    for _ in range(40):
        if not s.inflight:
            s.schedule(now=t)
        if not s.inflight:
            break
        t += 10.0
        s.add_done(s.inflight[0].batch_id + 1, 0, t)
    assert r in s.done
    assert len(r.itl) == 4, r.itl


def test_decodes_take_the_budget_before_running_prefills():
    """vLLM v1's order under chunked prefill. A decode costs one token; a
    mid-chunk prefill taking the budget first drops decodes out of the batch,
    and the batch's shape is what indexes the profiled latency table."""
    orch, kv, s = build(budget=64)
    p = add(s, "p", prompt=4000, gen=1)          # will chunk
    d = add(s, "d", prompt=16, gen=50)
    s.schedule(now=0.0)
    s.add_done(s.inflight[0].batch_id + 1, 0, 1.0)   # d finishes prefill
    while s.inflight:
        s.add_done(s.inflight[0].batch_id + 1, 0, 2.0)
    b = s.schedule(now=3.0)
    assert not d.is_prefill() and p.is_prefill()
    assert d in b.requests, "the decode was squeezed out by a mid-chunk prefill"


# --------------------------------------------------- decode occupies KV

def test_a_generated_token_occupies_kv():
    """A five-hundred-token answer is five hundred tokens of cache. A key that
    stops at the prompt makes the pool look emptier than it is, so nothing is
    ever tight and no retention policy can separate from any other."""
    orch, kv, s = build(capacity=100_000, enable_prefix_caching=True)
    r = add(s, "a", prompt=320, gen=500, ids=list(range(320)))
    r.num_computed_tokens = 320
    before = len(s._key(r, r.num_computed_tokens))
    r.num_computed_tokens = 820
    assert len(s._key(r, r.num_computed_tokens)) == 820 > before


def test_generated_tokens_are_private_to_their_request():
    """Two requests that produced the same text are two sequences with
    different prefixes behind them; vLLM holds two blocks, not one."""
    orch, kv, s = build(enable_prefix_caching=True)
    a = add(s, "a", prompt=32, gen=100, ids=list(range(32)))
    b = add(s, "b", prompt=32, gen=100, ids=list(range(32)))
    a.num_computed_tokens = b.num_computed_tokens = 132
    ka, kb = s._key(a, 132), s._key(b, 132)
    assert ka[:32] == kb[:32], "the shared prompt must still share"
    assert not set(ka[32:]) & set(kb[32:]), "generated tokens must not share"


def test_the_two_private_namespaces_cannot_collide():
    """One request has real prompt ids, another has none and gets a synthetic
    prompt. Neither may land on the other's generated tokens."""
    orch, kv, s = build(enable_prefix_caching=True)
    withids = add(s, "a", prompt=32, gen=64, ids=list(range(32)))
    noids = add(s, "b", prompt=32, gen=64)
    withids.num_computed_tokens = noids.num_computed_tokens = 96
    assert not set(s._key(withids, 96)) & set(s._key(noids, 96))


def test_decode_pressure_is_visible_to_the_pool():
    """The end of the chain: occupancy must actually rise as a turn decodes."""
    orch, kv, s = build(capacity=4096, enable_prefix_caching=True)
    add(s, "a", prompt=64, gen=2000, ids=list(range(64)))
    s.schedule(now=0.0)
    start = kv.occupancy()["locked"] + kv.occupancy()["cached"]
    for i in range(60):
        if not s.inflight:
            s.schedule(now=float(i))
        if not s.inflight:
            break
        s.add_done(s.inflight[0].batch_id + 1, 0, float(i))
    end = kv.occupancy()["locked"] + kv.occupancy()["cached"]
    assert end > start, f"pool did not grow while decoding: {start} -> {end}"


def test_a_turn_too_large_for_the_pool_is_refused_not_hung():
    """No reclaim frees enough and there is nobody to preempt but itself, so
    the scheduler would return no batch forever. That presents as a hang, which
    is the hardest possible way to learn the pool is too small."""
    orch, kv, s = build(capacity=1_000)
    with pytest.raises(ValueError, match="No reclaim or preemption can make this fit"):
        add(s, "a", prompt=900, gen=200)


def test_a_turn_that_exactly_fills_the_pool_is_allowed():
    orch, kv, s = build(capacity=1_000)
    r = add(s, "a", prompt=800, gen=200)
    assert r in s.waiting


# ------------------------------- preemption through the LIVE path
# The tests above drive `ensure_capacity`, which nothing in serving/ calls.
# These drive `schedule()`, which is the only way preemption happens for real.

def _pair(pool=1200, prompt=500, gen=300):
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, pool, orch, block_size=16)
    s = ProgramBatchScheduler(0, orch, kv, model="m", max_num_batched_tokens=16384,
                              max_num_seqs=128, enable_prefix_caching=True)
    reqs = []
    for pid in ("a", "b"):
        base = abs(hash(pid)) % 9000
        reqs.append(s.add_request(
            [f"{pid}:0", "m", prompt, prompt + gen, 0.0, 0,
             list(range(base, base + prompt)), []],
            session_id=pid, sub_request_index=0))
    return orch, kv, s, reqs


def _drive(s, kv, limit=8000):
    t = 0.0
    for step in range(limit):
        b = s.schedule(now=t)
        if b is None:
            return step, False
        t += 1.0
        s.add_done(b.batch_id + 1, 0, t)
        if len(s.done) == 2:
            return step, True
    return limit, False


def test_two_requests_that_cannot_both_fit_still_both_finish():
    """They need 1,600 tokens between them and the pool holds 1,200.

    Asserts the two things that must hold: both turns complete their full
    output, and the pool is never over-subscribed. NOT that a preemption
    happens -- it used to, and that was the accounting bug: `allocate` sized
    the reservation against a prefix nothing was holding, the pool went over
    capacity, `free` clamped to zero and the scheduler preempted its way out.
    With the hold in place it packs to exactly 1,200 and nobody yields.
    """
    _, kv, s, reqs = _pair()
    step, done = _drive(s, kv)
    assert done, f"neither finished after {step} steps"
    for r in reqs:
        assert r.output <= r.num_computed_tokens + 1, f"{r.id} stopped short"


def test_the_pool_is_never_over_subscribed():
    """The tree must never hold more than the pool. Over-subscription is
    invisible while it happens -- `free` clamps at zero -- and surfaces much
    later as a stall with memory that nothing can reclaim."""
    _, kv, s, _ = _pair()
    worst = 0
    t = 0.0
    for _ in range(8000):
        b = s.schedule(now=t)
        if b is None:
            break
        t += 1.0
        s.add_done(b.batch_id + 1, 0, t)
        occ = kv.occupancy()
        worst = max(worst, occ["locked"] + occ["pinned"] + occ["cached"])
        if len(s.done) == 2:
            break
    assert worst <= kv.capacity_tokens, (
        f"pool over-subscribed by {worst - kv.capacity_tokens} tokens")


def test_a_commit_that_overshoots_its_reservation_gives_the_difference_back():
    """`allocate` reserves the tokens the STEP computes; `commit` publishes the
    whole key. The two agree only while the matched prefix stays resident, and
    the scheduler is what keeps it resident -- it holds the prefix it credited
    before anything reclaims (`_acquire`).

    This is the backstop for when it does not: a prefix taken out from under a
    live reservation by something outside that protocol. The overshoot is
    corrected where it is observable, so the pool is never LEFT over capacity,
    because over-subscription is invisible while it happens (`free` clamps at
    zero) and surfaces much later as a stall.
    """
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 1_024, orch, block_size=16)
    orch.on_turn_arrival("a", 0, now=0.0)
    orch.place("a", 0)
    kv.commit("a", list(range(640)), owner="a:0")
    kv.release(list(range(640)), owner="a:0")

    orch.on_turn_arrival("b", 0, now=0.0)
    orch.place("b", 0)
    key = list(range(640)) + list(range(9000, 9128))
    kv.allocate("b", 128)                 # one chunk, atop the shared 640
    kv.evict("a")                         # ...which now disappears
    kv.commit("b", key, owner="b:0")

    occ = kv.occupancy()
    used = occ["locked"] + occ["pinned"] + occ["cached"]
    assert used <= kv.capacity_tokens, (
        f"pool left {used - kv.capacity_tokens} tokens over capacity")


def test_a_preempted_victim_does_not_preempt_its_preemptor_back():
    """The running list is a snapshot. Without skipping requests preempted
    during the loop, the victim keeps its turn and takes the memory straight
    back -- 6,000 steps, zero completions."""
    _, kv, s, (a, b) = _pair()
    _drive(s, kv)
    assert a.n_preempted == 0 or b.n_preempted == 0, \
        "both sides were preempted: they swapped memory instead of progressing"


def test_preemption_returns_the_memory_it_took():
    """`evict` refuses locked nodes, so a preemption that does not first drop
    the victim's own lock frees nothing and shrinks the pool permanently."""
    _, kv, s, _ = _pair()
    _drive(s, kv)
    assert kv.occupancy()["locked"] == 0
    assert kv._lock_holder == {}


def test_the_admission_gate_sees_the_queue_not_its_neighbours():
    """A gate that could read other programs would be using state the real
    gateway has no access to, and its decision could not be reproduced there."""
    seen = []
    orch, kv, s = build(capacity=10_000,
                        admit_fn=lambda st, snap, now: seen.append(snap) or True)
    add(s, "a", prompt=320, gen=5, ids=list(range(320)))
    s.schedule(now=0.0)
    assert seen
    snap = seen[0]
    assert snap.prompt_tokens == 320
    assert snap.n_waiting >= 1 and snap.kv_free_tokens > 0
    assert not any(f.startswith("program") for f in vars(snap))


def test_the_victim_rule_sees_token_counts():
    """The published victim rules rank by held context, so a candidate list of
    bare ids cannot express them."""
    seen = []
    def pick(cands, now):
        seen.extend(cands)
        return cands[0].request_id
    orch, kv, s = build(capacity=1000, victim_fn=pick)
    r = add(s, "a", prompt=500, gen=50)
    s.schedule(now=1.0)
    r.num_computed_tokens = 520
    s.ensure_capacity(900, now=2.0)
    assert seen and seen[0].computed_tokens == 520
    assert seen[0].prompt_tokens == 500 and seen[0].generated_tokens == 20


# ------------------------------------- preemption vs the in-flight batch

def test_a_preempted_request_leaves_the_batch_it_was_computing_in():
    """Otherwise the batch reports later and `add_done` advances the victim
    anyway -- undoing the preemption from inside a step that already ended, and
    re-committing the KV preemption just released."""
    orch, kv, s = build(capacity=1000)
    r = add(s, "a", prompt=500, gen=50)
    b = s.schedule(now=1.0)
    assert r in b.requests
    r.num_computed_tokens = 500
    s.ensure_capacity(900, now=2.0)          # preempts r
    assert r in s.waiting and r not in s.running
    assert r not in b.requests, "the victim is still in the in-flight batch"


def test_a_stale_batch_cannot_resurrect_a_preempted_request():
    """The end-to-end shape of the deadlock: the victim came back as a DECODE
    sitting in the waiting queue, which nothing could schedule."""
    orch, kv, s = build(capacity=1000)
    r = add(s, "a", prompt=500, gen=50)
    b = s.schedule(now=1.0)
    r.num_computed_tokens = 500
    s.ensure_capacity(900, now=2.0)
    ncp_after_preempt = r.num_computed_tokens
    s.add_done(b.batch_id + 1, 0, 3.0)       # the batch reports afterwards
    assert r.num_computed_tokens == ncp_after_preempt
    assert r.is_prefill(), "the victim came back as a decode in the queue"


def test_a_decode_in_the_waiting_queue_is_schedulable():
    """`add_decode` puts one there on the prefill/decode path. `_chunk` sizes
    anything past prefill at zero, so the admission loop must not use it to
    decide whether the request can run."""
    orch, kv, s = build(capacity=10_000)
    # The orchestrator is cluster-wide, so a request handed over from a prefill
    # instance is already a program it knows.
    orch.on_turn_arrival("a", 0, now=0.0)
    orch.place("a", 0)
    r = req("a", prompt=64, gen=40, ids=list(range(64)))
    r.num_computed_tokens = 64               # prefill already done elsewhere
    assert not r.is_prefill()
    s.add_decode(r)
    assert r in s.waiting
    b = s.schedule(now=1.0)
    assert b is not None and r in b.requests, "a waiting decode was never admitted"


def test_a_request_preempted_after_being_scheduled_leaves_the_batch():
    """`scheduled` is a local list `_preempt` cannot reach. A request appended
    to it can still be chosen as a victim by a later request in the same pass,
    and building the batch from it anyway puts a preempted request back into a
    batch -- the same ghost as the in-flight case, one level earlier."""
    orch, kv, s = build(capacity=1000)
    victim = add(s, "a", prompt=400, gen=50, ids=list(range(400)))
    s.schedule(now=1.0)
    victim.num_computed_tokens = 420
    other = add(s, "b", prompt=400, gen=50, ids=list(range(5000, 5400)))
    b = s.schedule(now=2.0)
    if victim not in s.running:                 # it was preempted
        assert b is None or victim not in b.requests, \
            "a preempted request was built into the batch"


def test_the_victim_is_the_most_recently_admitted_not_the_latest_to_arrive():
    """vLLM v1 preempts `running[-1]` -- the most recently ADMITTED request,
    which has computed the least and so is the cheapest to undo. The old plane
    spells it `max(running_in, key=r.admit_seq)` (scheduler.py:613).

    The two rules disagree exactly for a request that was preempted and
    re-admitted: it keeps its original arrival but takes a fresh admit_seq.
    Ranking by arrival treats it as senior and preempts a longer-running
    request instead -- throwing away more computed tokens, which are then
    re-prefilled, which raises pressure, which preempts again.
    """
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 4_000, orch, block_size=16)
    s = ProgramBatchScheduler(0, orch, kv, model="m",
                              max_num_batched_tokens=16384, max_num_seqs=128,
                              enable_prefix_caching=True)
    # `late` arrived last but was admitted FIRST; `early` arrived first but was
    # re-admitted after a preemption, so it is the freshest in the running set.
    late = s.add_request(["late:0", "m", 100, 200, 100.0, 0,
                          list(range(10_000, 10_100)), []],
                         session_id="late", sub_request_index=0)
    early = s.add_request(["early:0", "m", 100, 200, 0.0, 0,
                           list(range(20_000, 20_100)), []],
                          session_id="early", sub_request_index=0)
    s.waiting.clear()
    s.running.extend([late, early])
    late.admit_seq, early.admit_seq = 1, 2      # early re-admitted after late

    victim = s._choose_victim(now=200.0)
    assert victim is early, (
        f"preempted {victim.id}: ranking by arrival takes the longest-running "
        f"request; vLLM takes the most recently admitted")
