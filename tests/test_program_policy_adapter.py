"""Running the published policies on the program-aware planes.

Two things are being checked here and they are different. First, that the
translation is faithful: a policy sees the same record on this plane that it
sees on the real GPU harness, or it is not the same policy and the arena's
comparison means nothing. Second, that the two adapters agree about what a flag
NAMES -- if `--retention continuum` built one class on one path and another
class on the other, both runs would be self-consistent and two different
numbers would be published under one name.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from serving.core.program_kv import ProgramKVManager               # noqa: E402
from serving.core.program_orchestrator import (                    # noqa: E402
    PlannedTurn, ProgramOrchestrator)
from serving.core.program_policy_adapter import ProgramPolicyAdapter       # noqa: E402
from serving.core.program_scheduler import (                       # noqa: E402
    ProgramBatchScheduler, QueueSnapshot, RunningView)

harness = pytest.importorskip("policies", reason="needs the harness checkout")
import policies                                                     # noqa: E402
from policies import program as h_program                        # noqa: E402
from policies.utils import waste_model as h_waste                 # noqa: E402

#: The three axis names the adapter's five-tuple still carries. They all
#: resolve to the registry now: the classes moved to their paper modules and
#: the registry is the one place that knows where.
h_retention = h_scheduling = h_routing = policies

MODS = (h_retention, h_scheduling, h_routing, h_waste, h_program)


def build(retention=None, scheduling=None, routing=None, **kw):
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 100_000, orch, block_size=16)
    s = ProgramBatchScheduler(0, orch, kv, model="m",
                              max_num_batched_tokens=16384, max_num_seqs=128,
                              enable_prefix_caching=True)
    pa = ProgramPolicyAdapter(orch, MODS, retention=retention,
                              scheduling=scheduling, routing=routing, **kw)
    s.policy = pa
    s.priority_fn, s.admit_fn, s.victim_fn = (
        pa.priority_fn, pa.admit_fn, pa.victim_fn)
    return orch, kv, s, pa


def program(orch, pid="p", tool="bash", turns=3, arrival=0.0):
    orch.define_program(
        pid, tuple(PlannedTurn(i, 100, 120, tool=tool) for i in range(turns)),
        arrival_ts=arrival)
    return orch.get(pid)


# ------------------------------------------------------------- projection

def test_the_record_a_policy_sees_is_the_harness_record():
    """Not a look-alike. A policy reads attribute names, and a near-miss field
    set fails at the first policy that touches the field we omitted."""
    orch, _, _, pa = build()
    program(orch)
    pcb = pa.pcb(orch.get("p"))
    assert isinstance(pcb, h_program.ProgramControlBlock)


def test_gate_original_arrival_learns_before_program_gap_is_cleared():
    orch, kv, _, pa = build(retention='gate', scheduling='gate')
    program(orch, tool='sed', turns=2)
    orch.take_next('p', now=0.)
    orch.place('p', 0)
    orch.on_turn_complete('p', now=1e9, service_s=1., node_id=0)
    pa.observe_arrival(orch.get('p'), 1.2e9)
    pa.observe_arrival(orch.get('p'), 1.3e9)
    assert pa.retention.gap_ema_by_tool['sed'] == pytest.approx(.2)
    orch.on_turn_arrival('p', 1, now=1.2e9)
    pa.on_turn_arrival(kv, orch.get('p'), 1.2e9)
    assert pa.retention.gap_ema_by_tool['sed'] == pytest.approx(.2)
    assert not hasattr(pa.retention, '_last_observed')


def test_every_pcb_field_is_populated_from_plane_state():
    """A field left at its default is a policy input silently set to zero."""
    orch, _, _, pa = build()
    program(orch, turns=2)
    orch.take_next("p", now=0.0)
    orch.place("p", 0)
    orch.on_turn_complete("p", now=5e9, service_s=4.0, node_id=0)
    orch.on_turn_arrival("p", 1, now=9e9)          # engine ns: a 4 s bash gap
    orch.take_next("p", now=9e9)
    pcb = pa.pcb(orch.get("p"))

    assert pcb.program_id == "p"
    assert pcb.turns_completed == 1
    assert pcb.attained_service_s == pytest.approx(4.0)
    assert pcb.kv_instance == 0
    assert pcb.gap_n == 1 and pcb.gap_sum_s == pytest.approx(4.0)
    assert pcb.tool_mean_gap_s == pytest.approx(4.0)


def test_no_second_program_table_is_kept():
    """The old adapter had to own a ProgramTable because the old engine had no
    program record. Keeping one now would put two copies of turns_completed in
    one process, updated by different code."""
    orch, _, _, pa = build()
    assert not any(isinstance(v, h_program.ProgramTable)
                   for v in vars(pa).values())
    program(orch)
    a, b = pa.pcb(orch.get("p")), pa.pcb(orch.get("p"))
    assert a == b and a is not b            # projected, never cached


def test_the_tool_mean_is_cluster_wide_not_per_program():
    """The correction that matters: Continuum keys on the tool. Two programs
    calling `pip` inform each other; a third calling `sed` does not."""
    orch, _, _, pa = build()
    for pid, tool, dur in (("a", "pip", 20.0), ("b", "pip", 20.0),
                           ("c", "sed", 0.1)):
        orch.define_program(pid, (PlannedTurn(0, 10, 20, tool=tool),
                                  PlannedTurn(1, 10, 20)), arrival_ts=0.0)
        orch.take_next(pid, now=0.0)
        orch.on_turn_complete(pid, now=0.0, service_s=0.0, node_id=0)
        orch.on_turn_arrival(pid, 1, now=dur * 1e9)
    assert pa.pcb(orch.get("a")).tool_mean_gap_s == pytest.approx(20.0)
    assert pa.pcb(orch.get("c")).tool_mean_gap_s == pytest.approx(0.1)


# ------------------------------------------------------ the flag mapping

@pytest.mark.parametrize("value", [
    "cache-lru", "evict-always", "ttl", "saga-ttl", "continuum"])
def test_retention_flags_build_the_same_class_on_both_adapters(value):
    """A flag that named different classes on the two paths would publish two
    numbers under one name, each internally consistent."""
    import serving.core.program_policy_adapter as pp
    mine = type(pp.build_retention(value, h_retention, tau_s=2.0))
    expected = {
        "cache-lru": h_retention.CacheLRURetention,
        "evict-always": h_retention.EvictAlwaysRetention,
        "ttl": h_retention.TTLRetention,
        "saga-ttl": h_retention.PressureTTLRetention,
        "continuum": h_retention.ContinuumTTLRetention,
    }[value]
    assert mine is expected


@pytest.mark.parametrize("value", ["fcfs", "program-fcfs", "plas", "continuum"])
def test_scheduling_flags_build_the_same_class(value):
    import serving.core.program_policy_adapter as pp
    mine = type(pp.build_scheduling(value, h_scheduling))
    expected = {
        "fcfs": h_scheduling.FCFSScheduling,
        "program-fcfs": h_scheduling.ProgramFCFSScheduling,
        "plas": h_scheduling.PLASScheduling,
        "continuum": h_scheduling.ContinuumScheduling,
    }[value]
    assert mine is expected


def test_an_unknown_flag_is_refused_not_defaulted():
    import serving.core.program_policy_adapter as pp
    with pytest.raises(ValueError, match="unknown retention value"):
        pp.build_retention("no-such-policy", h_retention)
    with pytest.raises(ValueError, match="unknown scheduling value"):
        pp.build_scheduling("no-such-policy", h_scheduling)


# ------------------------------------------------------------ mechanism

def test_a_pinning_policy_actually_pins():
    """The counter is the point. A retention policy that decides and never
    reaches the KV plane produces a perfectly plausible JCT and no protection."""
    orch, kv, s, pa = build(retention="ttl", tau_s=30.0)
    orch.define_program("p", (PlannedTurn(0, 100, 120, tool="bash"),
                              PlannedTurn(1, 100, 120)), arrival_ts=0.0)
    orch.take_next("p", now=0.0)
    orch.on_turn_complete("p", now=1.0, service_s=1.0, node_id=0)
    pa.on_turn_complete(kv, orch.get("p"), "p:0", now=1.0)
    assert pa.counters["protect"] == 1
    assert orch.get("p").pins, "the policy decided but nothing was pinned"


def test_cache_lru_pins_nothing():
    """The baseline must reach the engine's own LRU untouched, or 'same engine,
    one knob turned' stops being true."""
    orch, kv, s, pa = build(retention="cache-lru")
    orch.define_program("p", (PlannedTurn(0, 100, 120, tool="bash"),),
                        arrival_ts=0.0)
    orch.take_next("p", now=0.0)
    orch.on_turn_complete("p", now=1.0, service_s=1.0, node_id=0)
    pa.on_turn_complete(kv, orch.get("p"), "p:0", now=1.0)
    assert pa.counters["protect"] == 0 and not orch.get("p").pins


def test_a_scheduling_policy_stamps_through_the_plane():
    orch, kv, s, pa = build(scheduling="plas")
    orch.define_program("p", (PlannedTurn(0, 100, 120),), arrival_ts=0.0)
    s.add_request(["p:0", "m", 100, 220, 0.0, 0, list(range(100)), []],
                  session_id="p", sub_request_index=0)
    s.schedule(now=1.0)
    assert s.counters["priority_stamps"] >= 1


def test_a_policy_that_only_stamps_leaves_admission_alone():
    """PLAS overrides priority and nothing else, so the engine's own admission
    and victim code must run unchanged."""
    _, _, _, pa = build(scheduling="plas")
    assert pa.wants_priority and not pa.wants_admit and not pa.wants_victim


# ---------------------------------------------------------- misbehaviour

def test_a_raising_policy_costs_a_fallback_not_a_run():
    """Under policy search, candidates misbehave."""
    orch, kv, s, pa = build(scheduling="plas")

    class Boom:
        def priority(self, pcb, now):
            raise ValueError("bad candidate")

    pa.scheduling = Boom()
    pa.wants_priority = True
    program(orch)
    assert pa.priority_fn(orch.get("p"), 1.0) is None
    assert pa.counters["policy_errors"] == 1


def test_a_declining_policy_is_counted_separately_from_a_broken_one():
    """'No opinion' and 'threw an exception' are different events and a
    mechanism check has to tell them apart."""
    orch, kv, s, pa = build(scheduling="plas")

    class Quiet:
        def priority(self, pcb, now):
            return None

    pa.scheduling = Quiet()
    program(orch)
    pa.priority_fn(orch.get("p"), 1.0)
    assert pa.counters["declined"] == 1 and pa.counters["policy_errors"] == 0


def test_a_victim_index_out_of_range_does_not_crash_the_engine():
    orch, kv, s, pa = build(scheduling="plas")

    class Wild:
        def victim(self, cands, now):
            return 99

    pa.scheduling = Wild()
    pa.wants_victim = True
    program(orch)
    view = RunningView(request_id="p:0", state=orch.get("p"), priority=0,
                       prompt_tokens=100, computed_tokens=120,
                       generated_tokens=20, is_prefill=False)
    assert pa.victim_fn([view], 1.0) is None
    assert pa.counters["policy_errors"] == 1


# ------------------------------------------------------------- routing

def _router(routing, n=4):
    from serving.core.program_router import ProgramRouter
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 100_000, orch, block_size=16)
    scheds = [ProgramBatchScheduler(i, orch, kv, model="m",
                                    max_num_batched_tokens=16384,
                                    max_num_seqs=128) for i in range(n)]
    pa = ProgramPolicyAdapter(orch, MODS, routing=routing, num_instances=n)
    r = ProgramRouter(orch, n, scheds, policy="LOAD",
                      route_fn=pa.route_fn, on_placed=pa.on_placed)
    return orch, r, pa


def test_a_routing_decision_is_a_tuple_and_must_be_unpacked():
    """`RoutingPolicy.route` returns (instance, info). Passing that straight to
    the plane is not a decline it can fall back from -- it is a TypeError
    outside the try block."""
    orch, r, pa = _router("session-affinity")     # stateless: no RR counter
    program(orch, "p")
    orch.place("p", 2)
    out = pa.routing.route(pa.pcb(orch.get("p")), 0.0)
    assert isinstance(out, tuple), "the contract returns (instance, info)"
    assert pa.route_fn(orch.get("p"), [], 0.0) == out[0] == 2


def test_round_robin_actually_spreads():
    orch, r, pa = _router("round-robin")
    chosen = []
    for i in range(8):
        program(orch, f"p{i}")
        chosen.append(r.route(f"p{i}", 0.0))
    assert len(set(chosen)) == 4, f"every turn went to {set(chosen)}"


def test_least_loaded_is_told_about_the_load():
    """The policies keep their OWN per-instance in-flight count and choose from
    it. A host that never tells them leaves every policy deciding from a vector
    of zeros -- least-loaded then means 'instance 0', always, while looking
    exactly like a working policy."""
    orch, r, pa = _router("least-loaded")
    chosen = []
    for i in range(8):
        program(orch, f"p{i}")
        chosen.append(r.route(f"p{i}", 0.0))
    assert pa.routing.inflight != [0, 0, 0, 0], "the load view was never updated"
    assert len(set(chosen)) == 4, f"every turn went to {set(chosen)}"


def test_completion_gives_the_load_back():
    """`on_complete` asserts the count is positive, so a completion announced
    without a matching placement crashes the run."""
    orch, r, pa = _router("least-loaded")
    program(orch, "p")
    inst = r.route("p", 0.0)
    assert pa.routing.inflight[inst] == 1
    kv = ProgramKVManager(0, 1000, orch, block_size=16)
    pa.on_turn_complete(kv, orch.get("p"), "p:0", now=1.0)
    assert pa.routing.inflight[inst] == 0


def test_an_unannounced_completion_does_not_crash():
    orch, r, pa = _router("least-loaded")
    program(orch, "p")
    orch.place("p", 2)                       # placed without going through route
    kv = ProgramKVManager(0, 1000, orch, block_size=16)
    pa.on_turn_complete(kv, orch.get("p"), "p:0", now=1.0)   # must not assert
    assert pa.routing.inflight == [0, 0, 0, 0]


def test_session_affinity_follows_the_program(monkeypatch):
    orch, r, pa = _router("session-affinity")
    program(orch, "p")
    first = r.route("p", 0.0)
    program(orch, "q")
    r.route("q", 0.0)
    assert r.route("p", 1.0) == first, "the program left its own context behind"


# --------------------------------------------------- one object, three axes

def _unified(spec="harness.example_unified:ContextAwarePolicy", n=4):
    from serving.core.program_router import ProgramRouter
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 100_000, orch, block_size=16)
    scheds = [ProgramBatchScheduler(i, orch, kv, model="m",
                                    max_num_batched_tokens=16384,
                                    max_num_seqs=128,
                                    enable_prefix_caching=True)
              for i in range(n)]
    pa = ProgramPolicyAdapter(orch, MODS, unified=spec, num_instances=n)
    for s in scheds:
        s.policy = pa
        s.priority_fn, s.admit_fn, s.victim_fn = (
            pa.priority_fn, pa.admit_fn, pa.victim_fn)
    r = ProgramRouter(orch, n, scheds, policy="LOAD",
                      route_fn=pa.route_fn, on_placed=pa.on_placed)
    return orch, kv, scheds, r, pa


def test_one_object_is_used_for_every_axis_it_implements():
    _, _, _, _, pa = _unified()
    assert pa.axes == {"retention": True, "scheduling": True, "routing": True}
    assert pa.retention is pa.scheduling is pa.routing is pa.unified


def test_the_three_axes_share_one_piece_of_state():
    """The whole distinction. Split across three objects each would recompute
    the valuation, and the moment one used a slightly different rule the knobs
    would pull against each other with no counter showing it."""
    orch, kv, scheds, r, pa = _unified()
    orch.define_program("p", (PlannedTurn(0, 100, 120, tool="bash"),
                              PlannedTurn(1, 100, 120)), arrival_ts=0.0)
    orch.take_next("p", now=0.0)
    orch.place("p", 2)
    orch.on_turn_complete("p", now=1.0, service_s=1.0, node_id=0)

    assert pa.unified._value == {}                 # nothing valued yet
    pa.on_turn_complete(kv, orch.get("p"), "p:0", now=1.0)
    assert "p" in pa.unified._value, "retention never wrote the shared value"

    # scheduling and routing now read what retention wrote
    assert pa.priority_fn(orch.get("p"), 2.0) is not None
    assert r.route("p", 2.0) == 2                  # back to its own context


def test_a_policy_that_implements_two_axes_keeps_the_engine_rule_for_the_third():
    """Writing two of three must be ordinary, not a special case to configure."""
    class TwoAxis:
        def on_turn_complete(self, pcb, request_id, now):
            return ("protect", now + 1.0)
        def priority(self, pcb, now):
            return 7

    orch = ProgramOrchestrator()
    pa = ProgramPolicyAdapter(orch, MODS, unified=TwoAxis(), num_instances=2)
    assert pa.axes == {"retention": True, "scheduling": True, "routing": False}
    assert pa.retention is pa.scheduling
    assert pa.routing is not pa.unified
    assert isinstance(pa.routing, h_routing.LeastLoadedRouting)


def test_an_object_that_decides_nothing_is_refused():
    """Attaching it would replace the engine's rules with a function that
    always declines, while the counters said a policy was running."""
    class Empty:
        pass
    orch = ProgramOrchestrator()
    with pytest.raises(ValueError, match="implements none of the three axes"):
        ProgramPolicyAdapter(orch, MODS, unified=Empty())


def test_inheriting_a_base_method_unchanged_is_not_implementing_it():
    """A subclass that overrides only priority must not be handed the victim
    and admission hooks: the base's versions are 'no opinion', not a rule."""
    class OnlyPriority(h_scheduling.SchedulingPolicy):
        def priority(self, pcb, now):
            return 3

    orch = ProgramOrchestrator()
    pa = ProgramPolicyAdapter(orch, MODS, unified=OnlyPriority())
    assert pa.axes["scheduling"] and not pa.axes["retention"]
    assert pa.wants_priority and not pa.wants_admit and not pa.wants_victim


def test_a_routing_only_policy_gets_its_load_vector_initialised():
    """The routing base owns `inflight` and `_least_loaded`; a unified policy
    that never called its __init__ would have neither."""
    class RouteOnly:
        def route(self, pcb, now):
            return self._least_loaded(), None

    orch = ProgramOrchestrator()
    pa = ProgramPolicyAdapter(orch, MODS, unified=RouteOnly(), num_instances=3)
    assert pa.axes["routing"]
    assert pa.routing.inflight == [0, 0, 0]
    pa.on_placed(1)
    assert pa.routing.inflight == [0, 1, 0]


def test_a_bad_spec_says_what_is_wrong():
    import serving.core.program_policy_adapter as pp
    with pytest.raises(ValueError, match="module:Class"):
        pp.load_unified("harness.example_unified")
    with pytest.raises(AttributeError, match="has no NoSuchClass"):
        pp.load_unified("harness.example_unified:NoSuchClass")
    with pytest.raises(ImportError, match="cannot import"):
        pp.load_unified("harness.no_such_module:X")


def test_an_optional_hook_the_policy_never_wrote_is_not_an_error():
    """The published values inherit no-op defaults, so the method is always
    there. A unified policy subclasses nothing -- it cannot subclass all three
    bases -- so a hook it did not write is simply absent. Counting that as
    misbehaviour reported 141 policy errors on a policy working as written."""
    orch, kv, scheds, r, pa = _unified()
    assert not hasattr(pa.unified, "observe_arrival")
    orch.define_program("p", (PlannedTurn(0, 100, 120, tool="bash"),
                              PlannedTurn(1, 100, 120)), arrival_ts=0.0)
    pa.observe_arrival(orch.get("p"), 1.0)
    assert pa.counters["policy_errors"] == 0


def test_a_hook_that_exists_and_raises_is_still_an_error():
    """The two must stay distinguishable, or 'absent' becomes a way to hide a
    broken candidate."""
    class Boom:
        def priority(self, pcb, now):
            raise ValueError("bad candidate")
    orch = ProgramOrchestrator()
    pa = ProgramPolicyAdapter(orch, MODS, unified=Boom())
    program(orch, "p")
    assert pa.priority_fn(orch.get("p"), 1.0) is None
    assert pa.counters["policy_errors"] == 1


# ------------------------------------------------- the fourth event

def test_a_policy_can_choose_who_gives_up_context_under_pressure():
    """The published policies decide at turn boundaries and say nothing about
    who pays when the pool fills -- that is the valve's LRU, which is the
    engine's opinion, not a policy's. This is the seam that lets a policy have
    one."""
    seen = {}

    class PressureAware:
        def on_turn_complete(self, pcb, request_id, now):
            return ("protect", now + 100.0)
        def on_pressure(self, pcbs, need_tokens, now):
            seen["n"] = need_tokens
            seen["ids"] = [p.program_id for p in pcbs]
            return ["victim"]                      # give up this one

    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 1000, orch, block_size=16)
    pa = ProgramPolicyAdapter(orch, MODS, unified=PressureAware())
    for pid, rng in (("victim", range(0, 320)), ("keeper", range(1000, 1320))):
        orch.on_turn_arrival(pid, 0, now=0.0)
        orch.place(pid, 0)
        kv.commit(pid, list(rng), owner=pid + ":0")
        kv.release(list(rng), owner=pid + ":0")
        kv.pin(pid, pid + ":0", now=0.0, deadline_ts=1e9)

    r = kv.on_pressure(320, now=1.0, policy=pa.pressure_fn)
    assert seen["n"] == 320 and set(seen["ids"]) == {"victim", "keeper"}
    assert not r.declined
    assert kv.probe(list(range(0, 320))) == 0, "the named victim kept its context"
    assert kv.probe(list(range(1000, 1320))) == 320, "the keeper lost its context"
    assert pa.counters["pressure_choices"] == 1


def test_a_policy_without_the_hook_leaves_the_valve_alone():
    """Adding this event must change no published policy's behaviour, or it
    could not be added at all."""
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 1000, orch, block_size=16)
    pa = ProgramPolicyAdapter(orch, MODS, retention="ttl", tau_s=5.0)
    assert not hasattr(pa.retention, "on_pressure")
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.place("p", 0)
    kv.commit("p", list(range(320)), owner="p:0")
    kv.release(list(range(320)), owner="p:0")
    r = kv.on_pressure(320, now=1.0, policy=pa.pressure_fn)
    assert r.declined, "a policy with no opinion must not look like a choice"
    assert pa.counters["policy_errors"] == 0


def test_the_scheduler_passes_no_hook_when_no_policy_is_attached():
    """None here is the engine's LRU; a function that always declines would
    report identically in the counters but is a different code path."""
    orch, kv, s, _ = build()
    s.policy = None
    assert s._pressure_fn() is None


# --------------------------------------------- decision logs for parity

def test_every_knob_the_policy_decides_is_logged(tmp_path):
    """`utils/parity.py` diffs the real harness against the simulator PER KNOB,
    with a different field set for each. The executors produce three logs on
    the old and real paths; these planes do not use executors, but they still
    owe the three logs -- otherwise a run here cannot be decision-compared
    against a GPU at all."""
    import json

    from serving.core.program_router import ProgramRouter
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 100_000, orch, block_size=16)
    scheds = [ProgramBatchScheduler(i, orch, kv, model="m",
                                    max_num_batched_tokens=16384,
                                    max_num_seqs=128) for i in range(2)]
    pa = ProgramPolicyAdapter(orch, MODS,
                              unified="policies.example_unified:ContextAwarePolicy",
                              num_instances=2, log_dir=str(tmp_path))
    r = ProgramRouter(orch, 2, scheds, policy="LOAD",
                      route_fn=pa.route_fn, on_placed=pa.on_placed)

    orch.define_program("p", (PlannedTurn(0, 100, 120, tool="bash"),
                              PlannedTurn(1, 100, 120)), arrival_ts=0.0)
    orch.take_next("p", now=0.0)
    r.route("p", 0.0)
    orch.on_turn_complete("p", now=1.0, service_s=1.0, node_id=0)
    pa.on_turn_complete(kv, orch.get("p"), "p:0", now=1.0)
    pa.priority_fn(orch.get("p"), 2.0)
    pa.finish()

    import policies.utils.parity as parity
    for knob, fields in parity.KNOB_FIELDS.items():
        path = tmp_path / f"{knob}.jsonl"
        assert path.exists(), f"no {knob} log: parity cannot compare this knob"
        rec = json.loads(path.read_text().splitlines()[0])
        missing = [f for f in fields if f not in rec]
        assert not missing, f"{knob} log missing {missing}"


def test_an_axis_the_policy_ignores_writes_no_log(tmp_path):
    """An empty file and 'this policy has no opinion' are different claims."""
    orch = ProgramOrchestrator()
    kv = ProgramKVManager(0, 10_000, orch, block_size=16)
    pa = ProgramPolicyAdapter(orch, MODS, paper="autellix", num_instances=1,
                              log_dir=str(tmp_path))
    program(orch, "p")
    pa.priority_fn(orch.get("p"), 1.0)
    pa.finish()
    assert (tmp_path / "scheduling.jsonl").exists()
    assert not (tmp_path / "routing.jsonl").exists()
    assert not (tmp_path / "retention.jsonl").exists()
