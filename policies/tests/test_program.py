"""Unit tests for the Program Control Block and its table.

The last test is the one that matters for the design claim: no policy may
keep per-program state of its own. It is checked mechanically rather than
asserted in a comment, because a policy that quietly caches program state
would still pass every other test here while breaking the property the
whole abstraction rests on.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from policies.utils.kv_control import RecordingKVControl
from policies.program import ProgramControlBlock, ProgramTable
from policies import (
    CacheLRURetention,
    MinWasteRetention,
    RetentionExecutor,
    TTLRetention,
)
from policies import (
    LeastLoadedRouting,
    RoundRobinRouting,
    RoutingExecutor,
    SessionAffinityRouting,
)
from policies import (
    FCFSScheduling,
    PLASScheduling,
    ProgramFCFSScheduling,
    SchedulingExecutor,
)
from policies.utils.waste_model import WasteProfile

PROFILE = WasteProfile(a=0.0279, c=15.4, S=384)


def test_pcb_is_frozen():
    pcb = ProgramControlBlock(program_id="p0")
    try:
        pcb.turn_idx = 3
    except Exception:
        return
    raise AssertionError("PCB must be immutable; policies would mutate it")


def test_release_sets_position_and_first_arrival_only_once():
    t = ProgramTable()
    p = t.on_turn_release("p0", 0, now=100.0, instance=2, context_tokens=500)
    assert (p.arrival_ts, p.turn_idx, p.kv_instance, p.context_tokens) == (
        100.0, 0, 2, 500)
    p = t.on_turn_release("p0", 1, now=180.0)
    assert p.arrival_ts == 100.0 and p.turn_idx == 1


def test_completion_accumulates_service_and_sets_tool_state():
    t = ProgramTable()
    t.on_turn_release("p0", 0, now=100.0)
    p = t.on_turn_complete("p0", 0, now=104.0, service_s=4.0, tool_name="pytest")
    assert p.attained_service_s == 4.0 and p.turns_completed == 1
    assert p.in_gap and p.tool_name == "pytest"
    assert p.gap_elapsed_s(109.0) == 5.0


def test_completion_is_idempotent_and_order_independent():
    # Scheduling reports service, retention reports the tool, in either
    # order; the accumulating fields must land exactly once.
    for order in ("service_first", "tool_first"):
        t = ProgramTable()
        t.on_turn_release("p0", 0, now=100.0)
        if order == "service_first":
            t.on_turn_complete("p0", 0, now=104.0, service_s=4.0)
            p = t.on_turn_complete("p0", 0, now=104.0, tool_name="grep")
        else:
            t.on_turn_complete("p0", 0, now=104.0, tool_name="grep")
            p = t.on_turn_complete("p0", 0, now=104.0, service_s=4.0)
        assert p.attained_service_s == 4.0, order
        assert p.turns_completed == 1, order
        assert p.tool_name == "grep", order


def test_release_after_gap_clears_tool_state():
    t = ProgramTable()
    t.on_turn_release("p0", 0, now=100.0)
    t.on_turn_complete("p0", 0, now=104.0, service_s=1.0, tool_name="pytest")
    p = t.on_turn_release("p0", 1, now=160.0)
    assert not p.in_gap and p.tool_name is None and p.gap_started_ts is None
    assert p.gap_elapsed_s(170.0) is None


def test_memory_pressure_refreshes_residency():
    t = ProgramTable()
    t.on_turn_release("p0", 0, now=100.0, context_tokens=1000)
    t.note_retention("p0", "protect", deadline_ts=160.0, request_id="p0:0")
    assert t.get("p0").kv_protected
    # The valve broke this program's protection under pressure.
    p = t.on_memory_pressure("p0", kv_protected=False, context_tokens=0)
    assert not p.kv_protected and p.kv_deadline_ts is None and p.context_tokens == 0


def test_programs_are_independent():
    t = ProgramTable()
    t.on_turn_release("pA", 0, now=100.0)
    t.on_turn_release("pB", 0, now=101.0)
    t.on_turn_complete("pA", 0, now=105.0, service_s=5.0)
    assert t.get("pA").attained_service_s == 5.0
    assert t.get("pB").attained_service_s == 0.0
    assert len(t) == 2


def test_shared_table_across_the_three_executors():
    # The routing pin, the priority stamp, and the retention valuation
    # must read one record, not three copies of it.
    programs = ProgramTable()
    kv = RecordingKVControl(blocks_per_request=4)
    rx = RoutingExecutor(policy=SessionAffinityRouting(2), programs=programs)
    sx = SchedulingExecutor(policy=PLASScheduling(), programs=programs)
    tx = RetentionExecutor(policy=TTLRetention(tau_s=60.0), kv=kv,
                           programs=programs)

    assert rx.route("p0", 0, now=100.0) == 0
    assert sx.stamp("p0", 0, now=100.0) == 0        # no service attained yet
    tx.turn_complete("p0", 0, "p0:0", "pytest", now=102.0, context_tokens=900)
    sx.turn_complete("p0", 2.0, turn_idx=0)
    rx.turn_complete(0)

    pcb = programs.get("p0")
    assert pcb.kv_instance == 0                     # routing wrote residency
    assert pcb.attained_service_s == 2.0            # scheduling wrote service
    assert pcb.tool_name == "pytest" and pcb.kv_protected  # retention wrote both

    # PLAS now reads the service the scheduling executor accrued, and
    # affinity follows the instance the retention protection sits on.
    assert sx.stamp("p0", 1, now=160.0) == 2000
    assert rx.route("p0", 1, now=160.0) == 0


def _drive(programs, route_fn, stamp_fn, complete_fn, ids):
    for i, pid in enumerate(ids):
        now = 100.0 + i
        route_fn(pid, 0, now)
        stamp_fn(pid, 0, now)
        complete_fn(pid, 0, now + 1.0)


def test_no_policy_keeps_per_program_state():
    """Every per-program value a policy reads must come from the PCB.

    Drives each policy through several programs and then looks for any
    program id inside the policy object. A policy that memoized program
    state would show up here as a dict or set keyed by program id.
    """
    ids = [f"prog-{i}" for i in range(4)]

    def scan(policy, label):
        found = []

        def walk(obj, path, depth=0):
            if depth > 3:
                return
            if isinstance(obj, str):
                if obj in ids:
                    found.append(path)
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(k, f"{path}[key]", depth + 1)
                    walk(v, f"{path}[val]", depth + 1)
                return
            if isinstance(obj, (list, tuple, set, frozenset)):
                for v in obj:
                    walk(v, f"{path}[*]", depth + 1)
                return
            if hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    walk(v, f"{path}.{k}", depth + 1)

        walk(policy, label)
        assert not found, f"{label} keeps per-program state at {found}"

    for policy in (RoundRobinRouting(2), LeastLoadedRouting(2),
                   SessionAffinityRouting(2, capacity_limit=8)):
        programs = ProgramTable()
        ex = RoutingExecutor(policy=policy, programs=programs)
        for i, pid in enumerate(ids):
            inst = ex.route(pid, 0, now=100.0 + i)
            ex.turn_complete(inst)
        scan(policy, type(policy).__name__)

    for policy in (FCFSScheduling(), ProgramFCFSScheduling(), PLASScheduling()):
        programs = ProgramTable()
        ex = SchedulingExecutor(policy=policy, programs=programs)
        for i, pid in enumerate(ids):
            ex.stamp(pid, 0, now=100.0 + i)
            ex.turn_complete(pid, service_s=1.0, turn_idx=0)
        scan(policy, type(policy).__name__)

    for policy in (CacheLRURetention(), TTLRetention(tau_s=60.0),
                   MinWasteRetention(PROFILE, default_gap_s=0.01)):
        programs = ProgramTable()
        ex = RetentionExecutor(policy=policy, kv=RecordingKVControl(
            blocks_per_request=3), programs=programs)
        for i, pid in enumerate(ids):
            ex.turn_complete(pid, 0, f"{pid}:0", "grep", now=100.0 + i,
                             context_tokens=800)
            ex.turn_arrival(pid, now=200.0 + i)
        scan(policy, type(policy).__name__)
