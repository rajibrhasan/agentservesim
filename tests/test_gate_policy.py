"""`--retention gate --scheduling gate` is the evolved champion by name.

policies/gate.py is a copy of evolve/champion_joint_41008503.py (the file the
2026-09-15 board staged as a candidate). The copy must stay identical to the
original apart from its import lines, and the adapter must build it through
the same path as the published values.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import policies
from policies.gate import EvolvedRetention, EvolvedScheduling
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def body(path):
    """The file without import lines and comments: the rules themselves."""
    out = []
    for line in open(path):
        s = line.strip()
        if s.startswith(("from ", "import ", "#")):
            continue
        out.append(line)
    return "".join(out)


def pcb(**kw):
    d = dict(tool_name="grep", kv_protected=False, arrival_ts=10.0,
             context_tokens=3000, program_id="p")
    d.update(kw)
    p = types.SimpleNamespace(**d)
    p.gap_elapsed_s = lambda now: kw.get("gap", None)
    return p


def test_copy_is_the_champion():
    assert body(os.path.join(REPO, "policies", "gate.py")) == \
        body(os.path.join(REPO, "evolve", "champion_joint_41008503.py"))


def test_registered_on_both_axes():
    assert policies.VALUES["kv"]["gate"] is EvolvedRetention
    assert policies.VALUES["scheduling"]["gate"] is EvolvedScheduling
    assert "gate" in policies.choices_for("kv")
    assert "gate" in policies.choices_for("scheduling")


def test_rules_behave_as_recorded():
    r = EvolvedRetention()
    assert r.on_turn_complete(pcb(tool_name="grep"), "r", 1.0) == ("protect", 3.0)
    assert r.on_turn_arrival(pcb(tool_name="pytest", gap=40.0), 42.0) == "release"
    assert r.on_turn_complete(pcb(tool_name="pytest"), "r", 43.0) is None  # EMA 40 s > 3 s
    s = EvolvedScheduling()
    pinned = s.priority(pcb(arrival_ts=12.0, kv_protected=True), 12.5)
    unpinned = s.priority(pcb(arrival_ts=10.0), 12.5)
    assert pinned < unpinned  # resident context runs first
    v = types.SimpleNamespace
    assert s.victim([v(prompt_tokens=5000, generated_tokens=10),
                     v(prompt_tokens=100, generated_tokens=5)], 0.0) == 1
    view = v(kv_utilization=0.95, kv_free_tokens=100, kv_evictable_tokens=200,
             prompt_tokens=3000, cached_tokens=0)
    assert s.admit(pcb(), 0.0, view) is False
    assert s.admit(pcb(kv_protected=True), 0.0, view) is True


def test_adapter_builds_gate_by_name():
    ad = UnifiedPolicyAdapter(retention_value="gate", scheduling_value="gate",
                              routing_value="session-affinity", num_instances=1,
                              block_size=16)
    assert isinstance(ad.retention_exec.policy, EvolvedRetention)
    assert isinstance(ad.scheduling_exec.policy, EvolvedScheduling)
    assert ad.priority_scheduling  # the stamp reaches the scheduler


def test_host_observes_once_without_releasing_before_priority_stamp():
    from policies.program import ProgramTable
    from policies.arrival_adapter import ArrivalAdapter
    table, policy, host = ProgramTable(), EvolvedRetention(), ArrivalAdapter()
    table.on_turn_complete('p', 0, now=1, tool_name='sed')
    table.note_retention('p', 'protect', 3., request_id='r0')
    table.observe_completed_tool('p', .2)
    before = table.get('p')
    host.observe(policy, before, 11.2)
    host.observe(policy, before, 12.2)
    assert table.get('p') == before  # real gap and protection clocks intact
    assert table.get('p').kv_protected
    assert abs(policy.gap_ema_by_tool['sed'] - .2) < 1e-9
    table.on_turn_release('p', 1, 12.2)
    assert host.action(policy, table.get('p'), 12.2) == 'release'
    assert host.action(policy, table.get('p'), 12.2) is None
    table.on_turn_complete('p', 1, now=13, tool_name='sed')
    table.observe_completed_tool('p', 20.)
    host.observe(policy, table.get('p'), 33)
    host.observe(policy, table.get('p'), 34)
    assert abs(policy.gap_ema_by_tool['sed'] - 4.16) < 1e-9
    assert policy.on_turn_complete(table.get('p'), 'r', 40) is None
    assert vars(policy) == {'gap_ema_by_tool': policy.gap_ema_by_tool}


def test_gate_pending_running_allocation_crosses_utilization_threshold():
    from collections import defaultdict
    from policies.base import QueueView
    ad = UnifiedPolicyAdapter.__new__(UnifiedPolicyAdapter)
    ad.custom_hooks = True
    ad.min_waste_fcfs_restore = False
    ad.scheduling_value = 'gate'
    ad._gate_trace = None
    ad.stats = defaultdict(int)
    ad._scheduling_mod = types.SimpleNamespace(QueueView=QueueView)
    ad.scheduling_exec = types.SimpleNamespace(policy=EvolvedScheduling())
    ad.programs = types.SimpleNamespace(get=lambda _: pcb(context_tokens=200))
    memory = types.SimpleNamespace(
        _bytes_per_token=1, npu_mem=1000, weight=0, npu_used=890,
        npu_reserved=0, avail_size=lambda _: 110,
        evictable_size=lambda _: 0, peek_prefix_hit=lambda _: 0)
    req = types.SimpleNamespace(infercept_session=False, preempt_seq=None,
                                session_id='p', original_input=200)
    assert ad.filter_waiting([req], [object()], memory, 0, 0) == [req]
    assert ad.filter_waiting([req], [object()], memory, 0, 0,
                             pending_reserve=20) == []
