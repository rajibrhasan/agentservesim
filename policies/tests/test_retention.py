"""Unit tests for the retention policies (no engine, no vllm import)."""

import pytest
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from policies.utils.kv_control import RecordingKVControl
from policies import (
    CacheLRURetention,
    EvictAlwaysRetention,
    MinWasteRetention,
    RetentionExecutor,
    TTLRetention,
)
from policies.utils.waste_model import WasteProfile, discard_waste, preserve_waste

# A100 defaults from InferCept's scheduler_v2 (test fixture only; real
# runs load a measured profile JSON from policies/profiles/).
PROFILE = WasteProfile(a=0.0279, c=15.4, S=384)


def test_cache_lru_and_evict_always_make_no_calls():
    for policy in (CacheLRURetention(), EvictAlwaysRetention()):
        kv = RecordingKVControl()
        ex = RetentionExecutor(policy=policy, kv=kv)
        dec = ex.turn_complete("p0", 0, "req-0", "pytest", now=100.0)
        assert dec.action == "none"
        assert ex.turn_arrival("p0", now=160.0) is None
        ex.finish()
        assert kv.calls == []


def test_engine_flags_per_value():
    assert EvictAlwaysRetention.engine_flags["enable_prefix_caching"] is False
    assert CacheLRURetention.engine_flags["enable_prefix_caching"] is True
    assert CacheLRURetention.engine_flags["kv_protection"] is False


def test_ttl_fixed_tau_protect_then_release():
    kv = RecordingKVControl(blocks_per_request=7)
    ex = RetentionExecutor(policy=TTLRetention(tau_s=60.0), kv=kv)

    dec = ex.turn_complete("p0", 0, "req-0", "pytest", now=100.0)
    assert dec.action == "protect"
    assert dec.deadline_ts == 160.0
    assert dec.blocks == 7
    assert kv.calls == [("protect", "req-0", 160.0)]

    rel = ex.turn_arrival("p0", now=130.0)
    assert rel is not None and rel.action == "release"
    assert rel.request_id == "req-0"
    assert kv.calls[-1] == ("release", "req-0")

    # Second arrival releases nothing.
    assert ex.turn_arrival("p0", now=131.0) is None


def test_ttl_predictor_overrides_tau_and_falls_back():
    def predictor(tool_name):
        return 12.5 if tool_name == "pytest" else None

    kv = RecordingKVControl()
    ex = RetentionExecutor(policy=TTLRetention(tau_s=60.0, predictor=predictor), kv=kv)
    d1 = ex.turn_complete("p0", 0, "req-0", "pytest", now=100.0)
    assert d1.deadline_ts == 112.5
    ex.turn_arrival("p0", now=101.0)
    d2 = ex.turn_complete("p0", 1, "req-1", "unknown_tool", now=200.0)
    assert d2.deadline_ts == 260.0


def test_ttl_stale_protection_released_before_new_protect():
    # Turn 1 protected but its arrival was never observed; protecting
    # turn 2 must release turn 1 first so nothing leaks.
    kv = RecordingKVControl()
    ex = RetentionExecutor(policy=TTLRetention(tau_s=60.0), kv=kv)
    ex.turn_complete("p0", 0, "req-0", None, now=100.0)
    ex.turn_complete("p0", 1, "req-1", None, now=200.0)
    assert kv.calls == [
        ("protect", "req-0", 160.0),
        ("release", "req-0"),
        ("protect", "req-1", 260.0),
    ]


def test_ttl_programs_tracked_independently():
    kv = RecordingKVControl()
    ex = RetentionExecutor(policy=TTLRetention(tau_s=10.0), kv=kv)
    ex.turn_complete("pA", 0, "req-A0", None, now=1.0)
    ex.turn_complete("pB", 0, "req-B0", None, now=2.0)
    rel = ex.turn_arrival("pA", now=3.0)
    assert rel.request_id == "req-A0"
    ex.finish()  # releases pB's still-parked request
    assert ("release", "req-B0") in kv.calls


def test_zero_parked_protect_is_not_tracked():
    # Engine reports 0 parked (blocks not cached / already reclaimed):
    # arrival must not issue a bogus release.
    kv = RecordingKVControl(blocks_per_request=0)
    ex = RetentionExecutor(policy=TTLRetention(tau_s=60.0), kv=kv)
    ex.turn_complete("p0", 0, "req-0", None, now=100.0)
    assert ex.turn_arrival("p0", now=110.0) is None


def test_min_waste_protects_short_gap_evicts_long_gap():
    # ctx=1000 on an idle engine: discard waste ~30 token*s. A 10 ms
    # gap (w_p=10) retains; a 1 s gap (w_p=1000) evicts.
    kv = RecordingKVControl(blocks_per_request=5)
    policy = MinWasteRetention(PROFILE, default_gap_s=0.01)
    ex = RetentionExecutor(policy=policy, kv=kv)
    dec = ex.turn_complete("p0", 0, "p0:0", "grep", now=100.0, context_tokens=1000)
    assert dec.action == "protect"
    assert dec.deadline_ts == 100.01
    assert dec.info["w_preserve"] < dec.info["w_discard"]

    policy_long = MinWasteRetention(PROFILE, default_gap_s=1.0)
    ex2 = RetentionExecutor(policy=policy_long, kv=RecordingKVControl())
    dec2 = ex2.turn_complete("p0", 0, "p0:0", "pytest", now=100.0, context_tokens=1000)
    assert dec2.action == "evict"
    assert dec2.info["w_preserve"] > dec2.info["w_discard"]
    assert ex2.kv.calls == [("evict", "p0:0")]


def test_min_waste_load_probe_shifts_decision_to_protect():
    # The same 1 s gap that evicts on an idle engine retains under
    # load: recompute chunks shrink and slow the running batch.
    load = (300, 50_000)  # (inflight_tokens, running_ctx_tokens)
    policy = MinWasteRetention(PROFILE, default_gap_s=1.0, load_probe=lambda: load)
    ex = RetentionExecutor(policy=policy, kv=RecordingKVControl())
    dec = ex.turn_complete("p0", 0, "p0:0", "pytest", now=100.0, context_tokens=1000)
    assert dec.action == "protect"
    assert dec.info["inflight_tokens"] == 300
    assert dec.info["running_ctx_tokens"] == 50_000
    # Consistency with the model functions themselves.
    assert dec.info["w_preserve"] == preserve_waste(1000, 1.0)
    assert dec.info["w_discard"] == discard_waste(1000, 300, 50_000, PROFILE)


def test_min_waste_gap_predictor_and_release_on_arrival():
    policy = MinWasteRetention(
        PROFILE, default_gap_s=1.0,
        predictor=lambda tool: 0.005 if tool == "grep" else None,
    )
    kv = RecordingKVControl(blocks_per_request=3)
    ex = RetentionExecutor(policy=policy, kv=kv)
    # Predictor gives grep a 5 ms gap -> protect despite the long default.
    dec = ex.turn_complete("p0", 0, "p0:0", "grep", now=100.0, context_tokens=1000)
    assert dec.action == "protect"
    assert dec.deadline_ts == 100.005
    rel = ex.turn_arrival("p0", now=100.004)
    assert rel is not None and rel.request_id == "p0:0"
    assert kv.calls == [("protect", "p0:0", 100.005), ("release", "p0:0")]


def test_min_waste_requires_context_tokens():
    policy = MinWasteRetention(PROFILE, default_gap_s=1.0)
    ex = RetentionExecutor(policy=policy, kv=RecordingKVControl())
    try:
        ex.turn_complete("p0", 0, "p0:0", "grep", now=100.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without context_tokens")


def test_min_waste_info_serialized_in_jsonl_log():
    buf = io.StringIO()
    policy = MinWasteRetention(PROFILE, default_gap_s=1.0)
    ex = RetentionExecutor(policy=policy, kv=RecordingKVControl(), log_file=buf)
    ex.turn_complete("p0", 0, "p0:0", "pytest", now=100.0, context_tokens=1000)
    line = json.loads(buf.getvalue().splitlines()[0])
    assert line["action"] == "evict"
    for key in ("gap_pred_s", "w_preserve", "w_discard", "context_tokens"):
        assert key in line["info"]


def test_profile_json_loading(tmp_path):
    """Loads the JSON shape a measured profile sweep writes.

    Built inline so the test does not depend on which measured profiles happen
    to be checked in under policies/profiles/."""
    path = tmp_path / "infercept_profile_test.json"
    path.write_text(json.dumps({"a": 0.0279, "c": 15.4, "S": 384,
                                "points": [[1, 15.4], [384, 26.1]]}))
    prof = WasteProfile.from_json(str(path))
    assert prof == WasteProfile(a=0.0279, c=15.4, S=384)


def test_decision_log_jsonl_sequence():
    kv = RecordingKVControl()
    buf = io.StringIO()
    ex = RetentionExecutor(policy=TTLRetention(tau_s=60.0), kv=kv, log_file=buf)
    ex.turn_complete("p0", 0, "req-0", "git", now=100.0)
    ex.turn_arrival("p0", now=120.0)
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert [x["action"] for x in lines] == ["protect", "release"]
    assert [x["seq"] for x in lines] == [0, 1]
    assert lines[0]["tool_name"] == "git"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(("%d FAILED" % fails) if fails else "all passed")
    sys.exit(1 if fails else 0)


# ------------------------------------------- Continuum keys on the TOOL

def test_continuum_reads_the_tool_mean_not_a_private_dict():
    """The policy must hold no per-program state: a decision that is not
    reproducible from the record diverges between hosts whenever one of them
    skips an observation hook. That is exactly how the tool-gap bug happened."""
    from policies import ContinuumTTLRetention
    p = ContinuumTTLRetention(pin_s=2.0)
    assert not [a for a in vars(p) if a.startswith("_")]


def test_continuum_pins_a_fast_tool_and_declines_a_slow_one():
    from policies.program import ProgramTable
    from policies import ContinuumTTLRetention

    t = ProgramTable()
    p = ContinuumTTLRetention(pin_s=2.0, threshold_s=2.0)

    # sed: two observed gaps of 0.1s. pip: one of 20s.
    t.on_turn_release("a", 0, now=0.0)
    t.on_turn_complete("a", 0, now=1.0, tool_name="sed")
    t.on_turn_release("a", 1, now=1.1)                    # 0.1s sed gap
    t.on_turn_complete("a", 1, now=2.0, tool_name="pip")
    t.on_turn_release("a", 2, now=22.0)                   # 20s pip gap

    assert t.tool_mean_gap_s("sed") == pytest.approx(0.1)
    assert t.tool_mean_gap_s("pip") == pytest.approx(20.0)

    fast = t.on_turn_complete("a", 2, now=23.0, tool_name="sed")
    assert p.on_turn_complete(fast, "r", 23.0)[0] == "protect"

    slow = t.on_turn_complete("a", 3, now=24.0, tool_name="pip")
    assert p.on_turn_complete(slow, "r", 24.0) is None    # slow tool: no pin


def test_a_tool_is_learned_across_programs_not_within_one():
    """Continuum's table is cluster-scoped: program B benefits from what
    program A observed about the same tool."""
    from policies.program import ProgramTable
    t = ProgramTable()
    t.on_turn_release("a", 0, now=0.0)
    t.on_turn_complete("a", 0, now=1.0, tool_name="pip")
    t.on_turn_release("a", 1, now=21.0)                   # A observes 20s

    t.on_turn_release("b", 0, now=0.0)
    pcb = t.on_turn_complete("b", 0, now=1.0, tool_name="pip")
    assert pcb.tool_mean_gap_s == pytest.approx(20.0)     # B inherits it


def test_an_unseen_tool_has_no_mean_which_is_not_zero():
    from policies.program import ProgramTable
    t = ProgramTable()
    t.on_turn_release("a", 0, now=0.0)
    pcb = t.on_turn_complete("a", 0, now=1.0, tool_name="never_seen")
    assert pcb.tool_mean_gap_s is None


def test_per_program_gap_history_still_accumulates():
    """Kept as an observable for search, though no published policy reads it."""
    from policies.program import ProgramTable
    t = ProgramTable()
    t.on_turn_release("a", 0, now=0.0)
    t.on_turn_complete("a", 0, now=1.0, tool_name="cat")
    t.on_turn_release("a", 1, now=3.0)
    pcb = t.get("a")
    assert pcb.gap_n == 1 and pcb.gap_sum_s == pytest.approx(2.0)
