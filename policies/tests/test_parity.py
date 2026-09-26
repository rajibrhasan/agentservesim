"""Unit tests for the decision-parity checker."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from policies.utils.kv_control import RecordingKVControl
from policies.utils.parity import check_parity
from policies import MinWasteRetention, RetentionExecutor, TTLRetention
from policies import RoutingExecutor, SessionAffinityRouting
from policies import PLASScheduling, SchedulingExecutor
from policies.utils.waste_model import WasteProfile

PROFILE = WasteProfile(a=0.0279, c=15.4, S=384)


def _retention_log(now_offset=0.0, blocks=4):
    """Drive the retention executor through a fixed event stream and
    return its JSONL lines. now_offset shifts the clock (parity must
    ignore timestamps); blocks emulates the engine's parked count."""
    buf = io.StringIO()
    ex = RetentionExecutor(
        policy=TTLRetention(tau_s=30.0),
        kv=RecordingKVControl(blocks_per_request=blocks),
        log_file=buf,
    )
    ex.turn_complete("p0", 0, "p0:0", "grep", now=100.0 + now_offset)
    ex.turn_arrival("p0", now=101.0 + now_offset)
    ex.turn_complete("p0", 1, "p0:1", "pytest", now=105.0 + now_offset)
    ex.finish()
    return buf.getvalue().splitlines()


def test_retention_parity_same_events_different_clock():
    rep = check_parity("retention", _retention_log(0.0), _retention_log(7.5))
    assert rep.ok, rep.summary()


def test_retention_parity_catches_action_divergence():
    real = _retention_log()
    sim = [
        json.dumps({**json.loads(x), "action": "evict"})
        if json.loads(x)["action"] == "protect" and json.loads(x)["turn_idx"] == 1
        else x
        for x in real
    ]
    rep = check_parity("retention", real, sim)
    assert not rep.ok
    assert any("action" in m for m in rep.mismatches)


def test_retention_parity_catches_block_count_divergence():
    rep = check_parity(
        "retention", _retention_log(blocks=4), _retention_log(blocks=5)
    )
    assert not rep.ok
    assert any("blocks" in m for m in rep.mismatches)


def test_retention_parity_min_waste_info_tolerance():
    def log(eps):
        buf = io.StringIO()
        ex = RetentionExecutor(
            policy=MinWasteRetention(PROFILE, default_gap_s=1.0 + eps),
            kv=RecordingKVControl(),
            log_file=buf,
        )
        ex.turn_complete("p0", 0, "p0:0", "pytest", now=100.0, context_tokens=1000)
        return buf.getvalue().splitlines()

    assert check_parity("retention", log(0.0), log(0.0)).ok
    # A materially different gap prediction shows up as an info mismatch.
    rep = check_parity("retention", log(0.0), log(0.5))
    assert not rep.ok
    assert any("info.gap_pred_s" in m for m in rep.mismatches)


def test_scheduling_parity_and_priority_divergence():
    def log(extra_service):
        buf = io.StringIO()
        ex = SchedulingExecutor(policy=PLASScheduling(), log_file=buf)
        ex.stamp("pA", 0, now=100.0)
        ex.turn_complete("pA", 1.0 + extra_service)
        ex.stamp("pA", 1, now=110.0)
        return buf.getvalue().splitlines()

    assert check_parity("scheduling", log(0.0), log(0.0)).ok
    rep = check_parity("scheduling", log(0.0), log(0.25))
    assert not rep.ok
    assert any("priority" in m for m in rep.mismatches)


def test_routing_parity_and_placement_divergence():
    def log(capacity_limit):
        buf = io.StringIO()
        ex = RoutingExecutor(
            policy=SessionAffinityRouting(2, capacity_limit=capacity_limit),
            log_file=buf,
        )
        ex.route("pA", 0, now=100.0)
        ex.route("pA", 1, now=100.1)
        ex.route("pA", 2, now=100.2)  # capacity fallback when limit=2
        return buf.getvalue().splitlines()

    assert check_parity("routing", log(2), log(2)).ok
    # Without the capacity fallback the third turn stays pinned:
    # divergence in both the placement and the fallback event.
    rep = check_parity("routing", log(2), log(None))
    assert not rep.ok
    assert any("instance" in m for m in rep.mismatches)
    assert any("fallback" in m for m in rep.mismatches)


def test_length_mismatch_reported():
    real = _retention_log()
    rep = check_parity("retention", real, real[:-1])
    assert not rep.ok
    assert any("length" in m for m in rep.mismatches)


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
