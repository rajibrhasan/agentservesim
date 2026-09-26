"""Unit tests for the routing policies (no engine, no network)."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from policies import (
    LeastLoadedRouting,
    RoundRobinRouting,
    RoutingExecutor,
    SessionAffinityRouting,
)


def test_round_robin_cycles():
    ex = RoutingExecutor(policy=RoundRobinRouting(3))
    got = [ex.route(f"p{i}", 0, now=100.0 + i) for i in range(7)]
    assert got == [0, 1, 2, 0, 1, 2, 0]


def test_least_loaded_tracks_inflight_and_completion():
    ex = RoutingExecutor(policy=LeastLoadedRouting(2))
    assert ex.route("pA", 0, now=100.0) == 0  # tie -> lowest index
    assert ex.route("pB", 0, now=100.1) == 1
    assert ex.route("pC", 0, now=100.2) == 0  # tie again
    # Instance 1 frees up: next turn goes there.
    ex.turn_complete(1)
    assert ex.route("pD", 0, now=100.3) == 1


def test_session_affinity_pins_and_sticks_under_load():
    ex = RoutingExecutor(policy=SessionAffinityRouting(2))
    assert ex.route("pA", 0, now=100.0) == 0
    assert ex.route("pB", 0, now=100.1) == 1
    # pA's later turns stay pinned to 0 even when 0 is more loaded.
    assert ex.route("pC", 0, now=100.2) == 0  # tie -> 0, now 0 has 2 in flight
    assert ex.route("pA", 1, now=100.3) == 0
    assert ex.decisions[0].info == {"pin": "new"}
    assert ex.decisions[3].info is None  # pinned follow-up, no event


def test_session_affinity_capacity_fallback_repins_and_logs():
    ex = RoutingExecutor(policy=SessionAffinityRouting(2, capacity_limit=2))
    assert ex.route("pA", 0, now=100.0) == 0
    assert ex.route("pA", 1, now=100.1) == 0  # pinned, at 2 in flight now
    # Pinned instance at capacity: fall back to least-loaded, re-pin.
    assert ex.route("pA", 2, now=100.2) == 1
    fb = ex.decisions[2].info
    assert fb["fallback"] is True and fb["from"] == 0
    # Re-pinned: subsequent turns follow the new instance.
    assert ex.route("pA", 3, now=100.3) == 1


def test_routing_log_jsonl_sequence():
    buf = io.StringIO()
    ex = RoutingExecutor(policy=RoundRobinRouting(2), log_file=buf)
    ex.route("pA", 0, now=100.0)
    ex.route("pB", 0, now=101.0)
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert [x["seq"] for x in lines] == [0, 1]
    assert [x["instance"] for x in lines] == [0, 1]
    assert lines[1]["program_id"] == "pB"


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
