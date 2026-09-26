"""Unit tests for the scheduling policies (no engine, no vllm import)."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from policies import (
    FCFSScheduling,
    PLASScheduling,
    ProgramFCFSScheduling,
    SchedulingExecutor,
)


def test_fcfs_stamps_nothing():
    ex = SchedulingExecutor(policy=FCFSScheduling())
    assert ex.stamp("p0", 0, now=100.0) is None
    assert ex.stamps == []
    assert FCFSScheduling.engine_args == {}


def test_engine_args_priority_values():
    assert ProgramFCFSScheduling.engine_args == {"scheduling_policy": "priority"}
    assert PLASScheduling.engine_args == {"scheduling_policy": "priority"}


def test_program_fcfs_constant_within_program_ordered_across():
    ex = SchedulingExecutor(policy=ProgramFCFSScheduling())
    pa0 = ex.stamp("pA", 0, now=100.0)   # epoch: priority 0
    pb0 = ex.stamp("pB", 0, now=100.5)   # 500 ms later
    pa1 = ex.stamp("pA", 1, now=104.0)   # later turn, same priority
    pb1 = ex.stamp("pB", 1, now=105.0)
    assert pa0 == 0 and pa1 == 0
    assert pb0 == 500 and pb1 == 500
    # Earlier program's turns always precede later program's.
    assert pa1 < pb0


def test_program_fcfs_explicit_epoch():
    ex = SchedulingExecutor(policy=ProgramFCFSScheduling(epoch=100.0))
    assert ex.stamp("pA", 0, now=100.25) == 250


def test_plas_accumulates_measured_service():
    ex = SchedulingExecutor(policy=PLASScheduling())
    # New programs start at 0.
    assert ex.stamp("pA", 0, now=100.0) == 0
    assert ex.stamp("pB", 0, now=100.1) == 0
    # pA's turn 0 consumed 2.5 s of service; its next turn is stamped
    # with the attained ms, pB is unaffected.
    ex.turn_complete("pA", 2.5)
    assert ex.stamp("pA", 1, now=110.0) == 2500
    assert ex.stamp("pB", 1, now=110.1) == 0
    ex.turn_complete("pA", 0.5)
    ex.turn_complete("pB", 1.0)
    assert ex.stamp("pA", 2, now=120.0) == 3000
    assert ex.stamp("pB", 2, now=120.1) == 1000


def test_stamp_log_jsonl_sequence():
    buf = io.StringIO()
    ex = SchedulingExecutor(policy=PLASScheduling(), log_file=buf)
    ex.stamp("pA", 0, now=100.0)
    ex.turn_complete("pA", 1.0)
    ex.stamp("pA", 1, now=105.0)
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert [x["seq"] for x in lines] == [0, 1]
    assert [x["priority"] for x in lines] == [0, 1000]
    assert lines[1]["program_id"] == "pA" and lines[1]["turn_idx"] == 1


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
