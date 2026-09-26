"""What a trace file means.

The loader's job is to turn three on-disk shapes into one in-memory model, so
these tests are about meaning rather than parsing: that `output` is cumulative
and must be made per-turn at ingestion, that a flat request is a one-turn
program, that a chain is the path graph of a DAG, and that a fan-in waits for
its LAST parent. Get any of those wrong and every downstream number is wrong in
a way no plane below can detect.
"""
import json
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.core.program_orchestrator import ProgramOrchestrator   # noqa: E402
from serving.core.program_workload import load_programs             # noqa: E402


# ------------------------------------------------------------- ingestion

def _write_trace(tmp_path, rows):
    import json
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_loading_makes_output_cumulative():
    """The engine compares `output` against a counter running from zero through
    the prompt into decode, so the field is input+output. Applying it anywhere
    else gives a completion rule wrong by the length of the prompt."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = _write_trace(pathlib.Path(d), [{
            "session_id": "s1", "arrival_time_ns": 0,
            "sub_requests": [{"input_toks": 100, "output_toks": 20,
                              "tool_duration_ns": 5}],
        }])
        orch = ProgramOrchestrator()
        assert load_programs(path, orch) == 1
        turn = orch.get("s1").pending[0]
        assert turn.input_toks == 100 and turn.output_toks == 120


def test_a_flat_request_is_a_one_turn_program():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = _write_trace(pathlib.Path(d),
                            [{"input_toks": 10, "output_toks": 5,
                              "arrival_time_ns": 0}])
        orch = ProgramOrchestrator()
        assert load_programs(path, orch) == 1
        assert len(orch.get("program_0").pending) == 1


# ------------------------------------------------------------------ DAG

def _dag_trace(tmp_path):
    """Diamond: root fans out to two, which fan in to a join."""
    import json
    rows = [{
        "workflow_id": "w1", "arrival_time_ns": 0,
        "nodes": [{"node_id": n, "input_toks": 10, "output_toks": 5}
                  for n in ("root", "l", "r", "join")],
        "edges": [{"src": "root", "dst": "l", "delay_ns": 10},
                  {"src": "root", "dst": "r", "delay_ns": 50},
                  {"src": "l", "dst": "join"},
                  {"src": "r", "dst": "join"}],
    }]
    p = tmp_path / "dag.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_a_dag_releases_roots_first():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        orch = ProgramOrchestrator()
        assert load_programs(_dag_trace(pathlib.Path(d)), orch) == 4
        assert [t.node_id for _, t in orch.due(0.0)] == ["root"]


def test_a_fan_out_makes_two_turns_runnable_at_different_times():
    """Each edge carries its own delay, so siblings do not become runnable
    together -- which a single per-program 'next available' could not express."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        orch = ProgramOrchestrator()
        load_programs(_dag_trace(pathlib.Path(d)), orch)
        orch.take_next("w1", now=0.0, node_id="root")
        orch.on_turn_complete("w1", now=100.0, service_s=1.0, node_id="root")
        assert [t.node_id for _, t in orch.due(105.0)] == []          # 100+10
        assert [t.node_id for _, t in orch.due(115.0)] == ["l"]
        assert [t.node_id for _, t in orch.due(160.0)] == ["l", "r"]  # 100+50


def test_a_fan_in_waits_for_the_last_parent():
    """The barrier: a join is runnable only when every parent has landed, which
    is why completions are a set and not one timestamp."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        orch = ProgramOrchestrator()
        load_programs(_dag_trace(pathlib.Path(d)), orch)
        orch.take_next("w1", now=0.0, node_id="root")
        orch.on_turn_complete("w1", now=10.0, service_s=1.0, node_id="root")
        orch.take_next("w1", now=20.0, node_id="l")
        orch.on_turn_complete("w1", now=30.0, service_s=1.0, node_id="l")
        assert [t.node_id for _, t in orch.due(1e9)] == ["r"]   # join still waits
        orch.take_next("w1", now=60.0, node_id="r")
        orch.on_turn_complete("w1", now=70.0, service_s=1.0, node_id="r")
        assert [t.node_id for _, t in orch.due(70.0)] == ["join"]


def test_a_chain_is_the_path_graph_case_of_a_dag():
    """One model, not two code paths -- the old loader's own docstring says the
    chain is the path-graph special case."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "chain.jsonl"
        import json
        p.write_text(json.dumps({
            "session_id": "s", "arrival_time_ns": 0,
            "sub_requests": [{"input_toks": 10, "output_toks": 5,
                              "tool_duration_ns": 7},
                             {"input_toks": 10, "output_toks": 5}]}))
        orch = ProgramOrchestrator()
        load_programs(str(p), orch)
        turns = orch.get("s").pending
        assert turns[0].parents == ()
        assert turns[1].parents == ((0, 7),)     # previous turn, its tool gap
