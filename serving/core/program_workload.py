
from __future__ import annotations

import os
from typing import Sequence

from .program_orchestrator import PlannedTurn, ProgramOrchestrator


def load_programs(path: str, orchestrator: ProgramOrchestrator,
                  enable_prefix_caching: bool = True) -> int:
   
    import json

    turns_total = 0
    # The simulator chdir's into astra-sim/, so a repo-relative dataset path is
    # reached via '../'. Absolute paths (a trace on /orange) are used as-is.
    # `save_workflow_metrics` does the same on the way out; the two have to
    # agree or a run reads one tree and writes another.
    if not os.path.isabs(path):
        path = f"../{path}"
    with open(path) as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            arrival = float(row.get("arrival_time_ns", 0))
            ids = enable_prefix_caching

            if row.get("nodes"):
                program_id = str(row.get("workflow_id") or f"workflow_{line_no}")
                specs = {n.get("node_id", i): n
                         for i, n in enumerate(row["nodes"])}
                parents = {nid: [] for nid in specs}
                for e in row.get("edges", []):
                    if e["dst"] in parents:
                        parents[e["dst"]].append(
                            (e["src"], int(e.get("delay_ns", 0))))
                turns = tuple(
                    PlannedTurn(
                        node_id=nid,
                        input_toks=int(n["input_toks"]),
                        output_toks=int(n["input_toks"]) + int(n["output_toks"]),
                        tool=n.get("tool") or n.get("tool_name"),
                        parents=tuple(parents[nid]),
                        input_hash_ids=tuple(n.get("input_tok_ids", ())) if ids else (),
                        model=n.get("model"),
                    )
                    for nid, n in specs.items())
            else:
                subs = row.get("sub_requests") or [
                    {"input_toks": row.get("input_toks", 0),
                     "output_toks": row.get("output_toks", 0),
                     "tool_duration_ns": 0}]
                program_id = str(row.get("session_id") or f"program_{line_no}")
                turns = tuple(
                    PlannedTurn(
                        node_id=i,
                        input_toks=int(sr["input_toks"]),
                        output_toks=int(sr["input_toks"]) + int(sr["output_toks"]),
                        tool=sr.get("tool") or sr.get("tool_name"),
                        # the path graph: parented on the previous turn, with
                        # that turn's tool call as the edge delay
                        parents=() if i == 0 else
                        ((i - 1, int(subs[i - 1].get("tool_duration_ns", 0) or 0)),),
                        input_hash_ids=tuple(sr.get("input_tok_ids", ())) if ids else (),
                    )
                    for i, sr in enumerate(subs))

            orchestrator.define_program(program_id, turns, arrival)
            turns_total += len(turns)
    return turns_total


def generate(router, path: str, schedulers: Sequence,
             enable_prefix_caching: bool = False, model: str = "") -> int:
   
    from .program_router import dispatch
    load_programs(path, router.orch, enable_prefix_caching)
    return dispatch(router, schedulers, float("inf"), model=model)
