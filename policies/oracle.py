"""Clairvoyant probe policies: per-axis headroom bounds (sim-only).

These policies read TRACE knowledge (the next tool gap's duration, the
program's remaining work) that no deployable policy can have, which is
exactly the point: they bound what each axis can possibly buy in a cell.
The gap between the best published policy and the oracle on an axis is
the headroom that justifies (or kills) searching that axis.

They are simulator-only probes. The real driver never instantiates
them: the OracleTable requires the full trace up front, and the harness
PCB deliberately excludes future knowledge (see program.gap_elapsed_s).

- OracleTTLRetention: TTL with a perfect per-turn tau — protect each
  gap until exactly its measured end. Bounds the gap-predicting TTL
  family (Continuum's predictive TTL with a perfect predictor). Under
  pressure the safety valve breaks unexpired protections
  latest-deadline-first, which with exact deadlines evicts the context
  whose turn returns furthest in the future — a Belady-flavored bonus
  that a real TTL's inflated deadlines cannot reproduce.
- OracleSRPTScheduling: priority = the program's true remaining work
  (new input + output tokens over its remaining turns). Bounds the
  program-level size-based scheduling family (PLAS approximates this
  with attained service as the proxy).
"""

from __future__ import annotations

import json

from .base import RetentionPolicy
from .base import SchedulingPolicy


class OracleTable:
    """Trace ground truth keyed by (program_id, turn_idx): the gap that
    follows each turn and the work remaining from each turn on."""

    def __init__(self):
        self.gap_s: dict = {}
        self.remaining_toks: dict = {}

    @classmethod
    def from_jsonl(cls, path: str) -> "OracleTable":
        t = cls()
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                subs = row.get("sub_requests")
                if not subs:
                    continue  # flat request: no program structure
                pid = row["session_id"]
                # per-turn work: tokens this turn adds (new input beyond
                # the previous context + its own output)
                work = []
                prev_ctx = 0
                for s in subs:
                    new_in = max(0, s["input_toks"] - prev_ctx)
                    work.append(new_in + s["output_toks"])
                    prev_ctx = s["input_toks"] + s["output_toks"]
                suffix = 0
                for i in range(len(subs) - 1, -1, -1):
                    suffix += work[i]
                    t.remaining_toks[(pid, i)] = suffix
                    t.gap_s[(pid, i)] = subs[i].get("tool_duration_ns", 0) / 1e9
        return t


class OracleTTLRetention(RetentionPolicy):
    """Protect each gap for exactly its measured duration."""

    def __init__(self, table: OracleTable) -> None:
        self.table = table

    def on_turn_complete(self, pcb, request_id, now):
        g = self.table.gap_s.get((pcb.program_id, pcb.turn_idx))
        if not g or g <= 0.0:
            return None  # final turn (or zero gap): nothing to cover
        return ("protect", now + g, {"oracle_gap_s": g})

    def on_turn_arrival(self, pcb, now):
        return "release"


class OracleSRPTScheduling(SchedulingPolicy):
    """Priority = true remaining program work in tokens (lower first)."""

    engine_args = {"scheduling_policy": "priority"}

    def __init__(self, table: OracleTable) -> None:
        self.table = table

    def priority(self, pcb, now):
        return int(self.table.remaining_toks.get(
            (pcb.program_id, pcb.turn_idx), 0))
