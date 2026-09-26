"""Continuum (Li et al., 2025, arXiv:2511.02230).

Two halves that only make sense together: the KV half pins a finished turn's
context for a fixed window when its tool is fast, and the scheduling half puts
turns whose context is still pinned ahead of those whose is not. The second is
meaningful only because the first exists.

As RELEASED (github.com/Hanchenli/vllm-continuum, preview code), not the
paper's CDF estimation.
"""
from typing import Optional

from .base import KVPolicy, SchedulingPolicy

PAPER = "Continuum 2511.02230"


class ContinuumKV(KVPolicy):
    """Continuum as RELEASED (github.com/Hanchenli/vllm-continuum, preview
    code; deep-dive 2026-09-01), not the paper's CDF estimation:

    - Pin for a FIXED pin_s (2.0 s in the release,
      FIXED_THRESHOLD_CONTINUUM) iff the tool's observed mean exec time
      is <= threshold_s, or there is no history yet (cold start pins).
      Slow tools are NOT pinned at all (return None -> blocks fall to
      the ordinary LRU; discard-long, never pin-then-expire).
    - release_event = "scheduled": the release fires when the next turn
      is ADMITTED to the running batch, not when it arrives — the real
      unpin requires both TTL expiry and the job absent from the waiting
      queue, so protection persists through the entire queue wait.
    - Mean tool time is learned online from completed tool observations at arrival
      (real code: request_arrives), keyed BY TOOL NAME across all
      programs -- not per program. The distinction is large on real
      traces: on the leaderboard board trace `pip` averages 19.9 s
      against `sed` at 0.12 s, so averaging a program's mix of tools
      into one number pins or drops all of its turns together, which is
      a different policy. Corrected 2026-09-13; the mirror keyed by
      program until then, on a stale note that the traces carried no
      tool identity. They do -- every sub-request has a `tool` field.

    Trace replay supplies the completed synthetic tool duration explicitly;
    it excludes policy callbacks and dispatch overhead. Hosts without a
    separate measurement fall back to the observed inter-call gap.

    The mean is read from `pcb.tool_mean_gap_s`, which `ProgramTable`
    projects into the record. The policy keeps no dictionary of its own,
    so the decision is reproducible from the record alone and cannot
    depend on which host called which observation hook.
    """

    release_event = "scheduled"

    @classmethod
    def from_config(cls, cfg):
        # Unset means Continuum's own pin window wins -- the released code's
        # FIXED_THRESHOLD_CONTINUUM. An explicit value is honoured, including 60,
        # which the old `!= 60.0` sentinel made impossible to ask for.
        return cls(pin_s=cfg.pin_s if cfg.tau_s is None else cfg.tau_s)

    def __init__(self, pin_s: float = 2.0, threshold_s: float = None) -> None:
        self.pin_s = pin_s
        self.threshold_s = pin_s if threshold_s is None else threshold_s

    def on_turn_complete(self, pcb, request_id, now):
        # The reference does not retain a completed request with no tool call.
        if pcb.tool_name is None:
            return None
        # The tool this turn is about to wait on, as learned across every
        # program that has called it. None = never seen, and a cold start
        # pins (the released code does the same).
        mean = pcb.tool_mean_gap_s
        if mean is not None and mean > self.threshold_s:
            return None  # slow tool: no pin
        return ("protect", now + self.pin_s,
                {"tool": pcb.tool_name, "tool_mean_gap_s": mean})

    def on_turn_arrival(self, pcb, now):
        return "release"


class ContinuumScheduling(SchedulingPolicy):
    """Continuum (Li et al., 2025, arXiv:2511.02230) waiting-queue order:
    turns of programs whose KV is pinned within its TTL window go before
    unpinned ones; within a class, program-level FCFS. Encoded as a single
    integer priority (vLLM: lower runs first): class * 2^40 + program
    arrival ms. Must be stamped BEFORE the arrival release of the
    program's protection (the executor order guarantees this), so the
    record still says whether the returning turn's context is pinned."""

    engine_args = {"scheduling_policy": "priority"}
    _CLASS = 1 << 40

    def __init__(self, epoch: Optional[float] = None) -> None:
        self.epoch = epoch

    def priority(self, pcb, now):
        if self.epoch is None:
            self.epoch = now
        arrival = pcb.arrival_ts if pcb.arrival_ts is not None else now
        pinned = 0 if pcb.kv_protected else 1
        return pinned * self._CLASS + int(round((arrival - self.epoch) * 1000))
