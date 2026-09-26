"""The Program Control Block and the table that owns one per program.

This is the cross-turn state a serving policy is allowed to read, and the
only such state: the routing, scheduling, and retention policies in this
package keep no per-program dictionaries of their own, so a decision is
reproducible from the record alone. That is what makes the simulator mirror
and the real harness comparable, and it is what bounds a policy to what a
deployed system can actually observe at decision time (a trace also contains
this turn's output length and this gap's true duration; a PCB never does).

Five field groups, per the design section:
  identifier      program_id
  position        arrival_ts, turn_idx, turns_completed
  service history attained_service_s
  KV residency    kv_instance, context_tokens, kv_protected, kv_deadline_ts,
                  kv_request_id
  tool state      in_gap, tool_name, gap_started_ts

The record is FROZEN. Every transition goes through ProgramTable, which
replaces the record wholesale, so a policy holding a PCB cannot write to it
and cannot accumulate hidden state behind it. Transitions happen at exactly
three points:

  1. turn release      on_turn_release   placement and position are written
  2. turn completion   on_turn_complete  service accrues, tool state is set
  3. memory pressure   on_memory_pressure  KV residency is refreshed

Retention's own protect/evict stamp is part of (2), not a fourth point: the
executor reports it through note_retention, which writes the same residency
fields the pressure callback refreshes. Between these events the record is
stable, which is what makes a decision reproducible from it.

What is deliberately NOT here: instance state (queue depth, in-flight turns,
free blocks). That is not program state, it does not belong to any one
program, and policies that need it take an explicit probe (see
RoutingPolicy.inflight and MinWasteRetention.load_probe). Keeping it out is
the point of the boundary, not an omission.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

# Distinguishes "caller said nothing about the tool state" from "caller said
# there is no tool call". The three executors each report the same completion
# with the part they know, so an unmentioned field must not be overwritten:
# scheduling reports service, retention reports the tool. Without this,
# whichever ran second would clear the other's write.
_UNSET = object()


@dataclass(frozen=True)
class ProgramControlBlock:
    # identifier
    program_id: str
    # position
    arrival_ts: Optional[float] = None
    turn_idx: int = 0
    turns_completed: int = 0
    # service history
    attained_service_s: float = 0.0
    # KV residency
    kv_instance: Optional[int] = None
    context_tokens: int = 0
    kv_protected: bool = False
    kv_deadline_ts: Optional[float] = None
    # Handle to the protected blocks: the request whose KV the engine is
    # holding for this program. Residency, not bookkeeping, so it lives
    # on the record with the rest of the residency group.
    kv_request_id: Optional[str] = None
    # tool state
    in_gap: bool = False
    tool_name: Optional[str] = None
    #: Mean observed duration of THIS turn's tool, learned across every program
    #: that has called it. Continuum keys its pin rule on the tool, not on the
    #: program: on the board trace `pip` averages 19.9 s against `sed` at 0.12 s,
    #: so a per-program average over a program's mix of tools is a different
    #: policy. None until that tool has been observed at least once.
    tool_mean_gap_s: Optional[float] = None
    gap_started_ts: Optional[float] = None
    # Supplied only after tool completion; excludes callback/dispatch delays.
    completed_tool_duration_s: Optional[float] = None
    # The program's OWN gap history (accumulated at each turn release from
    # gaps already observed): the per-program mean tool time Continuum's
    # released code keys by tool name. Past observations only, never the
    # current gap's duration.
    gap_n: int = 0
    gap_sum_s: float = 0.0

    @property
    def gap_mean_s(self) -> Optional[float]:
        """Mean duration of this program's PAST tool gaps, or None before
        the first one completes."""
        return (self.gap_sum_s / self.gap_n) if self.gap_n else None

    def gap_elapsed_s(self, now: float) -> Optional[float]:
        """Time spent in the current tool gap, or None if not in one.

        A policy may read how long the gap has ALREADY lasted. It may not
        read how long the gap will last: that is trace knowledge, and it is
        why gap-dependent policies carry an explicit predictor instead.
        """
        if not self.in_gap or self.gap_started_ts is None:
            return None
        return max(0.0, now - self.gap_started_ts)


@dataclass
class ProgramTable:
    """Owns one PCB per program and applies the three transitions.

    One table is shared by the retention, routing, and scheduling
    executors, so all three read the same record. Completion is
    idempotent per (program, turn): each executor reports the turn it
    saw complete, and attained service must accrue once, not once per
    executor.
    """

    _pcbs: dict[str, ProgramControlBlock] = field(default_factory=dict)
    _completed: set = field(default_factory=set)
    _service_seen: set = field(default_factory=set)
    #: tool name -> (summed gap seconds, count), across every program. Cluster
    #: scoped on purpose: this is the quantity Continuum learns, and a
    #: per-program average over a program's mix of tools is a different policy.
    _tool_sum: dict = field(default_factory=dict)
    _tool_n: dict = field(default_factory=dict)

    # -- reads ---------------------------------------------------------
    def tool_mean_gap_s(self, tool_name: Optional[str]) -> Optional[float]:
        """Mean observed duration of a tool, across all programs.

        Cluster-scoped rather than per-program, because that is the quantity
        Continuum learns. None means the tool has not been seen yet, which is
        distinct from 0.0 and is what a cold-start rule keys on.
        """
        n = self._tool_n.get(tool_name, 0)
        return (self._tool_sum[tool_name] / n) if n else None

    def get(self, program_id: str) -> ProgramControlBlock:
        """The program's record, created on first contact."""
        pcb = self._pcbs.get(program_id)
        if pcb is None:
            pcb = ProgramControlBlock(program_id=program_id)
            self._pcbs[program_id] = pcb
        return pcb

    def known(self, program_id: str) -> bool:
        return program_id in self._pcbs

    def __len__(self) -> int:
        return len(self._pcbs)

    def _set(self, program_id: str, **changes) -> ProgramControlBlock:
        pcb = replace(self.get(program_id), **changes)
        self._pcbs[program_id] = pcb
        return pcb

    # -- transition 1: turn release ------------------------------------
    def observe_completed_tool(self, program_id: str, duration_s: float):
        """Stage an explicit tool-only observation for the next release.

        Callers must supply this only once the tool has completed. Hosts with
        no separate tool measurement retain the elapsed-gap fallback.
        """
        import math
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError('completed tool duration must be finite and nonnegative')
        pcb = self.get(program_id)
        if pcb.gap_started_ts is not None:
            self._set(program_id, completed_tool_duration_s=float(duration_s))

    def on_turn_release(
        self,
        program_id: str,
        turn_idx: int,
        now: float,
        instance: Optional[int] = None,
        context_tokens: Optional[int] = None,
    ) -> ProgramControlBlock:
        """A turn becomes ready and the Dispatch Plane's choices land."""
        pcb = self.get(program_id)
        changes = {
            "turn_idx": turn_idx,
            "in_gap": False,
            "tool_name": None,
            "gap_started_ts": None,
            "completed_tool_duration_s": None,
        }
        if pcb.arrival_ts is None:
            changes["arrival_ts"] = now
        if pcb.gap_started_ts is not None:
            # The gap that just ended becomes history, twice over: once for the
            # program, and once for the TOOL that produced it. The tool table is
            # what Continuum learns from; the per-program figures are kept
            # because they are a legitimate observable, though no published
            # policy reads them.
            gap = (pcb.completed_tool_duration_s
                   if pcb.completed_tool_duration_s is not None
                   else max(0.0, now - pcb.gap_started_ts))
            changes["gap_n"] = pcb.gap_n + 1
            changes["gap_sum_s"] = pcb.gap_sum_s + gap
            if pcb.tool_name is not None:
                self._tool_sum[pcb.tool_name] = \
                    self._tool_sum.get(pcb.tool_name, 0.0) + gap
                self._tool_n[pcb.tool_name] = \
                    self._tool_n.get(pcb.tool_name, 0) + 1
        if instance is not None:
            changes["kv_instance"] = instance
        if context_tokens is not None:
            changes["context_tokens"] = context_tokens
        return self._set(program_id, **changes)

    # -- transition 2: turn completion ---------------------------------
    def on_turn_complete(
        self,
        program_id: str,
        turn_idx: int,
        now: Optional[float] = None,
        service_s: Optional[float] = None,
        tool_name=_UNSET,
        context_tokens: Optional[int] = None,
    ) -> ProgramControlBlock:
        """Service accrues and the tool state is set.

        Each of the three executors reports the completion it saw, with
        the fields it knows, in any order. The two ACCUMULATING effects
        (turn count, attained service) are guarded per (program, turn) so
        they land exactly once; the assignments (tool state, context size)
        are applied whenever supplied.
        """
        pcb = self.get(program_id)
        key = (program_id, turn_idx)
        changes = {}
        if key not in self._completed:
            self._completed.add(key)
            changes["turns_completed"] = pcb.turns_completed + 1
        if service_s is not None and key not in self._service_seen:
            self._service_seen.add(key)
            changes["attained_service_s"] = (
                pcb.attained_service_s + max(0.0, service_s)
            )
        if tool_name is not _UNSET:
            changes["in_gap"] = tool_name is not None
            changes["tool_name"] = tool_name
            changes["gap_started_ts"] = now if tool_name is not None else None
            # Project the tool's learned mean into the record, so a retention
            # policy reads it from the PCB rather than keeping a dictionary of
            # its own -- the invariant this module states, and the failure that
            # let the two hosts learn different means from one trace.
            changes["tool_mean_gap_s"] = self.tool_mean_gap_s(tool_name)
        if context_tokens is not None:
            changes["context_tokens"] = context_tokens
        if not changes:
            return pcb
        return self._set(program_id, **changes)

    def note_retention(
        self,
        program_id: str,
        action: Optional[str],
        deadline_ts: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> ProgramControlBlock:
        """Record the retention action taken for this gap (part of
        transition 2, reported by the RetentionExecutor once applied)."""
        if action == "protect":
            return self._set(
                program_id, kv_protected=True, kv_deadline_ts=deadline_ts,
                kv_request_id=request_id,
            )
        if action in ("evict", "release", "swap"):
            # A swapped context leaves the NPU but survives on the host tier,
            # so it keeps its size; an evicted one is gone.
            return self._set(
                program_id, kv_protected=False, kv_deadline_ts=None,
                kv_request_id=None,
                **({"context_tokens": 0} if action == "evict" else {}),
            )
        return self.get(program_id)

    def protected_programs(self) -> list:
        """(program_id, request_id) for every program still holding a
        protection. Used to drain at end of run."""
        return [(p.program_id, p.kv_request_id) for p in self._pcbs.values()
                if p.kv_protected and p.kv_request_id is not None]

    # -- transition 3: memory pressure ---------------------------------
    def on_memory_pressure(
        self,
        program_id: str,
        context_tokens: Optional[int] = None,
        kv_protected: Optional[bool] = None,
        kv_deadline_ts: Optional[float] = None,
        kv_instance: Optional[int] = None,
    ) -> ProgramControlBlock:
        """Residency is refreshed after the engine reclaims against it.

        Called when the safety valve breaks a protection, so the record
        reflects what the engine actually holds rather than what the
        policy asked for.
        """
        changes = {}
        if context_tokens is not None:
            changes["context_tokens"] = context_tokens
        if kv_protected is not None:
            changes["kv_protected"] = kv_protected
            if not kv_protected:
                changes["kv_deadline_ts"] = None
        if kv_deadline_ts is not None:
            changes["kv_deadline_ts"] = kv_deadline_ts
        if kv_instance is not None:
            changes["kv_instance"] = kv_instance
        if not changes:
            return self.get(program_id)
        return self._set(program_id, **changes)
