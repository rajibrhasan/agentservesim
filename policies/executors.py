
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional, TextIO

from .base import (
    ProgramControlBlock, ProgramTable, PriorityStamp, RetentionDecision,
    RetentionPolicy, RoutingDecision, RoutingPolicy, SchedulingPolicy,
    SystemSignals)
from .utils.kv_control import KVControl
from .arrival_adapter import ArrivalAdapter



@dataclass
class RetentionExecutor:
    """Applies policy decisions through the transport and logs them.

    The request whose blocks are currently protected is a PCB field
    (kv_request_id), not executor bookkeeping, so arrival-time release
    targets the right request and a program's next protect never leaks
    the previous one without a second copy of the mapping.
    """

    policy: RetentionPolicy
    kv: KVControl
    log_file: Optional[TextIO] = None
    programs: ProgramTable = field(default_factory=ProgramTable)
    _seq: int = 0
    decisions: list[RetentionDecision] = field(default_factory=list)
    _arrival: ArrivalAdapter = field(default_factory=ArrivalAdapter)

    def observe_arrival(self, pcb, now):
        self._arrival.observe(self.policy, pcb, now)

    def _record(self, dec: RetentionDecision) -> None:
        self.decisions.append(dec)
        if self.log_file is not None:
            self.log_file.write(dec.to_json() + "\n")
            self.log_file.flush()

    def _next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def turn_complete(
        self,
        program_id: str,
        turn_idx: int,
        request_id: str,
        tool_name: Optional[str],
        now: float,
        context_tokens: Optional[int] = None,
        kv_utilization: Optional[float] = None,
    ) -> RetentionDecision:
        dec = self.prepare_complete(program_id, turn_idx, request_id, tool_name,
                                    now, context_tokens, kv_utilization)
        if dec.action == "none":
            return self.ack_complete(dec, None)
        if dec.action == "protect":
            stale = self.programs.get(program_id).kv_request_id
            if stale is not None:
                self.kv.release(stale)
                self.programs.note_retention(program_id, "release")
            blocks = self.kv.protect(request_id, dec.deadline_ts)
        elif dec.action == "evict":
            blocks = self.kv.evict(request_id)
        else:
            blocks = self.kv.swap(request_id)
        return self.ack_complete(dec, blocks)

    def prepare_complete(self, program_id, turn_idx, request_id, tool_name,
                         now, context_tokens=None, kv_utilization=None):
        """Decide under the shared policy-state lock, without engine I/O."""
        self.policy.signals = SystemSignals(kv_utilization=kv_utilization, ts=now)
        pcb = self.programs.on_turn_complete(
            program_id, turn_idx, now=now, tool_name=tool_name,
            context_tokens=context_tokens)
        out = self.policy.on_turn_complete(pcb, request_id, now)
        action, deadline_ts, info = ("none", None, None) if out is None else (
            out if len(out) == 3 else (*out, None))
        if action not in ("none", "protect", "evict", "swap"):
            raise ValueError(f"unknown retention action: {action}")
        return RetentionDecision(self._next_seq(), now, action, program_id,
                                 turn_idx, request_id, deadline_ts=deadline_ts,
                                 tool_name=tool_name, info=info)

    def ack_complete(self, dec, blocks):
        """Publish ownership only after the engine acknowledges the action."""
        if dec.action != "none":
            self.programs.note_retention(
                dec.program_id, dec.action, dec.deadline_ts,
                request_id=dec.request_id if blocks > 0 else None)
            if dec.action == "protect" and blocks <= 0:
                self.programs.note_retention(dec.program_id, "release")
        dec.blocks = blocks
        self._record(dec)
        return dec

    def turn_arrival(self, program_id: str, now: float,
                     kv_utilization: Optional[float] = None) -> Optional[RetentionDecision]:
        self.policy.signals = SystemSignals(kv_utilization=kv_utilization, ts=now)
        action = self._arrival.action(self.policy, self.programs.get(program_id), now)
        if action != "release":
            return None
        request_id = self.programs.get(program_id).kv_request_id
        if request_id is None:
            return None  # nothing protected (first turn, or valve took it)
        blocks = self.kv.release(request_id)
        self.programs.note_retention(program_id, "release")
        dec = RetentionDecision(
            self._next_seq(), now, "release", program_id, -1, request_id,
            blocks=blocks,
        )
        self._record(dec)
        return dec

    def finish(self) -> None:
        """End of run: release everything still parked."""
        for program_id, request_id in self.programs.protected_programs():
            self.kv.release(request_id)
            self.programs.note_retention(program_id, "release")


@dataclass
class SchedulingExecutor:
    """Computes and logs the stamp for each submitted turn."""

    policy: SchedulingPolicy
    log_file: Optional[TextIO] = None
    programs: ProgramTable = field(default_factory=ProgramTable)
    _seq: int = 0
    stamps: list[PriorityStamp] = field(default_factory=list)

    def stamp(self, program_id: str, turn_idx: int, now: float) -> Optional[int]:
        # Release is the transition; stamping reads the record after it,
        # so a first turn has its arrival_ts before the stamp is computed.
        pcb = self.programs.on_turn_release(program_id, turn_idx, now)
        priority = self.policy.priority(pcb, now)
        if priority is None:
            return None
        rec = PriorityStamp(self._seq, now, program_id, turn_idx, priority)
        self._seq += 1
        self.stamps.append(rec)
        if self.log_file is not None:
            self.log_file.write(rec.to_json() + "\n")
            self.log_file.flush()
        return priority

    admission_holds: int = 0
    admission_admits: int = 0

    @property
    def has_admit(self) -> bool:
        """True when the policy overrides admit(): the real driver runs the
        gateway hold loop only then (the simulator's custom_hooks flag)."""
        return type(self.policy).admit is not SchedulingPolicy.admit

    def admit(self, program_id: str, turn_idx: int, now: float,
              view: "QueueView") -> bool:
        """Gateway-side admission gate for one waiting turn at one tick.
        Counts holds/admits so a result can be checked against the
        simulator's admission_holds counter."""
        pcb = self.programs.get(program_id)
        ok = bool(self.policy.admit(pcb, now, view))
        if ok:
            self.admission_admits += 1
        else:
            self.admission_holds += 1
        if self.log_file is not None and not ok:
            self.log_file.write(json.dumps({
                "event": "hold", "ts": now, "program_id": program_id,
                "turn_idx": turn_idx, "kv_utilization": view.kv_utilization,
                "kv_free_tokens": view.kv_free_tokens,
                "kv_evictable_tokens": view.kv_evictable_tokens,
                "prompt_tokens": view.prompt_tokens,
                "cached_tokens": view.cached_tokens}) + "\n")
        return ok

    def turn_complete(
        self,
        program_id: str,
        service_s: float,
        turn_idx: Optional[int] = None,
    ) -> None:
        """Report a completed turn's measured service time (seconds).

        Accrues on the PCB, guarded per (program, turn), so reporting the
        same completion through more than one executor cannot double-count.
        """
        if turn_idx is None:
            turn_idx = self.programs.get(program_id).turn_idx
        self.programs.on_turn_complete(
            program_id, turn_idx, now=None, service_s=service_s
        )


@dataclass
class RoutingExecutor:
    """Applies the policy per turn, keeps the in-flight view, logs
    every placement."""

    policy: RoutingPolicy
    log_file: Optional[TextIO] = None
    programs: ProgramTable = field(default_factory=ProgramTable)
    _seq: int = 0
    decisions: list[RoutingDecision] = field(default_factory=list)

    def route(self, program_id: str, turn_idx: int, now: float) -> int:
        # The policy sees the record as it stands BEFORE this release, so
        # an affinity router reads the instance that holds the context
        # rather than the one it is about to choose.
        pcb = self.programs.get(program_id)
        instance, info = self.policy.route(pcb, now)
        self.programs.on_turn_release(program_id, turn_idx, now, instance=instance)
        rec = RoutingDecision(self._seq, now, program_id, turn_idx, instance, info)
        self._seq += 1
        self.decisions.append(rec)
        if self.log_file is not None:
            self.log_file.write(rec.to_json() + "\n")
            self.log_file.flush()
        self.policy.on_submit(instance)
        return instance

    def turn_complete(self, instance: int) -> None:
        self.policy.on_complete(instance)
