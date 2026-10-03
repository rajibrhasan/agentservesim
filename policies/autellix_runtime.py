from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class QueueConfig:
    service_boundaries_s: Tuple[float, ...]
    quanta_s: Tuple[float, ...]
    starvation_ratio: float

    def __post_init__(self):
        if len(self.quanta_s) != len(self.service_boundaries_s) + 1:
            raise ValueError("one quantum is required for each service interval")
        values = (*self.service_boundaries_s, *self.quanta_s,
                  self.starvation_ratio)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("queue parameters must be positive and finite")
        if any(a >= b for a, b in zip(self.service_boundaries_s,
                                     self.service_boundaries_s[1:])):
            raise ValueError("service boundaries must be strictly increasing")


@dataclass
class Process:
    service_s: float = 0.0
    wait_s: float = 0.0


@dataclass
class Call:
    request_id: str
    program_id: str
    inherited_service_s: float
    queue: int
    remaining_s: float
    ready_at: float
    # Lifetime accounting survives starvation promotion. The paper resets
    # the local fairness counters; that must not erase completed GPU work.
    execution_s: float = 0.0
    wait_s: float = 0.0
    fairness_service_s: float = 0.0
    fairness_wait_s: float = 0.0


@dataclass(frozen=True)
class SchedulePlan:
    selected: Tuple[str, ...]
    preempt: Tuple[str, ...]
    #: Next calls in queue order beyond the first non-fitting one. Autellix's
    #: multi-step scheduler keeps them queued on the engine so they start the
    #: moment a selected call finishes inside the N-step window.
    overprovisioned: Tuple[str, ...] = ()


class AutellixRuntime:
    """PLAS for sequential programs, inherited critical-path service for forks.

    A batch is indivisible: its full duration is charged to each participating
    call, once, regardless of the number of tensor-parallel ranks. A waiting
    call accrues only queue time. Tool gaps accrue neither. Only one batch may
    be in flight; asynchronous/pipelined adapters need a different timing
    protocol and must not silently feed overlapping intervals here.
    """

    def __init__(self, config: QueueConfig):
        self.config = config
        self.processes: Dict[str, Process] = {}
        self.calls: Dict[str, Call] = {}
        self.queues = [deque() for _ in config.quanta_s]
        self.completed_service: Dict[str, Tuple[str, float]] = {}
        self._batch: Optional[Tuple[Tuple[str, ...], float]] = None
        self._time = 0.0

    def _check_time(self, now: float):
        if not math.isfinite(now) or now < self._time:
            raise ValueError("time must be finite and monotonic")

    def arrive(self, request_id: str, program_id: str, now: float,
               inherited_service_s: Optional[float] = None,
               inherited_wait_s: Optional[float] = None):
        self._check_time(now)
        if request_id in self.calls or request_id in self.completed_service:
            raise ValueError(f"duplicate request: {request_id}")
        process = self.processes.get(program_id, Process())
        service = (process.service_s if inherited_service_s is None
                   else inherited_service_s)
        if not math.isfinite(service) or service < 0:
            raise ValueError("inherited service must be finite and nonnegative")
        if inherited_wait_s is not None and (
                not math.isfinite(inherited_wait_s) or inherited_wait_s < 0):
            raise ValueError('inherited wait must be finite and nonnegative')
        # A different engine may not have seen any earlier program turns.
        process.service_s = max(process.service_s, service)
        if inherited_wait_s is not None:
            process.wait_s = max(process.wait_s, inherited_wait_s)
        self.processes.setdefault(program_id, process)
        queue = bisect_right(self.config.service_boundaries_s, service)
        self.calls[request_id] = Call(request_id, program_id, service, queue,
                                      self.config.quanta_s[queue], now)
        self.queues[queue].append(request_id)
        self._time = now

    def _wait_until(self, call: Call, now: float):
        elapsed = now - call.ready_at
        if elapsed < 0:
            raise ValueError("call cannot be processed before its arrival")
        call.wait_s += elapsed
        call.fairness_wait_s += elapsed
        call.ready_at = now

    def start_batch(self, request_ids: Sequence[str], now: float):
        self._check_time(now)
        ids = tuple(request_ids)
        if self._batch is not None:
            raise RuntimeError("a batch is already in flight")
        if not ids or len(ids) != len(set(ids)) or any(r not in self.calls for r in ids):
            raise ValueError("batch must contain distinct active requests")
        for rid in ids:
            self._wait_until(self.calls[rid], now)
        self._batch = (ids, now)
        self._time = now

    def finish_batch(self, now: float, finished: Sequence[str] = (),
                     execution_s: Optional[float] = None):
        self._check_time(now)
        if self._batch is None:
            raise RuntimeError("no batch is in flight")
        ids, start = self._batch
        done = set(finished)
        if len(done) != len(finished) or not done.issubset(ids):
            raise ValueError("finished requests must be distinct batch members")
        elapsed = now - start if execution_s is None else execution_s
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('execution duration must be finite and nonnegative')
        for rid in ids:
            call = self.calls[rid]
            call.execution_s += elapsed
            call.fairness_service_s += elapsed
            call.remaining_s -= elapsed
            call.ready_at = now
            if rid in done:
                process = self.processes[call.program_id]
                path_service = call.inherited_service_s + call.execution_s
                process.service_s = max(process.service_s, path_service)
                process.wait_s += call.wait_s
                self.completed_service[rid] = (call.program_id, path_service)
                self.queues[call.queue].remove(rid)
                del self.calls[rid]
            elif call.remaining_s <= 0:
                self._move(call, min(call.queue + 1, len(self.queues) - 1))
        self._batch = None
        self._time = now

    def cancel(self, request_id: str, now: float):
        """Remove an aborted call at a batch boundary, preserving consumed work."""
        self._check_time(now)
        if self._batch is not None:
            raise RuntimeError('cannot cancel inside an active batch')
        call = self.calls.pop(request_id)
        self._wait_until(call, now)
        self.queues[call.queue].remove(request_id)
        process = self.processes[call.program_id]
        path_service = call.inherited_service_s + call.execution_s
        process.service_s = max(process.service_s, path_service)
        process.wait_s += call.wait_s
        self.completed_service[request_id] = (call.program_id, path_service)
        self._time = now

    def _move(self, call: Call, queue: int):
        self.queues[call.queue].remove(call.request_id)
        call.queue = queue
        call.remaining_s = self.config.quanta_s[queue]
        self.queues[queue].append(call.request_id)

    def plan(self, now: float, resident: Sequence[str],
             can_fit: Callable[[str, Tuple[str, ...]], bool],
             overprovision: int = 0) -> SchedulePlan:
        """Plan in queue order; the engine reports cumulative batch feasibility.

        The callback must be read-only: reservations/transfers are committed by
        the engine after planning. Stop at the first non-fitting request, as in
        Algorithm 1. Resident requests excluded from the batch are preempted;
        this runtime does not substitute recomputation for required swapping.
        """
        self._check_time(now)
        if self._batch is not None:
            raise RuntimeError("cannot replan while a batch is in flight")
        if len(resident) != len(set(resident)) or any(r not in self.calls for r in resident):
            raise ValueError("resident requests must be distinct active calls")
        for queue in self.queues:
            for rid in tuple(queue):
                call = self.calls[rid]
                self._wait_until(call, now)
                process = self.processes[call.program_id]
                wait = process.wait_s + call.fairness_wait_s
                service = process.service_s + call.fairness_service_s
                if (call.queue > 0 and service > 0
                        and wait >= self.config.starvation_ratio * service):
                    self._move(call, 0)
                    call.fairness_wait_s = 0.0
                    call.fairness_service_s = 0.0
        if not isinstance(overprovision, int) or overprovision < 0:
            raise ValueError('overprovision must be a nonnegative count')
        selected, extra = [], []
        blocked = False
        for queue in self.queues:
            for rid in queue:
                if blocked or not can_fit(rid, tuple(selected)):
                    blocked = True
                    if len(extra) < overprovision:
                        extra.append(rid)
                    continue
                selected.append(rid)
        chosen = tuple(selected)
        self._time = now
        kept = set(chosen) | set(extra)
        return SchedulePlan(chosen, tuple(r for r in resident if r not in kept),
                            tuple(extra))
