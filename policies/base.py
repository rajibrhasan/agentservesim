"""The contract: three bases, one per decision plane, and the records they read."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, TextIO

from .program import ProgramControlBlock, ProgramTable



@dataclass
class RetentionDecision:
    seq: int
    ts: float
    action: str  # "protect" | "release" | "evict" | "swap" | "none"
    program_id: str
    turn_idx: int
    request_id: str
    deadline_ts: Optional[float] = None
    tool_name: Optional[str] = None
    blocks: Optional[int] = None  # engine-reported count, filled by executor
    info: Optional[dict] = None  # policy inputs/scores, for parity replay

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


@dataclass
class SystemSignals:
    """System-level observations a policy may read at decision time, in
    addition to the program's own record. Filled by the executor from
    whatever the host provides (simulator memory model, or the engine's
    scheduler stats on the real side). None = not provided."""

    kv_utilization: Optional[float] = None  # fraction of the KV pool in use, 0..1
    ts: Optional[float] = None


@dataclass
class PolicyConfig:
    """Everything a flag can tune, in one object.

    A policy takes what it needs and ignores the rest. Without this, the only
    way to construct "the policy this flag names" was an if-chain per axis
    listing every value and its constructor arguments -- so the registry that
    maps a name to a class could not be the thing that builds it, and adding a
    policy meant editing the chain as well as the table.
    """
    #: None means the caller did not choose one; each policy applies its
    #: own. Never a magic number that is also a legal value.
    tau_s: Optional[float] = None
    pin_s: float = 2.0
    default_gap_s: float = 0.0
    min_waste_profile: Optional[str] = None
    num_instances: int = 1
    capacity_limit: Optional[float] = None
    oracle_table: object = None


class _Configurable:
    """Build from a `PolicyConfig`. The default ignores it, which is right for
    every policy that takes no tuning; the rest override."""

    @classmethod
    def from_config(cls, cfg: "PolicyConfig"):
        return cls()


class RetentionPolicy(_Configurable):
    """One decision per event; the executor applies and logs it."""

    # Engine-launch configuration this value requires.
    engine_flags: dict = {"enable_prefix_caching": True, "kv_protection": True}
    # Latest system signals (set by the executor before each decision).
    signals: SystemSignals = SystemSignals()
    # When the protection release fires: "arrival" (gap-scoped: released the moment the
    # program's next turn arrives, exposing the context to LRU for the whole queue wait)
    # or "scheduled" (queue-persistent: held until the next turn is actually admitted to
    # the running batch, the semantics of Continuum's released code). The host (real
    # driver or simulator adapter) reads this and routes the release event.
    release_event: str = "arrival"

    def observe_arrival(self, pcb: ProgramControlBlock, now: float) -> None:
        """Called at the next turn's ARRIVAL regardless of release_event,
        BEFORE the dispatch transition clears the gap record — the only
        point where the just-ended gap's duration is observable. Policies
        that learn tool times override this."""
        return None

    def on_turn_complete(
        self,
        pcb: ProgramControlBlock,
        request_id: str,
        now: float,
    ) -> Optional[tuple]:
        """Return (action, deadline_ts) or (action, deadline_ts, info) or None for no
        action. info is a JSON-serializable dict of the decision's inputs/scores,
        logged for parity replay."""
        return None

    def on_turn_arrival(
        self, pcb: ProgramControlBlock, now: float
    ) -> Optional[str]:
        """Return "release" to release the program's protected blocks."""
        return None


#: The plane-aligned name. The same class, not a subclass: a second
#: definition of a base is a second contract.
KVPolicy = RetentionPolicy


@dataclass
class PriorityStamp:
    seq: int
    ts: float
    program_id: str
    turn_idx: int
    priority: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


@dataclass
class QueueView:
    """Engine-side state visible to the admission gate at one scheduling
    tick. Aggregate only: nothing here identifies a program."""

    n_running: int          # requests currently in the running batch
    n_waiting: int          # requests in the waiting queue (after ordering)
    n_inflight: int         # batches in flight on the pipeline
    kv_utilization: float   # fraction of the KV pool in use, 0..1
    kv_free_tokens: int     # tokens allocatable without evicting anything
    kv_evictable_tokens: int  # tokens reclaimable from unreferenced cache
    prompt_tokens: int      # this waiting turn's prompt length
    cached_tokens: int      # of which already resident (prefix hit)


@dataclass
class VictimView:
    """One running request as seen by the preemption victim rule."""

    pcb: ProgramControlBlock
    priority: int
    prompt_tokens: int
    computed_tokens: int    # prompt + generated tokens held in KV
    generated_tokens: int
    is_prefill: bool


class SchedulingPolicy(_Configurable):
    """Priority for one turn at submission; None means do not stamp."""

    # Engine-launch configuration this value requires.
    engine_args: dict = {}

    def priority(
        self, pcb: ProgramControlBlock, now: float
    ) -> Optional[int]:
        """Priority for one turn at submission, from the record alone.
        None means do not stamp."""
        return None

    def victim(
        self, candidates: list, now: float
    ) -> Optional[int]:
        """Under memory pressure: index into `candidates` (VictimView
        list, running requests) of the request to preempt by recompute.
        None = engine default (largest priority, then latest arrival)."""
        return None

    def admit(
        self, pcb: ProgramControlBlock, now: float, view: QueueView
    ) -> bool:
        """Admission gate for one WAITING turn at a scheduling tick.
        False holds the turn at the gateway this tick (it keeps its
        place); True lets the engine try to fit it. Called in queue
        order; never called for preempted requests, and overridden to
        True for the queue head when the engine is idle."""
        return True


@dataclass
class RoutingDecision:
    seq: int
    ts: float
    program_id: str
    turn_idx: int
    instance: int
    info: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


class RoutingPolicy(_Configurable):
    """Pick an instance for one turn. Deterministic given the same
    event sequence: ties always break to the lowest index."""

    def __init__(self, num_instances: int) -> None:
        assert num_instances > 0
        self.num_instances = num_instances
        self.inflight = [0] * num_instances

    def _least_loaded(self) -> int:
        return min(range(self.num_instances), key=lambda i: (self.inflight[i], i))

    def route(
        self, pcb: ProgramControlBlock, now: float
    ) -> tuple[int, Optional[dict]]:
        """Place one turn. Program state comes from the PCB and nowhere
        else; instance state comes from self.inflight."""
        raise NotImplementedError

    def on_submit(self, instance: int) -> None:
        self.inflight[instance] += 1

    def on_complete(self, instance: int) -> None:
        assert self.inflight[instance] > 0
        self.inflight[instance] -= 1
