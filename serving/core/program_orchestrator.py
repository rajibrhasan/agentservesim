"""Program state, owned in one place.

The engine's other planes -- `program_kv`, `program_scheduler`, `program_router`
-- read program state from here and keep no copy of their own. That single rule
is the reason this module exists: the previous design had the harness own a
program table while the engine owned request state, and every divergence found
in September 2026 was a failure to reconcile the two, not a modelling error.

What lives here is the cross-turn record of a program: who it is, where it is in
its turn sequence, what service it has accrued, what its tool gaps have looked
like, which instance holds its context, how much KV it is charged for, and what
pins it holds.

What deliberately does NOT live here is instance state -- queue depth, in-flight
count, free blocks. That belongs to no single program, and letting it in turns
the record into a global scratchpad. A policy that needs it takes an explicit
probe.

The admissibility rule for anything proposed for this record:

    past-derived aggregates are admissible, future information is not.

A deployed system can observe its own tool-gap history, so the running mean
belongs here. It cannot observe this turn's output length or this gap's true
duration, so those never appear -- a policy that could read them would be
unimplementable on real hardware and every number measured with it meaningless.

This module owns state only. It does not decide anything: no eviction, no
ordering, no placement. Those are the planes' business, and the policies'.
It also does not know the policy contract -- the adapter projects a snapshot
onto whatever the contract wants, which keeps the engine independent of it.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


#: Sentinel for 'any runnable turn', distinct from a node whose id is None.
_ANY = object()




class Attribution(enum.Enum):
    """How a block shared by several programs is charged.

    A prefix-cache node covering N programs has one length and N owners, so a
    program's footprint is undefined until this is chosen. It is not a
    bookkeeping detail: a retention policy asked "how much is this program
    holding" gets materially different answers, most sharply for programs
    sharing a long system prompt -- which in an agent workload is all of them.

    FULL       charge the whole node to every owner. Sums exceed the pool; this
               is what the eviction trace does, correct for a diagnostic and
               unusable as an accounting basis.
    SPLIT      charge len/N to each owner. Sums to the pool, but a program's
               footprint moves when an unrelated program arrives or leaves.
    HOLDER     charge the whole node to the pin holder, nothing to the rest.
               Stable per program; unpinned sharers look free while genuinely
               depending on those blocks.

    Whichever is chosen becomes part of the contract's observable semantics: a
    retention policy is only reproducible across hosts if both charge shared
    prefixes the same way.
    """

    FULL = "full"
    SPLIT = "split"
    HOLDER = "holder"


@dataclass(frozen=True)
class PlannedTurn:
    """A turn the trace says will happen, before it is scheduled.

    `output_toks` is CUMULATIVE -- input plus generated -- because that is what
    the engine compares its computed-token counter against. The old loader
    applies `input_toks + output_toks` at ingestion (router.py:193); doing it
    anywhere else produces a completion rule that is wrong by the length of the
    prompt.

    `gap_ns` is how long the tool call after this turn takes. It is trace input,
    not something a policy may read: a PCB never carries this gap's true
    duration, because a deployed system cannot know it in advance.
    """

    node_id: object
    input_toks: int
    output_toks: int
    tool: Optional[str] = None
    #: (parent node_id, delay after that parent completes). A turn is runnable
    #: when EVERY parent has completed, at max(completion + delay) over them --
    #: the fan-in barrier. A chain is the path-graph case: one parent, whose
    #: delay is the tool gap. Roots have none and are runnable at arrival.
    parents: Tuple[Tuple[object, int], ...] = ()
    input_hash_ids: Tuple[int, ...] = ()
    model: Optional[str] = None


@dataclass(frozen=True)
class Pin:
    """One protection grant. Ordered, so pressure can choose whom to break."""

    request_id: str
    turn_idx: int
    granted_ts: float
    deadline_ts: Optional[float] = None


@dataclass(frozen=True)
class ProgramState:
    """The record. Frozen, replaced wholesale on every transition.

    Frozen for the same reason the contract's PCB is: a component holding a
    reference cannot write through it, and cannot accumulate hidden state
    behind it. Every mutation goes through the orchestrator, so the sequence of
    records is the whole history of the program and a decision taken from one
    is reproducible from it alone.
    """

    # identity
    program_id: str

    # position
    arrival_ts: float
    turn_idx: int = 0
    turns_completed: int = 0

    # service history
    attained_service_s: float = 0.0

    # tool state, and the history derived from it. `gap_sum_s`/`gap_count` are
    # here rather than inside a policy because a policy holding its own
    # per-program dictionary makes its decisions unreproducible from the record,
    # and then the two hosts silently learn different things from one trace.
    in_gap: bool = False
    tool_name: Optional[str] = None
    gap_started_ts: Optional[float] = None
    gap_sum_s: float = 0.0
    gap_count: int = 0

    # KV residency
    live_instance: Optional[int] = None
    instance_history: frozenset = frozenset()
    context_tokens: int = 0
    #: Offload and shared-prefix accounting. Written by `program_kv`, read by
    #: nothing yet: no cell runs with tiering on and every cell is
    #: single-instance, so neither the tier split nor the SPLIT/HOLDER
    #: attribution rule has been exercised end to end. Kept because they are
    #: the mechanism the design calls for; noted because "implemented" and
    #: "measured" are not the same claim.
    charged_tokens: int = 0                 # under the active Attribution rule
    tier_tokens: Tuple[int, ...] = ()       # per tier, index = Device value
    pins: Tuple[Pin, ...] = ()

    # turn queue: what the trace still owes this program, and when the next of
    # it becomes runnable. This is program lifecycle, so it lives here rather
    # than in the router -- the router owning `_deferred_sessions` was a second
    # owner of program state, the pattern this design exists to remove.
    pending: Tuple[PlannedTurn, ...] = ()
    #: node_id -> completion time, for turns already finished. A DAG needs the
    #: whole set, not just the latest: a fan-in node waits on the LAST of its
    #: parents, and which that is is not known until they have all run.
    completions: Tuple[Tuple[object, float], ...] = ()
    #: When the program's last turn finished. Job completion time is measured
    #: from `arrival_ts` to here, and it is the number the benchmark reports, so
    #: it belongs with the program rather than with whatever component happened
    #: to notice the finish.
    end_ts: Optional[float] = None

    def ready_ts(self, turn: "PlannedTurn") -> Optional[float]:
        """When this turn becomes runnable, or None while a parent is missing."""
        if not turn.parents:
            return self.arrival_ts
        return self._ready_with(turn, dict(self.completions))

    def _ready_with(self, turn: "PlannedTurn", done) -> Optional[float]:
        """`ready_ts` against a completions map the caller already built.

        `completions` is a tuple, so `dict(...)` is O(turns completed) and
        `ready_ts` paid it on EVERY call -- once per pending turn, per program,
        per step. On the board cell that was 418,837,913 calls and 654 s of a
        5,768 s run: eleven percent of the whole simulation spent rebuilding
        the same dictionary. The request planes have no orchestrator and pay
        none of it, which is most of why they finished the same cell in forty
        minutes against seventy.
        """
        if not turn.parents:
            return self.arrival_ts
        best = None
        for pid, delay in turn.parents:
            if pid not in done:
                return None                 # a parent has not finished
            t = done[pid] + delay
            if best is None or t > best:
                best = t                    # fan-in barrier: the LAST parent
        return best

    @property
    def next_available_ts(self) -> Optional[float]:
        """Earliest time any owed turn becomes runnable. None when every
        remaining turn is still waiting on a parent."""
        if not self.pending:
            return None
        done = dict(self.completions)       # once, not once per pending turn
        best = None
        for turn in self.pending:
            t = self._ready_with(turn, done)
            if t is not None and (best is None or t < best):
                best = t
        return best

    @property
    def mean_gap_s(self) -> Optional[float]:
        """Running mean tool-gap, or None before the first gap is observed.

        A learning retention policy reads this instead of keeping its own
        counters. None means "no history yet" and is distinct from 0.0.
        """
        return (self.gap_sum_s / self.gap_count) if self.gap_count else None

    @property
    def is_pinned(self) -> bool:
        return bool(self.pins)

    def oldest_pin(self) -> Optional[Pin]:
        """The pin granted longest ago. What a pressure policy breaks first if
        it has no better rule."""
        return min(self.pins, key=lambda p: p.granted_ts) if self.pins else None


class ProgramOrchestrator:
    """The single owner of program state for one cluster.

    Cluster-scoped, not instance-scoped: a program is one entity even when its
    turns could be placed anywhere. The KV plane is per-instance, which is why
    this class enforces the invariant that a program's *live* context sits on
    exactly one instance at a time -- see `place`. Under that invariant a
    footprint is a scalar rather than a vector over instances, and blocks left
    behind on a previous instance are ordinary cache entries with no live owner.
    """

    def __init__(self, attribution: Attribution = Attribution.SPLIT) -> None:
        self._programs: Dict[str, ProgramState] = {}
        #: program_id -> the full turn plan, which `pending` consumes.
        self._plan: Dict[str, Tuple[PlannedTurn, ...]] = {}
        #: program_id -> {node_id: the turn that follows it}
        self._succ: Dict[str, Dict[object, PlannedTurn]] = {}
        #: program_id -> {node_id: that turn}
        self._by_id: Dict[str, Dict[object, PlannedTurn]] = {}
        #: (program_id, node_id) -> the turn currently running. Not part of
        #: ProgramState because a fan-out has several turns in flight at once,
        #: and because it is transient: an entry exists exactly between
        #: `take_next` and `on_turn_complete`.
        self._in_flight: Dict[tuple, PlannedTurn] = {}
        #: tool name -> observed gap durations, summed and counted across EVERY
        #: program. Cluster-scoped on purpose; see `on_turn_arrival`.
        self._tool_sum: Dict[str, float] = {}
        self._tool_n: Dict[str, int] = {}
        self.attribution = attribution

    # ------------------------------------------------------------- reading

    def get(self, program_id: str) -> Optional[ProgramState]:
        return self._programs.get(program_id)

    def __contains__(self, program_id: str) -> bool:
        return program_id in self._programs

    def __len__(self) -> int:
        return len(self._programs)

    def all(self) -> Iterable[ProgramState]:
        return self._programs.values()

    def pinned(self) -> Iterable[ProgramState]:
        """Programs currently holding at least one pin. The candidate set a
        pressure policy chooses from."""
        return (p for p in self._programs.values() if p.pins)

    def on_instance(self, instance: int) -> Iterable[ProgramState]:
        """Programs whose live context is on this instance. Pressure is
        per-instance, so this is the set a pressure event concerns."""
        return (p for p in self._programs.values() if p.live_instance == instance)

    # ------------------------------------------------------------- writing

    def _put(self, state: ProgramState) -> ProgramState:
        self._programs[state.program_id] = state
        return state

    def admit_program(self, program_id: str, now: float) -> ProgramState:
        """First sighting of a program. Idempotent: a re-admission keeps the
        original arrival, since that is what job completion time is measured
        from."""
        existing = self._programs.get(program_id)
        if existing is not None:
            return existing
        return self._put(ProgramState(program_id=program_id, arrival_ts=now))

    def on_turn_arrival(self, program_id: str, turn_idx: int,
                        now: float) -> ProgramState:
        """A turn has entered the system and is queued.

        Closes any outstanding tool gap and folds its duration into the
        program's history. That fold happens HERE, in the engine, precisely so
        it cannot depend on a policy hook one host calls and the other does not.
        """
        p = self.admit_program(program_id, now)
        gap_sum, gap_count = p.gap_sum_s, p.gap_count
        if p.in_gap and p.gap_started_ts is not None:
            elapsed = max(0.0, now - p.gap_started_ts)
            gap_sum += elapsed
            gap_count += 1
            # The same duration, also recorded against the TOOL, cluster-wide.
            # Continuum keys its rule on the tool rather than the program, and
            # the two are different policies: on the board trace `pip` averages
            # 19.9 s against `sed` at 0.12 s, so a per-program mean over a
            # program's mix of tools decides differently for every program that
            # runs the same tool. Recorded here rather than in a policy, for the
            # same reason the per-program fold is here -- a quantity a policy
            # accumulates itself differs between two hosts depending on which
            # hooks each one happens to call.
            if p.tool_name is not None:
                self._tool_sum[p.tool_name] = (
                    self._tool_sum.get(p.tool_name, 0.0) + elapsed)
                self._tool_n[p.tool_name] = self._tool_n.get(p.tool_name, 0) + 1
        return self._put(replace(
            p,
            turn_idx=turn_idx,
            in_gap=False,
            gap_started_ts=None,
            gap_sum_s=gap_sum,
            gap_count=gap_count,
        ))

    def on_turn_scheduled(self, program_id: str, now: float) -> ProgramState:
        """The turn has joined the running batch.

        One defined moment, which is the point: "admitted" used to be a site you
        configured rather than an event that happened. The EVENT is what matters
        -- a queue-persistent retention policy releases here rather than at
        arrival, and the difference is the whole queue wait. Nothing is recorded
        on the program: whether a turn is running is the scheduler's `running`
        list, and a copy of it here would be a second record of one fact,
        updated by different code, which is the arrangement these planes exist
        to remove.
        """
        p = self._programs[program_id]
        return self._put(p)          # nothing on the record changes

    def on_turn_complete(self, program_id: str, now: float, service_s: float,
                         tool_name: Optional[str] = None,
                         has_more_turns: Optional[bool] = None,
                         node_id: object = None) -> ProgramState:
        """The turn's last token has been emitted.

        `service_s` is the compute this turn actually consumed; it accrues so a
        scheduling policy can rank by attained service without keeping its own
        accumulator.

        No gap duration is reported here: how long the following tool call takes
        is a property of the EDGE to the next turn, recorded when the trace was
        loaded. Completion only says which node finished and when; readiness is
        derived. Two ways to express one delay is how they drift apart.
        """
        p = self._programs[program_id]
        # Whether more turns follow is a fact about the program, not something
        # the caller should have to assert: the trace already said.
        more = bool(p.pending) if has_more_turns is None else has_more_turns
        turn = self._in_flight.pop((program_id, node_id), None)
        if turn is None:
            # A caller that did not name the node still ends a turn, and the
            # entry has to go: this map is the only thing here that is not
            # derived, so anything it keeps is kept forever. With one turn in
            # flight there is no ambiguity about which; with several -- a
            # fan-out -- there is, and guessing would attribute the wrong tool.
            mine = [k for k in self._in_flight if k[0] == program_id]
            if len(mine) == 1:
                turn = self._in_flight.pop(mine[0])
        if tool_name is None and turn is not None:
            tool_name = turn.tool
        return self._put(replace(
            p,
            turns_completed=p.turns_completed + 1,
            attained_service_s=p.attained_service_s + max(0.0, service_s),
            in_gap=more,
            tool_name=tool_name,
            gap_started_ts=now if more else None,
            # Recording WHICH node finished is what lets a fan-in turn know its
            # last parent has landed. A chain needs only the latest; a DAG needs
            # all of them, so the set is kept rather than a single timestamp.
            completions=p.completions + ((node_id, now),)
            if node_id is not None else p.completions,
            end_ts=None if more else now,
        ))

    # ------------------------------------------------------ turn queue

    def define_program(self, program_id: str, turns: Sequence[PlannedTurn],
                       arrival_ts: float) -> ProgramState:
        """Register a program and the turns the trace says it will run.

        The first turn becomes available at `arrival_ts`; each later one becomes
        available when its predecessor completes and that turn's tool gap has
        elapsed. Holding the whole chain here is what lets the orchestrator
        answer "when does anything next happen", which the main loop needs in
        order to fast-forward through idle gaps instead of stepping through them.
        """
        state = self._programs.get(program_id)
        if state is None:
            state = ProgramState(program_id=program_id, arrival_ts=arrival_ts)
        # The IMMUTABLE plan, kept beside `pending` because `pending` shrinks as
        # turns are released and the KV plane needs to look FORWARD: a turn's
        # generated tokens are, verbatim, part of its successor's prompt.
        self._plan[program_id] = tuple(turns)
        # node_id -> the turn that follows it. Built ONCE here, because the KV
        # plane asks for a turn's generated ids on every decode-step commit:
        # scanning the plan per call is O(turns) on the hottest path there is,
        # and halved throughput on a 40-turn trace.
        by_id = {t.node_id: t for t in turns}
        succ = {}
        for turn in turns:
            for parent_id, _ in turn.parents:
                if parent_id in by_id and parent_id not in succ:
                    succ[parent_id] = turn
        self._succ[program_id] = succ
        self._by_id[program_id] = by_id
        return self._put(replace(state, pending=tuple(turns), completions=()))

    def generated_ids(self, program_id: str, node_id: object,
                      count: int) -> Tuple[int, ...]:
        """The real token ids this turn will GENERATE, or () if not knowable.

        A turn's output is not invented text: the next turn's prompt is this
        turn's prompt, plus what this turn generated, plus the tool result. So
        the ids live in the successor's `input_hash_ids`, at exactly the offset
        where this turn's prompt ends.

        Without this the plane numbers generated tokens in a private range, and
        they become blocks nothing can ever match -- while still occupying the
        pool, and still refreshing their LRU stamp on every decode step. Active
        requests' unmatchable decode blocks then outrank a waiting program's
        matchable context, and the LRU takes the context instead: whole
        programs losing their prefix and re-prefilling from 48 tokens.

        Real vLLM has no such class of block. Every block it caches is hashed
        from real token ids and is matchable by whoever presents them, which for
        an agent's next turn is exactly this program.
        """
        if count <= 0:
            return ()
        from .router import _DERIVE_FROM_SUCCESSOR, _SYNTH_ID_BASE
        plan = self._plan.get(program_id)
        if not plan:
            return ()
        if _DERIVE_FROM_SUCCESSOR:
            nxt = self._succ.get(program_id, {}).get(node_id)
            cur = self._by_id.get(program_id, {}).get(node_id)
            if cur is not None and nxt is not None \
                    and cur.input_hash_ids and nxt.input_hash_ids:
                start = len(cur.input_hash_ids)
                taken = tuple(nxt.input_hash_ids[start:start + count])
                if len(taken) == count:
                    return taken
        # Default: ids unique to this turn. They occupy KV and match nothing,
        # which is what the replay does -- it fixes the output LENGTH and lets
        # vLLM generate its own content, while the next prompt is replayed from
        # the trace, so the two do not line up. Measured on the reference run's
        # requests.jsonl over 2,823 turns: cached_tokens tracks the previous
        # turn's PROMPT in 38.3% of turns and prompt + generated in 2.6%.
        # Deriving handed this plane cross-turn hits the real engine does not
        # get; the request plane was changed for the same reason (a47c58e), and
        # leaving the two planes disagreeing about it made them incomparable.
        # SIM_DERIVE_OUTPUT_IDS=1 restores derivation for modelling a
        # deployment, where the agent really does send its own text on.
        salt = abs(hash((program_id, node_id))) % 1_000_003
        return tuple(_SYNTH_ID_BASE + salt * 1_000_003 + i for i in range(count))

    def due(self, now: float) -> List[Tuple[str, PlannedTurn]]:
        """Programs whose next turn is runnable, oldest first."""
        out = []
        for p in self._programs.values():
            if not p.pending:
                continue
            # The completions map ONCE per program rather than once per pending
            # turn: `ready_ts` rebuilds it from a tuple on every call, and this
            # loop runs every step over every program.
            done = dict(p.completions)
            for turn in p.pending:
                ready = p._ready_with(turn, done)
                if ready is not None and ready <= now:
                    out.append((ready, p.program_id, turn))
        # Oldest-ready first. A DAG can have several turns of one program
        # runnable at once (a fan-out), so this is a list per program, not one.
        out.sort(key=lambda t: (t[0], str(t[1]), str(t[2].node_id)))
        return [(pid, turn) for _, pid, turn in out]

    def next_arrival_ts(self) -> Optional[float]:
        """When anything next becomes runnable, or None when nothing will.

        The main loop jumps the clock to this rather than stepping toward it: a
        tool gap is seconds of simulated time in which no NPU has work.
        """
        # Once per program, not twice: the filter and the value both called
        # `next_available_ts`, and it walks every pending turn.
        best = None
        for p in self._programs.values():
            if not p.pending:
                continue
            t = p.next_available_ts
            if t is not None and (best is None or t < best):
                best = t
        return best

    def take_next(self, program_id: str, now: float,
                  node_id: object = _ANY) -> Optional[PlannedTurn]:
        """Remove one runnable turn from the program and mark it queued.

        `node_id` names which, for a fan-out where several are runnable at once;
        the default takes the earliest-ready.
        """
        p = self._programs[program_id]
        if not p.pending:
            return None
        if node_id is _ANY:
            # Once per turn, not twice: `ready_ts` rebuilds the completions
            # dict on every call and this ran 15.8M times in one cell.
            done = dict(p.completions)
            ready = [(r, t) for t in p.pending
                     for r in (p._ready_with(t, done),) if r is not None]
            if not ready:
                return None
            turn = min(ready, key=lambda rt: (rt[0], str(rt[1].node_id)))[1]
        else:
            turn = next((t for t in p.pending if t.node_id == node_id), None)
            if turn is None:
                return None
        rest = tuple(t for t in p.pending if t is not turn)
        self._put(replace(p, pending=rest))
        # Remember which turn this is while it runs. Completion then only has
        # to say WHICH node finished; what that node was -- its tool, above all
        # -- is looked up here rather than carried through the engine and
        # handed back. A fact that makes a round trip through another plane is
        # a fact with two owners.
        self._in_flight[(program_id, turn.node_id)] = turn
        self.on_turn_arrival(program_id, p.turns_completed, now)
        return turn

    def tool_mean_gap_s(self, tool_name: Optional[str]) -> Optional[float]:
        """Mean observed duration of a tool, across all programs.

        None means the tool has not been seen yet, which is distinct from 0.0
        and is what a cold-start rule keys on. Mirrors
        `harness/program.py::ProgramTable.tool_mean_gap_s` -- the same quantity
        under the same name, because a policy reads it on both hosts.
        """
        n = self._tool_n.get(tool_name, 0)
        return (self._tool_sum[tool_name] / n) if n else None

    def has_pending(self) -> bool:
        """True while any program still owes a turn. The loop must not exit
        while a tool call is outstanding."""
        return any(p.pending for p in self._programs.values())

    # ------------------------------------------------------- KV residency

    def place(self, program_id: str, instance: int) -> ProgramState:
        """Record where this program's live context is.

        Enforces the invariant the KV plane depends on: live context sits on
        exactly one instance. Moving a program does not move its blocks, so the
        previous instance keeps unreferenced copies -- those are ordinary cache
        entries with no live owner and are charged to nobody. What must not
        happen silently is a program being considered live in two places at
        once, because then its footprint is a vector and every per-instance
        pressure decision about it is ill-posed.
        """
        p = self._programs[program_id]
        if p.live_instance is not None and p.live_instance != instance:
            # the old residency is stale from this moment, not shared
            p = replace(p, context_tokens=0, charged_tokens=0, tier_tokens=(),
                        pins=())
        return self._put(replace(
            p,
            live_instance=instance,
            instance_history=p.instance_history | {instance},
        ))

    def set_footprint(self, program_id: str, context_tokens: int,
                      charged_tokens: int,
                      tier_tokens: Tuple[int, ...] = ()) -> ProgramState:
        """Record what the KV plane says this program occupies.

        `context_tokens` is the program's own context length; `charged_tokens`
        is what it is charged under the active attribution rule, and the two
        differ exactly when prefixes are shared. Both are recorded because a
        policy reasoning about recompute cost wants the first and one reasoning
        about pressure wants the second.
        """
        p = self._programs[program_id]
        return self._put(replace(p, context_tokens=context_tokens,
                                 charged_tokens=charged_tokens,
                                 tier_tokens=tuple(tier_tokens)))

    def grant_pin(self, program_id: str, request_id: str, now: float,
                  deadline_ts: Optional[float] = None) -> ProgramState:
        """Protect this program's context. Replaces any pin held by the same
        request rather than stacking, so a re-protect is not a leak."""
        p = self._programs[program_id]
        kept = tuple(x for x in p.pins if x.request_id != request_id)
        pin = Pin(request_id=request_id, turn_idx=p.turn_idx,
                  granted_ts=now, deadline_ts=deadline_ts)
        return self._put(replace(p, pins=kept + (pin,)))

    def release_pin(self, program_id: str,
                    request_id: Optional[str] = None) -> ProgramState:
        """Drop one pin, or all of them when no request is named."""
        p = self._programs[program_id]
        kept = () if request_id is None else tuple(
            x for x in p.pins if x.request_id != request_id)
        return self._put(replace(p, pins=kept))

    def expired_pins(self, now: float) -> Iterable[Tuple[str, Pin]]:
        """(program_id, pin) for every pin past its deadline. The valve's
        default source of reclaimable protection when a policy declines to
        choose."""
        for p in self._programs.values():
            for pin in p.pins:
                if pin.deadline_ts is not None and pin.deadline_ts <= now:
                    yield p.program_id, pin

    # ------------------------------------------------------------ metrics

    def completed(self) -> List[ProgramState]:
        """Programs whose last turn has finished, in arrival order."""
        return sorted((p for p in self._programs.values() if p.end_ts is not None),
                      key=lambda p: (p.arrival_ts, p.program_id))

    def workflow_metrics(self) -> List[Dict[str, float]]:
        """Per-program job completion time.

        This is the benchmark's headline number, and every field of it is
        already program state: when the program arrived, when its last turn
        finished, how many turns it ran. It lived in the router only because the
        router happened to be what noticed a chain ending -- which is the same
        reason `_deferred_sessions` lived there, and the same mistake.
        """
        return [{"workflow_id": p.program_id,
                 "arrival_ns": int(p.arrival_ts),
                 "end_ns": int(p.end_ts),
                 "jct_ns": int(p.end_ts - p.arrival_ts),
                 "num_nodes": p.turns_completed}
                for p in self.completed()]

    @staticmethod
    def _percentile(sorted_vals: Sequence[float], p: float) -> float:
        """Linear-interpolated percentile of a pre-sorted list. Copied from the
        old router verbatim: a different interpolation would move every tail
        number the project has published."""
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(sorted_vals) - 1)
        if lo == hi:
            return float(sorted_vals[lo])
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

    def workflow_metrics_summary(self) -> Optional[Dict[str, float]]:
        """Aggregate JCT and throughput over completed programs.

        Here rather than in the router for the same reason the per-program rows
        are: every input is program state. The router reported it only because
        it held the completion records, which it held because it owned
        dependency release -- one misplacement pulling in the next.
        """
        rows = self.workflow_metrics()
        if not rows:
            return None
        jcts = sorted(m["jct_ns"] for m in rows)
        n = len(jcts)
        first_arrival = min(m["arrival_ns"] for m in rows)
        last_end = max(m["end_ns"] for m in rows)
        makespan_ns = max(1, last_end - first_arrival)
        return {
            "num_workflows": n,
            "jct_mean_ns": sum(jcts) / n,
            "jct_p50_ns": self._percentile(jcts, 50),
            "jct_p90_ns": self._percentile(jcts, 90),
            "jct_p99_ns": self._percentile(jcts, 99),
            "jct_min_ns": jcts[0],
            "jct_max_ns": jcts[-1],
            "makespan_ns": makespan_ns,
            "workflow_throughput_per_s": n / (makespan_ns / 1e9),
        }

    def has_workflow_metrics(self) -> bool:
        return bool(self.completed())

    def first_arrival_ts(self) -> float:
        """Earliest arrival across all programs, or 1 when there are none.

        The floor of 1 is the old router's and is kept: the main loop uses this
        to set its starting clock, and a zero there means 'unset' elsewhere.
        """
        arrivals = [p.arrival_ts for p in self._programs.values()]
        return max(1, min(arrivals)) if arrivals else 1

    def save_workflow_metrics(self, path: str) -> int:
        """Write one row per completed program, byte-compatible with the old
        router's output: the arena reads this file, so the format is a contract
        and not an implementation detail."""
        import os
        if not os.path.isabs(path):
            path = f"../{path}"
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        rows = self.workflow_metrics()
        with open(path, "w") as f:
            f.write("workflow_id,arrival_ns,end_ns,jct_ns,num_nodes\n")
            for m in rows:
                f.write(f"{m['workflow_id']},{m['arrival_ns']},{m['end_ns']},"
                        f"{m['jct_ns']},{m['num_nodes']}\n")
        return len(rows)

    # ----------------------------------------------------------- reporting

    def counters(self) -> Dict[str, int]:
        """Normalized mechanism counters this plane can witness."""
        return {
            "programs": len(self._programs),
            "pinned_programs": sum(1 for p in self._programs.values() if p.pins),
            "pins_held": sum(len(p.pins) for p in self._programs.values()),
        }
