
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .program_orchestrator import ProgramOrchestrator


@dataclass(frozen=True)
class InstanceView:
   

    instance: int
    programs: int          # programs whose live context is here
    running: int           # turns currently executing
    waiting: int           # turns queued
    pressure: float        # locked + pinned, as a fraction of the pool
    capacity: int          # max_num_seqs, what LOAD normalises by

    @property
    def load_score(self) -> float:
        raw = self.waiting * 4 + self.running
        return raw / self.capacity if self.capacity not in (0, float("inf")) else raw


class ProgramRouter:
    """Placement for one cluster."""

    def __init__(self, orchestrator: ProgramOrchestrator, n_instances: int,
                 schedulers: Optional[Sequence] = None,
                 capacity_limit: Optional[float] = None,
                 policy: str = "LOAD",
                 seed: int = 0,
                 route_fn: Optional[Callable] = None,
                 on_placed: Optional[Callable] = None) -> None:
        self.orch = orchestrator
        self.n_instances = max(1, n_instances)
        self.schedulers = list(schedulers or [])
        #: Pressure above which affinity stops being honoured. None means
        #: affinity always wins, which is the right default: the published
        #: policy that routes by program (SAGA) pins a session to an instance
        #: precisely so its context survives.
        self.capacity_limit = capacity_limit
        #: LOAD | RR | RAND | AFFINITY | CUSTOM. Default LOAD, matching the CLI
        #: default and the old router, so `--planes program` does not silently
        #: change placement. AFFINITY is the program-aware addition and is
        #: opt-in, which is also what makes it measurable: you can run the same
        #: cell both ways and see what keeping a program's context is worth.
        self.policy = (policy or "LOAD").upper()
        self._rr = 0
        import random
        self._rnd = random.Random(seed)
        self.route_fn = route_fn
        #: Told after every commit, so a policy that keeps its own per-instance
        #: load can keep it. Without this a least-loaded rule reads zeros.
        self.on_placed = on_placed
        self.counters: Dict[str, int] = {
            "routes": 0, "affinity_hits": 0, "affinity_breaks": 0,
            "policy_overrides": 0, "cold_placements": 0,
        }

    # ------------------------------------------------------------- views

    def views(self) -> List[InstanceView]:
        """One view per instance, built fresh. Never cached: a stale load
        reading is how a router sends three turns to the instance that looked
        idle a moment ago."""
        by_instance = {i: 0 for i in range(self.n_instances)}
        for p in self.orch.all():
            if p.live_instance is not None and p.live_instance in by_instance:
                by_instance[p.live_instance] += 1

        out = []
        for i in range(self.n_instances):
            sched = self.schedulers[i] if i < len(self.schedulers) else None
            running = len(sched.running) if sched is not None else 0
            waiting = len(sched.waiting) if sched is not None else 0
            pressure = sched.kv.pressure() if sched is not None else 0.0
            capacity = getattr(sched, "max_num_seqs", 0) if sched is not None else 0
            out.append(InstanceView(instance=i, programs=by_instance[i],
                                    running=running, waiting=waiting,
                                    pressure=pressure, capacity=capacity))
        return out

    # ------------------------------------------------------------- route

    def route(self, program_id: str, now: float,
              only: Optional[Sequence[int]] = None) -> int:
        """Where this program's next turn should run."""
        self.counters["routes"] += 1
        views = self.views()
        if only is not None:
            allowed = set(only)
            narrowed = [v for v in views if v.instance in allowed]
            if narrowed:
                views = narrowed
        state = self.orch.get(program_id)

        if self.route_fn is not None:
            try:
                chosen = self.route_fn(state, views, now)
            except Exception:          # a bad candidate must not stall routing
                chosen = None
            allowed_ids = {v.instance for v in views}
            if chosen is not None and int(chosen) in allowed_ids:
                self.counters["policy_overrides"] += 1
                return self._commit(program_id, int(chosen), state)

        return self._commit(program_id, self._default(state, views), state)

    def _default(self, state, views: Sequence[InstanceView]) -> int:
        """Placement under the configured policy.

        AFFINITY is the program-aware one and the only one that reads program
        state: a turn goes back to the instance holding its context, because a
        turn placed away from its prefix pays the whole prompt again. The other
        three are the old router's, reproduced so that a flag means on this path
        what it means on the other.
        """
        allowed = {v.instance for v in views}
        if self.policy == "AFFINITY":
            home = state.live_instance if state is not None else None
            if home is not None and home in allowed:
                if self.capacity_limit is None:
                    return home
                here = next(v for v in views if v.instance == home)
                if here.pressure <= self.capacity_limit:
                    return home
                # Over the limit: affinity yields -- the invariant break. Fall
                # back only among instances that are UNDER the limit: plain LOAD
                # weighs queues, not memory, so on an idle-but-full instance it
                # would hand the program straight back to the one affinity just
                # declined, and the break would have bought nothing.
                under = [v for v in views if v.pressure <= self.capacity_limit]
                if under:
                    return self._load(under)
            return self._load(views)

        if self.policy == "RR":
            ids = sorted(allowed)
            chosen = ids[self._rr % len(ids)]
            self._rr = (chosen + 1) % self.n_instances
            return chosen

        if self.policy == "RAND":
            return self._rnd.choice(sorted(allowed))

        return self._load(views)

    def _load(self, views: Sequence[InstanceView]) -> int:
        """Least loaded among `views`, scanning from a rotating start so equal
        scores spread rather than all landing on the first -- the old router
        advances the same counter for the same reason.

        Takes a view list rather than reading them all, so a caller can narrow
        the candidates (affinity falling back only to instances under the
        pressure limit) without a second scoring function.
        """
        n = len(views)
        start = self._rr % n
        best_i, best = start, float("inf")
        for offset in range(n):
            i = (start + offset) % n
            if views[i].load_score < best:
                best, best_i = views[i].load_score, i
        chosen = views[best_i].instance
        self._rr = (chosen + 1) % self.n_instances
        return chosen

    def _commit(self, program_id: str, instance: int, state) -> int:
        """Record the placement and keep the orchestrator's invariant true.

        `place` is what clears a stale footprint when a program moves, so a
        router that chose without committing here would leave the KV plane
        charging a program for context on an instance it has left.
        """
        home = state.live_instance if state is not None else None
        if home is None:
            self.counters["cold_placements"] += 1
        elif home == instance:
            self.counters["affinity_hits"] += 1
        else:
            # Moving a program with live context discards its prefix. Counted,
            # because a silent break looks identical to a cache miss later.
            self.counters["affinity_breaks"] += 1
        self.orch.place(program_id, instance)
        if self.on_placed is not None:
            try:
                self.on_placed(instance)
            except Exception:
                pass          # a bad candidate must not stall placement
        return instance

    # ---------------------------------------------------------- reporting

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counters)


# ---------------------------------------------------------------- dispatch

def dispatch(router: "ProgramRouter", schedulers: Sequence, now: float,
             model: str = "") -> int:
    """Place every turn that has become runnable, and enqueue it.

    This is the old `route_arrived_requests` with the release half removed: what
    is runnable is a fact about the program, so the orchestrator decides it.
    """
    placed = 0
    for program_id, turn in router.orch.due(now):
        instance = router.route(program_id, now)
        sched = schedulers[instance]
        taken = router.orch.take_next(program_id, now, node_id=turn.node_id)
        if taken is None:
            continue
        sched.add_request(
            [f"{program_id}:{taken.node_id}", taken.model or model or sched.model,
             taken.input_toks, taken.output_toks, now, instance,
             list(taken.input_hash_ids), []],
            session_id=program_id, sub_request_index=taken.node_id)
        placed += 1
    return placed


def transfer_prefill(router: "ProgramRouter", requests: Sequence,
                     schedulers: Sequence) -> int:
   
    decode_ids = [i for i, s in enumerate(schedulers) if s.pd_type == "decode"]
    if not decode_ids:
        decode_ids = list(range(len(schedulers)))
    moved = 0
    for req in requests:
        pid = req.session_id or req.workflow_id
        instance = (router.route(pid, float(req.arrival), only=decode_ids)
                    if pid is not None else decode_ids[0])
        schedulers[instance].add_decode(req)
        moved += 1
    return moved
