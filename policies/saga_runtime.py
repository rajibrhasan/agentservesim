
from dataclasses import dataclass
import math
import random
from typing import Optional, Tuple


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class Successor:
    probability: float
    overlap: float

    def __post_init__(self):
        if not (0 <= self.probability <= 1 and 0 <= self.overlap <= 1):
            raise ValueError('successor probability and overlap must be in [0, 1]')


@dataclass(frozen=True)
class CacheObservation:
    session: str
    last_access_s: float
    size_bytes: int
    successors: Tuple[Successor, ...]

    def __post_init__(self):
        _nonnegative(self.last_access_s, 'last access')
        if self.size_bytes <= 0:
            raise ValueError('cache entry must have positive size')
        if sum(s.probability for s in self.successors) > 1 + 1e-12:
            raise ValueError('successors must describe exclusive next-step transitions')


def eviction_order(entries, now, max_observed_idle_s):

    _nonnegative(now, 'time')
    _nonnegative(max_observed_idle_s, 'maximum observed idle time')
    entries = tuple(entries)
    if any(e.last_access_s > now for e in entries):
        raise ValueError('cache access cannot be in the future')
    if not entries:
        return ()
    # Current idle ages are observations too; never normalize by future gaps.
    scale = max(max_observed_idle_s, max(now - e.last_access_s for e in entries))
    largest = max(e.size_bytes for e in entries)

    def score(entry):
        recency = (now - entry.last_access_s) / scale if scale else 0.0
        reuse = sum(s.probability * s.overlap for s in entry.successors)
        return 0.3 * recency + 0.5 * (1 - reuse) + 0.2 * entry.size_bytes / largest

    return tuple(sorted(entries, key=lambda e: (-score(e), e.last_access_s, e.session)))


@dataclass(frozen=True)
class TaskEstimate:
    program_id: str
    tenant: str
    remaining_gpu_s: float
    deadline_s: float

    def __post_init__(self):
        _nonnegative(self.remaining_gpu_s, 'predicted remaining work')
        _nonnegative(self.deadline_s, 'deadline')


def fair_shares(tasks, now, overdue_slack_s):
  
    _nonnegative(now, 'time')
    if not math.isfinite(overdue_slack_s) or overdue_slack_s <= 0:
        raise ValueError('overdue slack must be positive and finite')
    scores, seen = {}, set()
    for task in tasks:
        if task.program_id in seen:
            raise ValueError('each program contributes to AFS exactly once')
        seen.add(task.program_id)
        slack = max(overdue_slack_s, task.deadline_s - now)
        scores[task.tenant] = scores.get(task.tenant, 0.0) + task.remaining_gpu_s / slack
    total = sum(scores.values())
    return {tenant: score / total if total else 0.0 for tenant, score in scores.items()}


@dataclass(frozen=True)
class WorkerObservation:
    worker: int
    load: float
    queued_sessions: Tuple[Tuple[str, float], ...] = ()
    empty_since_s: Optional[float] = None

    def __post_init__(self):
        _nonnegative(self.load, 'worker load')
        if self.empty_since_s is not None:
            _nonnegative(self.empty_since_s, 'queue empty time')
            if self.queued_sessions:
                raise ValueError('nonempty queue cannot have an empty-since time')
        for _, arrival in self.queued_sessions:
            _nonnegative(arrival, 'pending arrival')


@dataclass(frozen=True)
class Steal:
    session: str
    source: int
    destination: int


class SagaPlacement:
 

    def __init__(self, seed=0, affinity_limit=0.8, idle_s=0.1, load_ratio=2.0):
        if not 0 < affinity_limit <= 1 or idle_s < 0 or not math.isfinite(idle_s):
            raise ValueError('invalid affinity or idle threshold')
        if not math.isfinite(load_ratio) or load_ratio <= 1:
            raise ValueError('load ratio must be finite and greater than one')
        self.affinity_limit, self.idle_s, self.load_ratio = affinity_limit, idle_s, load_ratio
        self.random = random.Random(seed)
        self.home = {}
        self.pending = {}

    def route(self, session, workers, cached_workers):
        workers = tuple(workers)
        if not workers:
            raise ValueError('routing needs a worker')
        if len({w.worker for w in workers}) != len(workers):
            raise ValueError('duplicate worker observation')
        home = self.home.get(session)
        for worker in workers:
            if (worker.worker == home and home in cached_workers
                    and worker.load < self.affinity_limit):
                return home
        return min(workers, key=lambda w: (w.load, w.worker)).worker

    def propose_steal(self, destination, workers, now):
        _nonnegative(now, 'time')
        workers = tuple(workers)
        by_id = {w.worker: w for w in workers}
        if len(by_id) != len(workers):
            raise ValueError('duplicate worker observation')
        if any((w.empty_since_s is not None and w.empty_since_s > now)
               or any(arrival > now for _, arrival in w.queued_sessions) for w in workers):
            raise ValueError('worker observation cannot be in the future')
        target = by_id[destination]
        if (target.queued_sessions or target.empty_since_s is None
                or now - target.empty_since_s < self.idle_s
                or any(s.destination == destination for s in self.pending.values())):
            return None
        candidates = [w for w in workers if w.worker != destination
                      and w.load > self.load_ratio * target.load
                      and any(s not in self.pending for s, _ in w.queued_sessions)]
        if not candidates:
            return None
        source = self.random.choice(sorted(candidates, key=lambda w: w.worker))
        session, _ = min((s for s in source.queued_sessions if s[0] not in self.pending),
                         key=lambda s: (s[1], s[0]))
        proposal = Steal(session, source.worker, destination)
        self.pending[session] = proposal
        return proposal

    def complete_steal(self, proposal, published):
        if self.pending.get(proposal.session) != proposal:
            raise ValueError('stale or unknown migration acknowledgement')
        if published:
            self.home[proposal.session] = proposal.destination
        del self.pending[proposal.session]
