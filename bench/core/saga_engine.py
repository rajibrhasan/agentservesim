"""SAGA gateway for the bench replay: routing, stealing, WA-LRU order, AFS, prefetch.

Every input is observed online: engine protection stats (which sessions are
cached where, pool utilization), the gateway's own queues, completed tool
gaps, observed program lengths and tool-result lengths. Trace lookahead
(future tool durations, future turns, future outputs) is never read.

Decision functions come from policies.saga_runtime; this module supplies the
observations, applies the decisions, publishes the WA-LRU order to each
engine's safety valve (kv_reclaim_order), and moves KV only through
acknowledged transfers (SagaCoordinator). Retention (the per-tool TTL) stays
with the retention policy the driver already runs.

Port choices, all visible here: the AFS share bounds a tenant's in-flight
calls at the gateway; SAGA's 500 ms preemption of a running low-share call
needs active-call migration and is not implemented. Stealing moves calls
held at the gateway (its queue), not calls already inside an engine.
"""
import asyncio
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from policies.saga_runtime import (CacheObservation, Successor, TaskEstimate, WorkerObservation,
                                   eviction_order, fair_shares)


@dataclass
class SagaConfig:
    routing_only: bool = False  # placement + retention; no stealing/migration/AFS/prefetch
    epoch_s: float = 0.1
    idle_s: float = 0.1
    load_ratio: float = 2.0
    affinity_limit: float = 0.8
    capacity_limit: Optional[int] = None   # in-flight calls an engine accepts; None = no holds
    overdue_slack_s: Optional[float] = None  # required when any program has a deadline
    ema: float = 0.2
    prefetch: bool = False
    prefetch_margin_s: float = 0.5

    def __post_init__(self):
        for name in ('epoch_s', 'idle_s', 'prefetch_margin_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 < self.ema <= 1:
            raise ValueError('ema must be in (0, 1]')
        if self.capacity_limit is not None and self.capacity_limit <= 0:
            raise ValueError('capacity_limit must be positive')


@dataclass
class Program:
    program_id: str
    tenant: str = 'default'
    deadline_s: Optional[float] = None
    turns_completed: int = 0
    context_tokens: int = 0
    last_completed_s: Optional[float] = None
    gap_started_s: Optional[float] = None
    tool: Optional[str] = None
    history: tuple = ()            # tokens of the last completed call plus its outputs
    home: Optional[int] = None
    prefetched: set = field(default_factory=set)
    finished: bool = False
    active: bool = False


@dataclass
class Held:
    program_id: str
    turn: int
    tokens: tuple
    ready_at: float
    worker: int
    admitted: asyncio.Event = field(default_factory=asyncio.Event)


class SagaGateway:
    """Gateway-side SAGA over several engines; engine access goes through
    `stats(i)`, `call(i, method, *args)` and `generate(i, ...)` so tests can
    substitute fakes."""

    def __init__(self, num_engines, config, placement, coordinator, *,
                 call, stats=None, generate=None, block_bytes=None, tool_ttl_s=None):
        if num_engines <= 0:
            raise ValueError('SAGA needs at least one engine')
        self.n, self.cfg = num_engines, config
        self.placement, self.coordinator = placement, coordinator
        # Routing-only runs sample on ready turns at most once per epoch.
        # Full-system runs also refresh for their background epoch work.
        self.latest_stats = {}
        self._observation_time = None
        # Bind to the running loop when refreshing, not when a synchronous
        # caller constructs the gateway (Python 3.9 binds locks eagerly).
        self._observation_lock = None
        self._stats = stats or (lambda instance: self.latest_stats.get(instance, {}))
        self._refresh = stats is None
        self._call, self._generate = call, generate
        self._block_bytes = block_bytes or (lambda instance: 1)
        self._tool_ttl_s = tool_ttl_s
        self.programs = {}
        self.inflight = [0] * num_engines
        self.held = [deque() for _ in range(num_engines)]
        self.empty_since = [0.0] * num_engines
        self.inflight_by_tenant = {}
        self.shares = {}
        # Online estimators: how many calls a program makes, how long a tool
        # result is, and whether a completed call is followed by another.
        self.mean_turns = None
        self.mean_result_tokens = {}
        self.continue_rate = None
        self.mean_service_s = None
        self.stats = {'steals': 0, 'orders_published': 0, 'prefetches': 0,
                      'holds': 0, 'afs_holds': 0, 'stolen': [],
                      'routes_by_instance': [0] * self.n}
        self._task = None
        self.failure = None
        self._published_shares = {}
        self._prefetch_handles = set()

    # -- program bookkeeping -------------------------------------------------
    def register(self, program_id, tenant='default', deadline_s=None):
        if program_id in self.programs:
            raise ValueError('program already registered')
        if deadline_s is not None and self.cfg.overdue_slack_s is None:
            raise ValueError('programs with deadlines need SagaConfig.overdue_slack_s')
        self.programs[program_id] = Program(program_id, tenant or 'default', deadline_s)

    def _observe_workers(self, now):
        workers, cached = [], {}
        for i in range(self.n):
            st = self._stats(i) or {}
            total = int(st.get('num_gpu_blocks', 0) or 0)
            free = int(st.get('free_queue_blocks', 0) or 0)
            load = max(0.0, min(1.0, 1 - free / (total - 1))) if total > 1 else 0.0
            if self.cfg.capacity_limit is not None:
                # Occupancy is part of load: a full engine is loaded even when its
                # KV pool is mostly free, as with few long calls.
                load = max(load, min(1.0, self.inflight[i] / self.cfg.capacity_limit))
            queued = tuple((h.program_id, h.ready_at) for h in self.held[i])
            empty_since = None if (queued or self.inflight[i]) else self.empty_since[i]
            workers.append(WorkerObservation(i, load, queued, empty_since))
            for program, count in (st.get('policy_observation') or {}).get(
                    'cached_blocks_by_program', {}).items():
                if count:
                    cached.setdefault(program, set()).add(i)
        return workers, cached

    def route(self, program_id, now):
        workers, cached = self._observe_workers(now)
        instance = self.placement.route(program_id, workers, cached.get(program_id, set()))
        from runtime.routing_trace import record_saga_route
        record_saga_route('real', program_id, now, self._observation_time,
                          self.placement.home.get(program_id), workers,
                          cached.get(program_id, set()), instance)
        self.stats['routes_by_instance'][instance] += 1
        return instance

    # -- admission with capacity and AFS holds --------------------------------
    def _quota_ok(self, tenant):
        share = self.shares.get(tenant)
        if share is None or self.cfg.capacity_limit is None:
            return True
        allowed = max(1, math.ceil(share * self.cfg.capacity_limit * self.n))
        return self.inflight_by_tenant.get(tenant, 0) < allowed

    def _capacity_ok(self, instance):
        return self.cfg.capacity_limit is None or self.inflight[instance] < self.cfg.capacity_limit

    async def acquire(self, program_id, turn, instance, tokens, now):
        """Admit a ready call to `instance`, or hold it until capacity/quota allow.

        Returns the instance that finally runs it (a steal may move it)."""
        program = self.programs[program_id]
        self._check_failure()
        if self._capacity_ok(instance) and self._quota_ok(program.tenant):
            self._admit(program, instance)
            return instance
        held = Held(program_id, turn, tuple(tokens), now, instance)
        self.held[instance].append(held)
        self.stats['holds'] += 1
        if not self._quota_ok(program.tenant):
            self.stats['afs_holds'] += 1
        self.coordinator.enqueue(program_id, f'{program_id}:{turn}', tokens, now, instance)
        await held.admitted.wait()
        self._check_failure()
        return held.worker

    def _check_failure(self):
        if self.failure is not None:
            raise RuntimeError('SAGA gateway stopped after a failed epoch') from self.failure

    def _admit(self, program, instance):
        self.inflight[instance] += 1
        self.inflight_by_tenant[program.tenant] = self.inflight_by_tenant.get(program.tenant, 0) + 1
        program.home = instance
        self.placement.home[program.program_id] = instance
        program.active = True

    def _drain_holds(self, now):
        for i in range(self.n):
            queue = self.held[i]
            for held in list(queue):
                program = self.programs[held.program_id]
                # An urgent tenant must reach the engine to trigger local
                # preemption; a full gateway must not hide it indefinitely.
                urgent = (now - held.ready_at > 0.5 and any(
                    p.active and p.home == i
                    and self.shares.get(p.tenant, 0) < self.shares.get(program.tenant, 0)
                    for p in self.programs.values()))
                if not ((self._capacity_ok(i) or urgent) and self._quota_ok(program.tenant)):
                    continue
                queue.remove(held)
                self.coordinator.take(held.program_id, i)
                self._admit(program, i)
                held.worker = i
                held.admitted.set()

    def release(self, program_id, instance, now, service_s, prompt_tokens, output_tokens,
                tool=None, last_turn=False):
        program = self.programs[program_id]
        program.active = False
        self.inflight[instance] -= 1
        self.inflight_by_tenant[program.tenant] -= 1
        if not self.inflight[instance] and not self.held[instance]:
            self.empty_since[instance] = now
        program.turns_completed += 1
        program.last_completed_s = now
        program.history = tuple(prompt_tokens) + tuple(output_tokens)
        program.context_tokens = len(program.history)
        program.tool = tool
        program.gap_started_s = None if last_turn else now
        self.mean_service_s = self._ema(self.mean_service_s, service_s)
        if last_turn:
            program.finished = True
            self.mean_turns = self._ema(self.mean_turns, program.turns_completed)
        self.continue_rate = self._ema(self.continue_rate, 0.0 if last_turn else 1.0)

    def observe_result(self, program_id, result_tokens):
        """A tool result arrived: learn its length for the overlap estimate."""
        program = self.programs[program_id]
        if program.tool is not None:
            self.mean_result_tokens[program.tool] = self._ema(
                self.mean_result_tokens.get(program.tool), result_tokens)

    def _ema(self, old, value):
        return value if old is None else old + self.cfg.ema * (value - old)

    # -- epoch work ------------------------------------------------------------
    def _cache_entries(self, instance, now):
        st = self._stats(instance) or {}
        by_tag = (st.get('policy_observation') or {}).get('protected_blocks_by_tag', {})
        block_bytes = self._block_bytes(instance)
        entries = []
        for tag, count in by_tag.items():
            program = self.programs.get(tag.rsplit(':', 1)[0])
            if program is None or not count:
                continue
            last = program.last_completed_s if program.last_completed_s is not None else now
            expected_result = self.mean_result_tokens.get(program.tool)
            if program.finished:
                successors = ()
            else:
                p_next = 1.0 if self.continue_rate is None else self.continue_rate
                if expected_result is None or program.context_tokens == 0:
                    overlap = 1.0
                else:
                    overlap = program.context_tokens / (program.context_tokens + expected_result)
                successors = (Successor(min(1.0, p_next), overlap),)
            entries.append(CacheObservation(tag, min(last, now), max(1, count * block_bytes),
                                            successors))
        return entries

    async def publish_orders(self, now):
        for i in range(self.n):
            entries = self._cache_entries(i, now)
            if not entries:
                continue
            idle = max(now - e.last_access_s for e in entries)
            order = [e.session for e in eviction_order(entries, now, idle)]
            await self._call(i, 'kv_reclaim_order', order)
            self.stats['orders_published'] += 1

    def update_shares(self, now):
        tasks = []
        for program in self.programs.values():
            if program.finished or program.deadline_s is None:
                continue
            expected = self.mean_turns if self.mean_turns is not None else program.turns_completed + 1
            remaining_calls = max(1.0, expected - program.turns_completed)
            step_s = self.mean_service_s if self.mean_service_s is not None else 1.0
            tasks.append(TaskEstimate(program.program_id, program.tenant,
                                      remaining_calls * step_s, program.deadline_s))
        self.shares = fair_shares(tasks, now, self.cfg.overdue_slack_s) if tasks else {}

    async def steal(self, now):
        workers, _ = self._observe_workers(now)
        loads = [w.load for w in workers]
        empty = [w.empty_since_s for w in workers]
        for i in range(self.n):
            if self.held[i] or self.inflight[i]:
                continue
            record = await self.coordinator.steal(i, loads, empty, now)
            if record is None:
                continue
            for source in range(self.n):
                for held in list(self.held[source]):
                    if held.program_id == record.session:
                        self.held[source].remove(held)
                        held.worker = i
                        self.held[i].append(held)
            self.stats['steals'] += 1
            self.stats['stolen'].append(record.session)

    async def prefetch(self, now):
        if not self.cfg.prefetch or self._generate is None or self._tool_ttl_s is None:
            return
        for program in self.programs.values():
            if (program.finished or program.gap_started_s is None or program.home is None
                    or program.turns_completed in program.prefetched):
                continue
            expected = self._tool_ttl_s(program.tool)
            if expected is None or now < program.gap_started_s + expected - self.cfg.prefetch_margin_s:
                continue
            st = self._stats(program.home) or {}
            by_tag = (st.get('policy_observation') or {}).get('protected_blocks_by_tag', {})
            if any(count and tag.rsplit(':', 1)[0] == program.program_id
                   for tag, count in by_tag.items()):
                continue  # still resident: nothing to prefetch
            program.prefetched.add(program.turns_completed)
            self.stats['prefetches'] += 1
            instance = program.home
            tag = await self._generate(instance, program.program_id, program.turns_completed,
                                       list(program.history))
            if tag is not None:
                self._prefetch_handles.add((instance, tag))

    async def refresh_stats(self):
        observations = await asyncio.gather(*(
            self._call(i, 'kv_protection_stats') for i in range(self.n)))
        for st in observations:
            if 'cached_blocks_by_program' not in (st or {}).get('policy_observation', {}):
                raise RuntimeError('SAGA requires cached_blocks_by_program observations; '
                                   'restart replay with the updated SagaScheduler')
        self.latest_stats = {i: dict(st or {}) for i, st in enumerate(observations)}

    async def refresh_for_route(self, now):
        """Sample on the first ready turn of each observation window.

        The simulator uses the same demand-driven 100 ms window. Publish all
        workers together, so callbacks never see a partially refreshed view.
        """
        if not self._refresh:
            return
        if self._observation_lock is None:
            self._observation_lock = asyncio.Lock()
        async with self._observation_lock:
            if (self._observation_time is None
                    or now - self._observation_time >= self.cfg.epoch_s):
                await self.refresh_stats()
                self._observation_time = now

    async def epoch(self, now):
        if self._refresh and not self.cfg.routing_only:
            await self.refresh_stats()
        if self.cfg.routing_only:
            self._drain_holds(now)
            return
        self.update_shares(now)
        if self.shares != self._published_shares:
            for i in range(self.n):
                await self._call(i, 'policy_migration', 'afs_update', [self.shares])
            self._published_shares = dict(self.shares)
        self._drain_holds(now)
        await self.steal(now)
        self._drain_holds(now)
        await self.publish_orders(now)
        await self.prefetch(now)

    async def run(self, clock):
        try:
            while True:
                await asyncio.sleep(self.cfg.epoch_s)
                await self.epoch(clock())
        except asyncio.CancelledError:
            pass
        except Exception as error:
            # A failed decision or transfer must stop the replay with its
            # cause, not leave held calls waiting forever.
            self.failure = error
            for queue in self.held:
                for held in queue:
                    held.admitted.set()

    def start(self, clock):
        self._task = asyncio.get_event_loop().create_task(self.run(clock))

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        for instance, tag in tuple(self._prefetch_handles):
            await self._call(instance, 'kv_release', tag)
            self._prefetch_handles.remove((instance, tag))
