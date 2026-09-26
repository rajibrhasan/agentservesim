"""Real-side driver for the agent serving policy tuple
(retention, scheduling, routing).

Mirrors ``serving/core/unified_policy_adapter.py`` (the simulator side) on top
of one or more in-process AsyncLLM engines: the SAME harness policy
objects (``agentservesim/harness``) decide, the same Program Control
Block table is shared by the three executors, and the same events fire
at the same points of a program's life:

    turn ready   -> route (instance)  -> arrival release + priority stamp
    turn done    -> service accrual   -> retention decision (protect/evict)

Transport differences from the simulator are confined here:
  * KV protection goes to the engine through EngineCore utilities
    (``kv_protect`` / ``kv_release`` / ``kv_evict``, branch agent-knobs,
    gated by VLLM_KV_PROTECTION=1), addressed by the ``kv_tag`` the turn
    was submitted with (``{program_id}:{turn_idx}``).
  * The executors are synchronous and call the transport inline, while
    AsyncLLM utilities are coroutines. Executor calls therefore run on a
    single worker thread and the transport hops back onto the event loop
    with ``run_coroutine_threadsafe``; one worker keeps the shared PCB
    table single-threaded.
  * ``now`` is wall-clock seconds on the event loop's monotonic clock,
    the clock the engine's parked-block deadlines (time.time()) are
    compared against is translated at the transport.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import policies as _policies
from policies.base import PolicyConfig as _PolicyConfig

#: The values each axis accepts, from the policy registry rather than a list
#: kept in step by hand. The hand-written tuples that used to live here were
#: read by nothing -- runner.py carries its own argparse `choices` -- and had
#: already drifted: SCHEDULING_VALUES omitted "evolved", which the driver
#: below builds and every evolved-gate replay ran.
RETENTION_VALUES = tuple(_policies.choices_for("kv"))
SCHEDULING_VALUES = tuple(_policies.choices_for("scheduling"))
ROUTING_VALUES = tuple(_policies.choices_for("routing"))


def default_harness_root() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(os.path.dirname(repo_root), "agentservesim")


def import_harness(harness_root: Optional[str] = None):
    """Put the policy package on sys.path and import it.

    Mirrors serving/core/unified_policy_adapter.py::import_harness, and must:
    the 2026-09-13 reorganisation (policies by paper, not by axis) dissolved
    `policies.retention` / `.scheduling` / `.routing` and moved `waste_model`
    under `policies.utils`, the simulator side was updated and this was not, so
    every replay died at import for two days (first hit by job 42265618 after
    31 minutes of RTX time). The axis names now resolve to the package itself,
    which binds every class the driver looks up by attribute.

    `root` is a place to import FROM, not a checkout to validate -- it may hold
    a whole `policies/` package or a single module naming one class, which is
    what a staged search candidate looks like. The old os.path.isdir check on
    `harness/` rejected the latter.
    """
    root = harness_root or default_harness_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        import policies
    except ImportError as e:
        raise FileNotFoundError(
            f"cannot import `policies` with {root!r} on the path. Run from the "
            f"AgentServingSim checkout, or pass --harness-root at a directory "
            f"that has it.") from e
    from policies import base, program                   # noqa: F401
    from policies.utils import waste_model               # noqa: F401
    return policies, policies, policies, waste_model, program


def _require_tau(tau_s, name):
    """A TTL-family window the caller must choose.

    `ttl` and `saga-ttl` have no intrinsic window -- TTLRetention takes tau_s
    with no default -- so an omitted value used to fall through to the CLI's
    60.0, a figure no paper or measurement supports. That is how SAGA ran at 60
    against a real leg at 2 and the gap read as a fidelity problem for most of a
    day. Failing here costs a second; running the wrong policy costs an hour and
    looks like a result.
    """
    if tau_s is None:
        raise ValueError(
            f"--retention {name} needs --retention-tau: it has no default window. "
            f"The agent tuple is 2; SAGA's paper caps a learned per-tool p95 at 300.")
    return tau_s


def _evolved_class(module, axis):
    """Load staged candidates after the policy-package reorganisation.

    Published policies are exported by policies; scratch candidates still
    live in harness/evolved_<axis>.py, as the engine admission gate expects.
    """
    name = "Evolved" + axis.capitalize()
    cls = getattr(module, name, None)
    if cls is None:
        cls = getattr(importlib.import_module(f"harness.evolved_{axis}"), name)
    return cls


def engine_flags(retention_value: str, scheduling_value: str) -> dict:
    """Engine-launch settings the tuple requires (checked before boot)."""
    flags = {"enable_prefix_caching": True, "kv_protection": False,
             "scheduling_policy": "fcfs"}
    if retention_value == "evict-always":
        flags["enable_prefix_caching"] = False
    elif retention_value in ("ttl", "min-waste", "continuum", "saga-ttl",
                             "saga-tool-ttl", "evolved", "gate", "search-seed"):
        flags["kv_protection"] = True
    if scheduling_value in ("program-fcfs", "plas", "continuum", "evolved", "gate", "search-seed"):
        flags["scheduling_policy"] = "priority"
    return flags


# VLLM_KV_RELEASE_AT_ARRIVAL: releases decided by the retention policy at
# turn arrival are not sent as an RPC (that would land before the turn is
# submitted). They are parked here by program and the runner attaches the
# tag to the next turn's SamplingParams.extra_args["kv_release_tag"], which
# Scheduler.add_request applies in the same event the turn enters waiting.
# kv_tag format "<program>:<turn_idx>" (see runner.py).
DEFERRED_RELEASES: dict[str, str] = {}


def take_deferred_release(program_id: str):
    return DEFERRED_RELEASES.pop(program_id, None)


class AsyncEngineKVControl:
    """KVControl over an AsyncLLM engine, callable from the executor
    worker thread. Deadlines arrive on the loop clock and are converted
    to time.time() for the engine."""

    def __init__(self, engine, loop: asyncio.AbstractEventLoop,
                 clock_offset: float) -> None:
        self._core = engine.engine_core
        self._loop = loop
        self._offset = clock_offset  # time.time() - loop.time()

    def _call(self, method: str, *args) -> Any:
        started = self._loop.time()
        fut = asyncio.run_coroutine_threadsafe(
            self._core.call_utility_async(method, *args), self._loop)
        result = None
        succeeded = False
        try:
            result = fut.result(timeout=30.0)
            succeeded = True
            return result
        finally:
            path = os.environ.get('BENCH_KV_RPC_TRACE')
            if path:
                with open(path, 'a') as stream:
                    stream.write(json.dumps(dict(method=method, started_ts=started,
                        finished_ts=self._loop.time(), request_id=args[0] if args else None,
                        succeeded=succeeded,
                        affected_blocks=result if method in ('kv_protect', 'kv_release', 'kv_evict') else None,
                        deadline_ts=(args[1] - self._offset if method == 'kv_protect' else None))) + '\n')

    def protect(self, request_id: str, deadline_ts: float) -> int:
        return int(self._call("kv_protect", request_id,
                              float(deadline_ts + self._offset)))

    def release(self, request_id: str) -> int:
        # VLLM_KV_RELEASE_AT_ARRIVAL: the scheduler releases the pin when
        # the next turn is added (same event as the simulator); an RPC
        # here would land before the turn is submitted. Harness state
        # still flips to unprotected (note_retention) on the gateway.
        if os.environ.get("VLLM_KV_RELEASE_AT_ARRIVAL"):
            DEFERRED_RELEASES[request_id.rsplit(":", 1)[0]] = request_id
            return 0
        return int(self._call("kv_release", request_id))

    def evict(self, request_id: str) -> int:
        return int(self._call("kv_evict", request_id))

    def cancel_provisional(self, request_id: str) -> int:
        # A rejected completion-time pin must be released now, even when
        # ordinary successor releases are deferred to engine admission.
        return int(self._call("kv_release", request_id))

    def swap(self, request_id: str) -> int:
        # The mirror's swap has no gateway counterpart: InferCept swaps inside
        # the engine (bench.core.infercept_scheduler). The mirror only emits it
        # when the simulator adapter supplies a budget, so this is not reached.
        raise RuntimeError('swap is a simulator-side retention outcome')

    def stats(self) -> dict:
        return dict(self._call("kv_protection_stats"))


class _NullKVControl:
    def protect(self, request_id, deadline_ts): return 0
    def release(self, request_id): return 0
    def evict(self, request_id): return 0
    def swap(self, request_id): return 0
    def stats(self): return {}


class _FanoutKVControl:
    """Routes each retention call to the instance that served the turn
    (recorded at protect time by kv_tag)."""

    def __init__(self, controls: list) -> None:
        self._controls = controls
        self._where: dict[str, int] = {}
        self.current_instance = 0  # set by the driver before each call

    def protect(self, request_id, deadline_ts):
        inst = self.current_instance
        n = self._controls[inst].protect(request_id, deadline_ts)
        if n > 0:
            self._where[request_id] = inst
        return n

    def release(self, request_id):
        inst = self._where.pop(request_id, None)
        if inst is None:
            return 0
        return self._controls[inst].release(request_id)

    def evict(self, request_id):
        inst = self._where.pop(request_id, self.current_instance)
        return self._controls[inst].evict(request_id)

    def cancel_provisional(self, request_id):
        inst = self._where.pop(request_id, self.current_instance)
        return self._controls[inst].cancel_provisional(request_id)

    def stats(self):
        return {i: c.stats() for i, c in enumerate(self._controls)}


@dataclass
class TupleConfig:
    retention: str = "cache-lru"
    scheduling: str = "fcfs"
    routing: Optional[str] = None
    tau_s: Optional[float] = None
    default_gap_s: float = 1.0
    min_waste_profile: Optional[str] = None
    capacity_limit: Optional[int] = None
    harness_root: Optional[str] = None
    log_dir: Optional[str] = None
    engine_observations: bool = False
    engine_policy: Optional[str] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class PolicyDriver:
    """Owns the executors and serializes their calls on one worker."""

    def __init__(self, cfg: TupleConfig, engines: list,
                 loop: asyncio.AbstractEventLoop) -> None:
        self.cfg = cfg
        self.engines = engines
        self._engine_inflight = [0] * len(engines)
        self._program_home = {}
        self.saga = None  # SagaGateway, attached by the runner for name: saga
        # KV block size (tokens) for the admission gate's QueueView.
        self._block_size = None
        try:
            self._block_size = int(engines[0].vllm_config.cache_config.block_size)
        except Exception:
            pass
        (r, s, rt, wm, pm) = import_harness(cfg.harness_root)
        self.programs = pm.ProgramTable()
        self._worker = ThreadPoolExecutor(max_workers=1)
        self._loop = loop
        offset = time.time() - loop.time()
        self._log_files = []

        def _log(name):
            if cfg.log_dir is None:
                return None
            os.makedirs(cfg.log_dir, exist_ok=True)
            f = open(os.path.join(cfg.log_dir, name), "w")
            self._log_files.append(f)
            return f

        # retention
        if cfg.retention in ("cache-lru", "evict-always"):
            policy = (r.CacheLRURetention() if cfg.retention == "cache-lru"
                      else r.EvictAlwaysRetention())
            kv = _NullKVControl()
        else:
            controls = [AsyncEngineKVControl(e, loop, offset) for e in engines]
            kv = _FanoutKVControl(controls)
            if cfg.retention == "ttl":
                policy = r.TTLRetention(_require_tau(cfg.tau_s, "ttl"))
            elif cfg.retention == "continuum":
                policy = r.ContinuumTTLRetention(
                    pin_s=2.0 if cfg.tau_s is None else cfg.tau_s)
            elif cfg.retention == "saga-ttl":
                policy = r.PressureTTLRetention(_require_tau(cfg.tau_s, "saga-ttl"))
            elif cfg.retention == "evolved":
                # evolved_retention.py dropped next to harness/retention.py in a
                # scratch copy of the harness (--harness-root), as the sim does.
                policy = _evolved_class(r, "retention")()
            elif cfg.retention == "min-waste":
                if not cfg.min_waste_profile:
                    raise ValueError("min-waste needs --min-waste-profile")
                profile = wm.WasteProfile.from_json(cfg.min_waste_profile)
                policy = r.MinWasteRetention(
                    profile, default_gap_s=cfg.default_gap_s)
            else:
                policy = _policies.resolve(
                    "kv", cfg.retention,
                    _PolicyConfig(tau_s=cfg.tau_s,
                                  default_gap_s=cfg.default_gap_s,
                                  min_waste_profile=cfg.min_waste_profile,
                                  capacity_limit=cfg.capacity_limit),
                    flag="--retention")
        self.kv = kv
        self.retention_exec = r.RetentionExecutor(
            policy=policy, kv=kv, log_file=_log("retention.jsonl"),
            programs=self.programs)
        if cfg.engine_observations and cfg.retention == "min-waste":
            policy.predictor = self.programs.tool_mean_gap_s
            policy.load_probe = self._observed_load

        # scheduling
        if cfg.scheduling == "fcfs":
            spolicy = s.FCFSScheduling()
        elif cfg.scheduling == "program-fcfs":
            spolicy = s.ProgramFCFSScheduling()
        elif cfg.scheduling == "plas":
            spolicy = s.PLASScheduling()
        elif cfg.scheduling == "continuum":
            spolicy = s.ContinuumScheduling()
        elif cfg.scheduling == "evolved":
            spolicy = _evolved_class(s, "scheduling")()
        else:
            spolicy = _policies.resolve(
                "scheduling", cfg.scheduling,
                _PolicyConfig(tau_s=cfg.tau_s,
                              default_gap_s=cfg.default_gap_s,
                              capacity_limit=cfg.capacity_limit),
                flag="--scheduling")
        self.scheduling_exec = s.SchedulingExecutor(
            policy=spolicy, log_file=_log("scheduling.jsonl"),
            programs=self.programs)

        # routing
        n = len(engines)
        if cfg.routing is None:
            self.routing_exec = None
        else:
            if cfg.routing == "rr":
                rpolicy = rt.RoundRobinRouting(n)
            elif cfg.routing == "least-loaded":
                rpolicy = rt.LeastLoadedRouting(n)
            elif cfg.routing == "session-affinity":
                rpolicy = rt.SessionAffinityRouting(
                    n, capacity_limit=cfg.capacity_limit)
            else:
                rpolicy = _policies.resolve(
                    "routing", cfg.routing,
                    _PolicyConfig(num_instances=n, capacity_limit=cfg.capacity_limit),
                    flag="--routing")
            self.routing_exec = rt.RoutingExecutor(
                policy=rpolicy, log_file=_log("routing.jsonl"),
                programs=self.programs)
        self._rr = 0
        self._async_callbacks = None
        if os.environ.get('BENCH_ASYNC_POLICY_RPC') == '1':
            from .async_policy_callbacks import AsyncPolicyCallbacks
            self._async_callbacks = AsyncPolicyCallbacks(self)

    # -- executor calls (serialized on the worker thread) --------------
    async def _run(self, fn, *args, trace_program_id=None):
        path = os.environ.get('BENCH_CALLBACK_TRACE')
        if path:
            # Async acknowledgement receives a decision object, not a program
            # ID. Keep diagnostics scalar and let that caller supply identity.
            program_id = trace_program_id
            if program_id is None and args and isinstance(args[0], (str, int)):
                program_id = args[0]
            queued = self._loop.time()
            def measured():
                started = self._loop.time()
                try:
                    return fn(*args)
                finally:
                    finished = self._loop.time()
                    with open(path, 'a') as stream:
                        stream.write(json.dumps(dict(
                            callback=fn.__name__, program_id=program_id,
                            queued_ts=queued, started_ts=started, finished_ts=finished,
                            worker_wait_s=started - queued,
                            execution_including_rpc_s=finished - started)) + '\n')
            return await self._loop.run_in_executor(self._worker, measured)
        return await self._loop.run_in_executor(self._worker, fn, *args)

    def _engine_observation(self, instance):
        # This method runs on the driver worker, just like protect/release.
        if self.cfg.engine_policy == 'saga':
            observation = self.saga.latest_stats[instance].get("policy_observation")
        else:
            control = self.kv._controls[instance]
            observation = control.stats().get("policy_observation")
        if observation is None or observation.get("schema_version") != 1:
            raise RuntimeError("Policy engine observations requested but unavailable")
        return observation

    def _observed_load(self):
        observation = self._engine_observation(self.kv.current_instance)
        return (int(observation["scheduled_query_tokens"]),
                int(observation["running_context_tokens"]))

    def _route_sync(self, program_id, turn_idx, now, prompt_tokens=None):
        if self.cfg.engine_policy == 'saga':
            # Observe the completed gap before on_turn_release clears it.
            self.retention_exec.observe_arrival(self.programs.get(program_id), now)
            instance = self.saga.route(program_id, now)
            self.programs.on_turn_release(program_id, turn_idx, now, instance=instance)
            return instance
        if self.cfg.engine_policy == 'autellix':
            if prompt_tokens is None or prompt_tokens <= 0:
                raise ValueError('Autellix routing requires the current prompt length')
            least = min(range(len(self.engines)),
                        key=lambda i: (self._engine_inflight[i], i))
            if prompt_tokens <= 2048:
                instance = least
            else:
                instance = self._program_home.setdefault(program_id, least)
            self._engine_inflight[instance] += 1
            self.programs.on_turn_release(program_id, turn_idx, now, instance=instance)
            return instance
        if self.routing_exec is not None:
            # RoutingExecutor applies on_turn_release, which clears the gap.
            # Observe first; the later arrival callback sees an inactive gap.
            self.retention_exec.observe_arrival(
                self.programs.get(program_id), now)
            return self.routing_exec.route(program_id, turn_idx, now)
        if len(self.engines) == 1:
            return 0
        inst = self._rr % len(self.engines)  # stock: plain round-robin
        self._rr += 1
        return inst

    def _pool_view(self, instance):
        """(util, free_blocks, block_size) for one instance, from the engine's
        protection stats. ONE definition, shared by the admission gate
        (_queue_view) and the retention signals (_kv_utilization), so the two
        policy consumers can never drift apart again.

        util = 1 - free_queue_blocks / (num_gpu_blocks - 1). The free queue
        holds cached-evictable blocks (upstream vLLM: evictable is FREE) and
        excludes parked blocks (parking removes them from the queue: parked is
        USED). This deliberately does NOT read the engine's kv_cache_usage,
        whose get_num_free_blocks carries our fork's `+ len(_protected)` and
        would make a mostly-pinned pool look empty to a retention policy.
        GATE_UTIL_SEMANTICS=sim reproduces the simulator's old accounting
        (evictable counted as used); default "vllm"."""
        st = (self.saga.latest_stats if self.cfg.engine_policy == 'saga'
              else self.kv._controls[instance].stats()
              if isinstance(self.kv, _FanoutKVControl) else self.kv.stats())
        if isinstance(st, dict) and instance in st:
            st = st[instance]
        st = st or {}
        return self._pool_view_from_stats(st)

    def _pool_view_from_stats(self, st):
        bs = int(self._block_size or 16)
        free_blocks = int(st.get("free_queue_blocks", 0) or 0)
        total = int(st.get("num_gpu_blocks", 0) or 0)
        util_used = free_blocks
        if os.environ.get("GATE_UTIL_SEMANTICS", "vllm") == "sim":
            util_used = free_blocks - int(st.get("cached_free_blocks", 0) or 0)
        util = None
        if total > 1:
            util = max(0.0, min(1.0, 1.0 - util_used / (total - 1)))
        return util, free_blocks, bs

    def _kv_utilization(self, instance):
        """Utilization fed to retention policies (SystemSignals.kv_utilization,
        turn_arrival / turn_complete). Same number the gate sees; None before
        the engine reports stats."""
        # Continuum uses completed tool durations, a fixed TTL and queue hold;
        # neither policy callback consumes utilization. Avoid a synchronous
        # engine RPC on the shared policy worker for an unused observation.
        if self.cfg.retention == 'continuum':
            return None
        try:
            return self._pool_view(instance)[0]
        except Exception:  # pragma: no cover
            return None

    # ---- admission gate (wider scheduling hooks; mirrors the simulator's
    # UnifiedPolicyAdapter.filter_waiting). Runs only when the scheduling
    # policy overrides admit(). Tick = ADMIT_TICK_S of wall time.
    ADMIT_TICK_S = float(os.environ.get("BENCH_ADMIT_TICK_S", "0.02"))

    @property
    def has_admit(self) -> bool:
        return self.scheduling_exec.has_admit

    def _queue_view(self, instance, prompt_tokens, n_inflight, pool_stats=None):
        """QueueView from the engine's protection stats. Real vLLM counts
        parked blocks as free (the valve can break them), so 'free' here is
        the free queue only: what an allocation takes WITHOUT breaking a
        pin. The free queue already holds the cached-evictable blocks, so
        evictable is folded into free and reported as 0. cached_tokens is
        unknown before submission (no prefix probe on the gateway): 0,
        which makes the gate slightly more conservative than in the sim."""
        from policies.base import QueueView
        util, free_blocks, bs = (self._pool_view(instance) if pool_stats is None
                                else self._pool_view_from_stats(pool_stats))
        return QueueView(
            n_running=n_inflight, n_waiting=0, n_inflight=n_inflight,
            kv_utilization=util, kv_free_tokens=free_blocks * bs,
            kv_evictable_tokens=0, prompt_tokens=int(prompt_tokens),
            cached_tokens=0)

    def _admit_sync(self, program_id, turn_idx, prompt_tokens, n_inflight, now, pool_stats=None):
        inst = self.programs.get(program_id).kv_instance or 0
        view = self._queue_view(inst, prompt_tokens, n_inflight, pool_stats)
        return self.scheduling_exec.admit(program_id, turn_idx, now, view)

    async def admit(self, program_id: str, turn_idx: int, prompt_tokens: int,
                    n_inflight: int, now: float) -> bool:
        if self._async_callbacks is not None:
            return await self._async_callbacks.admit(program_id, turn_idx, prompt_tokens, n_inflight, now)
        return await self._run(self._admit_sync, program_id, turn_idx,
                               prompt_tokens, n_inflight, now)

    def _arrival_sync(self, program_id, turn_idx, now, pool_stats=None):
        pcb = self.programs.get(program_id)
        if (self.cfg.engine_observations and pcb.kv_request_id is not None
                and isinstance(self.kv, _FanoutKVControl)):
            instance = self.kv._where.get(pcb.kv_request_id)
            protected = False
            if instance is not None:
                observation = self._engine_observation(instance)
                protected = observation["protected_blocks_by_tag"].get(
                    pcb.kv_request_id, 0) > 0
            # Keep the release handle even if the priority flag is cleared:
            # expired blocks may still need their normal successor release.
            self.programs.on_memory_pressure(program_id, kv_protected=protected)
        # Gap observation must precede the stamp: the dispatch transition
        # (on_turn_release inside stamp) clears gap_started_ts, the only
        # record of the just-ended gap's duration. Same order as the
        # simulator's UnifiedPolicyAdapter.on_turn_routed. Without this a
        # gap-learning policy (Continuum) never builds history and pins
        # every turn as a cold start (measured 2026-09-08: real pinned
        # 2,873/2,873 turns vs 2,662 in the simulator).
        self.retention_exec.observe_arrival(
            self.programs.get(program_id), now)
        # Stamp before release: a pinned-first scheduler needs the pin.
        priority = self.scheduling_exec.stamp(program_id, turn_idx, now)
        inst = self.programs.get(program_id).kv_instance or 0
        self.retention_exec.turn_arrival(program_id, now=now, kv_utilization=(
            self._pool_view_from_stats(pool_stats)[0] if pool_stats is not None
            else self._kv_utilization(inst)))
        return priority

    def _complete_sync(self, program_id, turn_idx, request_id, service_s,
                       context_tokens, instance, now, tool_name=None,
                       kv_snapshot=None):
        if self.cfg.engine_policy == 'autellix':
            self._engine_inflight[instance] -= 1
        if self.routing_exec is not None:
            self.routing_exec.turn_complete(instance)
        self.scheduling_exec.turn_complete(
            program_id, service_s, turn_idx=turn_idx)
        if hasattr(self.kv, "current_instance"):
            self.kv.current_instance = instance
        # tool_name must be real: ProgramTable.on_turn_complete only starts the
        # gap clock (gap_started_ts) when it is not None, and a policy that
        # learns tool times (Continuum) reads that clock at the next arrival.
        # Passing None here left the history empty, so every turn looked like a
        # cold start and was pinned (measured 2026-09-08).
        decision = self.retention_exec.turn_complete(
            program_id, turn_idx, request_id, tool_name, now=now,
            context_tokens=context_tokens,
            kv_utilization=(self._completion_utilization(kv_snapshot)
                            if kv_snapshot is not None else self._kv_utilization(instance)))
        if (decision.action == "none"
                and isinstance(self.kv, _FanoutKVControl)
                and os.environ.get("VLLM_KV_PIN_AT_FREE_TTL")):
            self.kv.cancel_provisional(request_id)

    async def turn_ready(self, program_id: str, turn_idx: int, now: float,
                         prompt_tokens: Optional[int] = None,
                         completed_tool_duration_s: Optional[float] = None
                         ) -> tuple[int, Optional[int]]:
        """Route then release+stamp. Returns (instance, priority)."""
        if self._async_callbacks is not None:
            return await self._async_callbacks.turn_ready(
                program_id, turn_idx, now, prompt_tokens, completed_tool_duration_s)
        if self.cfg.engine_policy == 'saga':
            await self.saga.refresh_for_route(now)
        if completed_tool_duration_s is not None:
            await self._run(self.programs.observe_completed_tool,
                            program_id, completed_tool_duration_s)
        instance = await self._run(self._route_sync, program_id, turn_idx, now, prompt_tokens)
        priority = await self._run(self._arrival_sync, program_id, turn_idx, now)
        return instance, priority

    async def turn_complete(self, program_id: str, turn_idx: int,
                            request_id: str, service_s: float,
                            context_tokens: int, instance: int, now: float,
                            tool_name: Optional[str] = None,
                            kv_snapshot=None) -> None:
        if self._async_callbacks is not None:
            return await self._async_callbacks.turn_complete(
                program_id, turn_idx, request_id, service_s, context_tokens,
                instance, now, tool_name, kv_snapshot)
        await self._run(self._complete_sync, program_id, turn_idx, request_id,
                        service_s, context_tokens, instance, now, tool_name, kv_snapshot)

    @staticmethod
    def _completion_utilization(snapshot):
        if (snapshot['schema_version'] != 1
                or snapshot['event'] != 'completion_after_free'):
            raise ValueError('Unsupported completion KV snapshot')
        capacity = snapshot['num_gpu_blocks'] - 1
        free = snapshot['free_queue_blocks']
        if capacity <= 0 or not 0 <= free <= capacity:
            raise ValueError('Invalid completion KV capacity')
        return 1.0 - free / capacity

    async def finish(self) -> dict:
        await self._run(self.retention_exec.finish)
        stats = await self._run(self.kv.stats)
        for f in self._log_files:
            f.close()
        self._worker.shutdown(wait=True)
        return {
            "tuple": self.cfg.as_dict(),
            "kv_protection_stats": stats,
            "retention_decisions": len(self.retention_exec.decisions),
            "priority_stamps": len(self.scheduling_exec.stamps),
            "admission_holds": self.scheduling_exec.admission_holds,
            "admission_admits": self.scheduling_exec.admission_admits,
            "routing_decisions": (len(self.routing_exec.decisions)
                                  if self.routing_exec else 0),
        }
