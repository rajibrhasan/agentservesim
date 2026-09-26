"""Simulator-side mirror of the unified serving policy (retention,
scheduling, routing).

The mirror imports the SAME policy classes the real harness runs
(agentservesim/harness: retention.py, scheduling.py, routing.py,
waste_model.py) and drives them from simulator events, so the decision
logic is shared by construction. Decisions are applied through the
simulator's native mechanisms:

- retention: a completed turn's cached prefix is parked by holding a
  radix lock (inc_lock_ref on the request's last node) until release,
  expiry, or the safety valve. The valve mirrors the engine mechanism
  (vllm agent-knobs branch): when evict_prefix_cache cannot free
  enough, protections are broken expired-first, then
  latest-deadline-first, with counters.
- scheduling: the harness stamp becomes Request.priority; the
  scheduler's waiting queue orders by (priority, arrival, id) when the
  policy is priority-based (mirror of vLLM's PriorityRequestQueue).
- routing: the Router delegates instance selection per request to the
  shared RoutingExecutor.

Each knob appends the same JSONL decision log the real harness writes;
harness/parity.py compares the two logs (Phase B decision parity).

Events (wired in router.py / __main__.py):
- turn routed  -> routing decision, priority stamp, retention
  turn_arrival (releases the previous turn's protection)
- turn complete -> retention turn_complete (protect/evict/none),
  PLAS service accounting, routing in-flight decrement

Time: simulator ns are converted to float seconds for the shared
policy code; deadlines are stored back in ns.
"""

import os
import json
import sys

# Default "vllm": match vLLM's kv_cache_usage, where cached-evictable blocks sit
# in the free queue and count as FREE (upstream v0.19.0 behaviour, verified by
# `git diff v0.19.0 -- vllm/v1/core/block_pool.py`). Parked/pinned KV stays USED
# on purpose: both the gate and the retention policies need a mostly-pinned pool
# to read as full. "sim" (the previous default) counted evictable as used, which
# the gate's own check already adds back explicitly -- double counting.
_KV_UTIL_SEMANTICS = os.environ.get("SIM_KV_UTIL_SEMANTICS", "vllm")
# SIM_GATE_PREFIX_PROBE=1 (NOT the default; see below): the gate sees the live
# prefix-cache hit of each waiting turn (vLLM looks computed blocks up at
# schedule time). 0 reproduces the pre-2026-09-06 behaviour, where a turn
# never scheduled reported cached_tokens=0 to the gate (prefix_match only
# ran after admission), so the gate tested the full prompt against
# free+evictable.
# The setting must match WHERE the gate runs, and there is no single right
# default. A gateway-site gate cannot know the prefix hit before submission
# (bench/core/policy_driver.py::_queue_view passes cached_tokens=0), so 0 mirrors
# it; an engine-site gate (GATE_SITE=engine GATE_CACHED_TOKENS=live) does know it
# and needs 1. Measured 2026-09-12 on B200·SWE50: live probe 875.23 vs cached=0
# 887.24 (real 729.34) -- ~1% either way there.
#
# CORRECTION 2026-09-15, default flipped to 1. The old comment justified 0 with
# "every board leg ran at gate_site=gateway", which is false -- the board gate
# leg (rtx6000_70b_swebench_gate__jps0.02_engine_pinrel/gate) ran ENGINE-side:
# 235,745 admission holds sit in the engine's kv_protection_stats and 0 in the
# gateway driver's, and it wrote decisions/engine_admission.jsonl.
#
# More to the point, THIS gate is engine-side by construction. filter_waiting is
# called from Scheduler.schedule (scheduler.py ~405), the same position as the
# engine's own gate (vllm/v1/core/sched/admission_gate.py), which reads the hit
# live via kvm.get_computed_blocks unless GATE_CACHED_TOKENS=zero. With the
# probe on, every QueueView field matches that gate: n_running, n_waiting,
# kv_free_tokens (free_uncached), kv_evictable_tokens (cached_free) and
# cached_tokens. Defaulting to 0 made the simulator mirror the GATEWAY driver's
# view (bench/core/policy_driver.py::_queue_view, which cannot probe before
# submission and passes 0) -- a different deployment from the one in the loop.
# Set 0 deliberately to model a gateway-site gate.
_GATE_PREFIX_PROBE = os.environ.get("SIM_GATE_PREFIX_PROBE", "1") != "0"


def _default_harness_root():
    """Where `harness/` lives when nobody passes --harness-root.

    The repo root. It used to be a sibling `agentservesim/` checkout, and that
    stopped being true when the trees were split: agentservesim is documents
    now and carries no code. Every real caller passes the flag explicitly
    (runtime/invoke.py: `--harness-root /app/LLMServingSim`), so the stale
    default only ever bit a bare invocation -- which is exactly the case a
    default is for.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


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


def _spec_or_raise(axis, value, flag, _cache=None, **cfg_kw):
    """A `module:Class` policy on an axis flag, or a clear refusal.

    ONE instance per spec. When the same `module:Class` is named on two axes,
    both get the same object -- so a policy that decides retention AND
    scheduling from one piece of state can run HERE, on the plane the
    published numbers were produced on, instead of being pushed to
    `--planes program` for the sake of sharing state. Two instances of one
    class coordinating through module globals is what this avoids, and it is
    what the old joint seed had to do.
    """
    from .program_policy_adapter import load_unified
    from policies.base import PolicyConfig
    if ":" not in (value or ""):
        raise ValueError(
            f"unknown {flag} value: {value!r} (and not module:Class)")
    if _cache is not None and value in _cache:
        return _cache[value]
    obj = load_unified(value, flag=f"--{flag}",
                       instantiate=False).from_config(PolicyConfig(**cfg_kw))
    if _cache is not None:
        _cache[value] = obj
    return obj


def load_custom_policy(mod, modname, clsname):
    """The class behind `--retention evolved` / `--scheduling evolved`.

    Only the policy search uses this: it stages a generated candidate into a
    scratch directory and runs it by that fixed name. A policy someone WRITES
    is named on the flag as `module:Class` and needs none of it.
    """
    cls = getattr(mod, clsname, None)
    if cls is None:
        import importlib
        cls = getattr(importlib.import_module(modname), clsname)
    return cls


def import_harness(harness_root=None):
    """Put agentservesim on sys.path and import the shared policy
    modules. Raises a clear error when the checkout is missing."""
    root = harness_root or _default_harness_root()
    # `root` is a place to import FROM, not a checkout to validate. It may hold
    # a whole `policies/` package, or a single module naming one class -- which
    # is what a policy someone writes looks like, and what the search stages.
    # Requiring a `policies/` directory here rejected the latter outright.
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        import policies
    except ImportError as e:
        raise FileNotFoundError(
            f"cannot import `policies` with {root!r} on the path. Run from the "
            f"AgentServingSim checkout, or pass --policy-root pointing at a "
            f"directory that has it.") from e
    from policies import base, program                   # noqa: F401
    from policies.utils import waste_model              # noqa: F401
    # `base` stands in for what used to be three modules: every class the
    # callers look up by attribute now lives on it or on the registry, so the
    # five-tuple keeps its shape while the classes moved.
    return policies, policies, policies, waste_model, program


# Deadline for protections held through the queue wait (release_event ==
# "scheduled"): far future so the valve's expired-first pass skips them
# and the unexpired latest-deadline-first pass breaks them first.
_HOLD_SENTINEL_NS = 1 << 62


class _ProtEntry:
    __slots__ = ("memory", "node", "deadline_ns", "tokens")

    def __init__(self, memory, node, deadline_ns, tokens):
        self.memory = memory
        self.node = node
        self.deadline_ns = deadline_ns
        self.tokens = tokens


class UnifiedPolicyAdapter:
    """Drives the shared harness policies from simulator events and is
    itself the KVControl transport for the retention executor."""

    def __init__(self, retention_value, scheduling_value, routing_value,
                 num_instances, block_size,
                 tau_s=None, default_gap_s=1.0, min_waste_profile=None,
                 capacity_limit=None, log_dir=None, harness_root=None,
                 oracle_table_path=None, min_waste_swap=False,
                 autellix_queues=None, autellix_overprovision=0,
                 saga_eviction_order=False, autellix_swap=False,
                 long_prompt_tokens=2048, min_waste_fcfs_restore=False,
                 saga_fairness=False, saga_fairness_slack=1.0,
                 saga_prefetch=False, saga_prefetch_margin_s=0.5,
                 saga_stealing=True):
        self.saga_stealing = bool(saga_stealing)
        (self._retention_mod, self._scheduling_mod,
         self._routing_mod, self._waste_mod,
         self._program_mod) = import_harness(harness_root)
        #: spec -> instance, so one object can decide several axes here.
        self._spec_cache = {}

        self.block_size = block_size
        self.num_instances = num_instances
        self._now_ns = 0
        # kv_tag -> _ProtEntry for currently parked protections
        self._parked = {}
        # program_id -> instance index of the last routed turn (retention
        # protections live on the instance that served the turn)
        self._ctx = None  # (memory, node, tokens) during turn_complete
        self._ctx_req = None  # the finished Request during turn_complete
        self.min_waste_swap = min_waste_swap
        self._host_swaps = {}
        self._swap_programs = {}
        self._infercept_arrivals = {}
        self._infercept_history = {}
        self._last_memory = None  # last instance memory seen (kv_utilization signal)
        self.stats = {
            "protected": 0, "released": 0, "unparked_by_hit": 0,
            "reclaimed_expired": 0, "reclaimed_forced": 0,
            "swapped_out": 0, "swapped_tokens": 0}

        self._oracle_table_path = oracle_table_path
        self._oracle_tbl = None

        self._log_files = []
        # SIM_GATE_TRACE_DIR: per-hold view (same fields as vLLM's
        # engine_admission.jsonl) and per-filter_waiting pool occupancy,
        # for sim-vs-real memory-model comparison. Absolute dir (must be
        # visible inside the container, e.g. under /orange).
        self._gate_trace = None
        tdir = os.environ.get("SIM_GATE_TRACE_DIR")
        if tdir:
            os.makedirs(tdir, exist_ok=True)
            self._gate_trace = (
                open(os.path.join(tdir, "gate_holds.jsonl"), "a", buffering=1),
                open(os.path.join(tdir, "gate_steps.jsonl"), "a", buffering=1))
            self._log_files.extend(self._gate_trace)
        def _log(name):
            if log_dir is None:
                return None
            os.makedirs(log_dir, exist_ok=True)
            f = open(os.path.join(log_dir, name), "w")
            self._log_files.append(f)
            return f

        # One Program Control Block table, shared by all three executors:
        # the routing pin, the scheduling stamp, and the retention
        # valuation must read the SAME record, or the three knobs are
        # deciding from three different views of the program.
        self.programs = self._program_mod.ProgramTable()

        # -------- retention --------
        self.retention_value = retention_value
        r = self._retention_mod
        if retention_value in (None, "cache-lru"):
            policy = r.CacheLRURetention()
        elif retention_value == "evict-always":
            policy = r.EvictAlwaysRetention()
        elif retention_value == "ttl":
            policy = r.TTLRetention(tau_s=_require_tau(tau_s, "ttl"))
        elif retention_value == "min-waste":
            profile = self._waste_mod.WasteProfile.from_json(min_waste_profile)
            policy = r.MinWasteRetention(profile, default_gap_s=default_gap_s)
            if min_waste_swap:
                policy.defer_swap = True
        elif retention_value == "saga-ttl":
            policy = r.PressureTTLRetention(_require_tau(tau_s, "saga-ttl"))
        elif retention_value == "saga-tool-ttl":
            from policies.saga import SagaToolTTL
            policy = SagaToolTTL(cold_start_s=tau_s)
        elif retention_value == "continuum":
            # Unset -> the released FIXED_THRESHOLD_CONTINUUM of 2 s.
            policy = r.ContinuumTTLRetention(
                pin_s=2.0 if tau_s is None else tau_s)
        elif retention_value == "evolved":
            # Search candidate: the harness at --harness-root carries an
            # EvolvedRetention class (agentservesim/evolve). It sees the
            # same PCB the published policies see and nothing else.
            policy = self._evolved_class(r, "harness.evolved_retention",
                                         "EvolvedRetention")()
        elif retention_value == "search-seed":
            from policies.search_seed import EvolvedRetention
            policy = EvolvedRetention()
        elif retention_value == "gate":
            # The evolved champion as a named policy (policies/gate.py, a
            # copy of evolve/champion_joint_41008503.py).
            from policies.gate import EvolvedRetention as GateRetention
            policy = GateRetention()
        elif retention_value == "oracle-ttl":
            # Clairvoyant probe (sim-only): perfect per-turn tau from the
            # trace; bounds the gap-predicting TTL family.
            from policies.oracle import OracleTTLRetention
            policy = OracleTTLRetention(self._oracle_table())
        else:
            policy = _spec_or_raise("kv", retention_value, "retention",
                                    self._spec_cache, tau_s=tau_s,
                                    default_gap_s=default_gap_s,
                                    min_waste_profile=min_waste_profile)
        self.retention_exec = r.RetentionExecutor(
            policy=policy, kv=self, log_file=_log("retention.jsonl"),
            programs=self.programs)
        # Built here rather than later: it needs the retention policy object,
        # which is the only thing that carries the learned gap estimate.
        self._prefetcher = None
        if saga_prefetch:
            from .saga_prefetch import SagaPrefetcher
            self._prefetcher = SagaPrefetcher(
                getattr(policy, "predicted_gap_s", None),
                margin_s=float(saga_prefetch_margin_s))
        # Queue-persistent policies (release_event == "scheduled", e.g.
        # Continuum's released code) release at batch admission, not at
        # arrival; programs whose release is deferred wait in this set.
        self._release_at_scheduled = (
            getattr(policy, "release_event", "arrival") == "scheduled")
        self._pending_release = set()

        # -------- scheduling --------
        self.scheduling_value = scheduling_value
        s = self._scheduling_mod
        if scheduling_value in (None, "fcfs"):
            spolicy = s.FCFSScheduling()
        elif scheduling_value == "search-seed":
            from policies.search_seed import EvolvedScheduling
            spolicy = EvolvedScheduling()
        elif scheduling_value == "program-fcfs":
            spolicy = s.ProgramFCFSScheduling()
        elif scheduling_value == "plas":
            spolicy = s.PLASScheduling()
        elif scheduling_value == "continuum":
            spolicy = s.ContinuumScheduling()
        elif scheduling_value == "evolved":
            # Search candidate: the harness at --harness-root carries an
            # EvolvedScheduling class (agentservesim/evolve). It stamps a
            # priority from the same PCB the published values see. Joint
            # candidates define both classes in one module, imported by
            # shims; the attribute can then be missing from the module
            # object while the shim imports cleanly on a second attempt.
            spolicy = self._evolved_class(s, "harness.evolved_scheduling",
                                          "EvolvedScheduling")()
        elif scheduling_value == "autellix-mlfq":
            # Autellix Algorithm 1 itself, driven per instance by
            # serving/core/autellix_driver.py. The policy object stamps
            # nothing: ordering is the planner's output, applied in
            # filter_waiting, so the waiting queue is not re-sorted
            # behind it.
            from .autellix_driver import queue_config
            if autellix_queues is None:
                queue_config(None, None, None)   # raises with the flag list
            self._autellix_cfg = queue_config(*autellix_queues)
            self._autellix_overprovision = int(autellix_overprovision)
            spolicy = s.SchedulingPolicy()
        elif scheduling_value == "gate":
            from policies.gate import EvolvedScheduling as GateScheduling
            spolicy = GateScheduling()
        elif scheduling_value == "oracle-srpt":
            # Clairvoyant probe (sim-only): true remaining program work;
            # bounds the size-based program-level scheduling family.
            from policies.oracle import OracleSRPTScheduling
            spolicy = OracleSRPTScheduling(self._oracle_table())
        else:
            spolicy = _spec_or_raise("scheduling", scheduling_value,
                                     "scheduling", self._spec_cache)
        self.scheduling_exec = s.SchedulingExecutor(
            policy=spolicy, log_file=_log("scheduling.jsonl"),
            programs=self.programs)
        # Does this policy actually stamp? Ask the CLASS, not a list of names.
        #
        # It was `scheduling_value in ("program-fcfs", "plas", ...)`, so a
        # policy named as `module:Class` -- which is every policy someone
        # writes, and every search candidate -- was not in the list, the engine
        # stayed on FCFS, and `priority()` was called and its answer discarded.
        # The policy ran, produced a plausible JCT, and decided nothing.
        _sbase = s.SchedulingPolicy
        self.priority_scheduling = (
            getattr(type(spolicy), "priority", None)
            is not getattr(_sbase, "priority", None)
            # AFS stamps the queue itself, so the scheduler must order by it.
            or bool(saga_fairness))
        # Wider scheduling hooks (victim rule, admission gate): active only
        # when the policy overrides them, so every published value runs
        # the scheduler's unchanged code path.
        base = s.SchedulingPolicy
        self.custom_hooks = (
            type(spolicy).victim is not base.victim
            or type(spolicy).admit is not base.admit
            or scheduling_value == "autellix-mlfq"
            or (retention_value == 'min-waste' and min_waste_swap
                and scheduling_value == 'fcfs')
            or min_waste_fcfs_restore)
        # instance id -> AutellixDriver, when the MLFQ planner is selected.
        self._autellix = {}
        # SAGA's workflow-aware reclaim ranking, opt-in so the recorded
        # saga-ttl / saga-tool-ttl rows stay reproducible.
        self.saga_eviction_order = bool(saga_eviction_order)
        # Autellix preempts by swapping the victim's KV to host memory; the
        # simulator's own preemption is by recomputation, so this replaces the
        # recompute with a copy the successor restores from.
        self.autellix_swap = bool(autellix_swap)
        # InferCept restores in arrival order inside the hiding window; without
        # it the simulator restores whatever the scheduler admitted.
        self.min_waste_fcfs_restore = bool(min_waste_fcfs_restore)
        self.infercept_paper_token_budget = None
        if (os.environ.get('INFERCEPT_PAPER_SCHEDULING') == '1'
                and retention_value == 'min-waste' and min_waste_swap
                and scheduling_value == 'fcfs'):
            self.infercept_paper_token_budget = int(policy.profile.S)
            if self.infercept_paper_token_budget <= 0:
                raise ValueError('InferCept saturation point must be positive')
        self._restore_gates = {}
        # SAGA's adaptive fair share. Off by default because it needs tenant
        # and deadline fields the standard traces do not carry.
        self._fairness = None
        if saga_fairness:
            from .saga_fairness import SagaFairness
            self._fairness = SagaFairness(saga_fairness_slack)
        self.stats["saga_fair_stamps"] = 0
        self.stats["autellix_swapped"] = 0
        self.stats["autellix_swap_declined"] = 0
        self._saga_evict = {}
        self.stats["saga_ranked"] = 0
        self.stats["saga_lru_fallback"] = 0
        self._autellix_cfg = getattr(self, "_autellix_cfg", None)
        self._autellix_overprovision = getattr(self, "_autellix_overprovision", 0)
        self.stats["autellix_demoted"] = 0
        self.stats["autellix_promoted"] = 0
        self.stats["autellix_overprovisioned"] = 0
        self.stats["victim_overrides"] = 0
        self.stats["admission_holds"] = 0
        self.stats["admission_releases"] = 0

        # -------- routing --------
        self.routing_value = routing_value
        rt = self._routing_mod
        if routing_value is None:
            self.routing_exec = None
        else:
            if routing_value in ("saga-placement", "autellix-route"):
                # Both need live instance state, so they are driven from
                # serving/core/routing_drivers.py rather than through the
                # shared RoutingExecutor's program-record-only interface.
                from .routing_drivers import AutellixRouter, SagaRouter
                self._live_router = (
                    SagaRouter(num_instances) if routing_value == "saga-placement"
                    else AutellixRouter(num_instances, long_prompt_tokens))
                rpolicy = None
            elif routing_value == "rr":
                rpolicy = rt.RoundRobinRouting(num_instances)
            elif routing_value == "least-loaded":
                rpolicy = rt.LeastLoadedRouting(num_instances)
            elif routing_value == "session-affinity":
                rpolicy = rt.SessionAffinityRouting(
                    num_instances, capacity_limit=capacity_limit)
            else:
                rpolicy = _spec_or_raise("routing", routing_value, "routing",
                                         self._spec_cache,
                                         num_instances=num_instances)
            self.routing_exec = None if rpolicy is None else rt.RoutingExecutor(
                policy=rpolicy, log_file=_log("routing.jsonl"),
                programs=self.programs)
        # request_id -> instance index (for the in-flight decrement)
        self._req_instance = {}
        # program -> the instance that last served it, which is where a
        # prefetch must land for its prefix to be the one the successor hits.
        self._instance_of = {}
        self._live_router = getattr(self, "_live_router", None)
        self._schedulers = ()

    # ------------------------------------------------------------------
    # Event: a turn is routed (arrival at the gateway)
    # ------------------------------------------------------------------

    def steal_tick(self, now_ns):
        """One SAGA steal attempt per routing tick, or nothing.

        The move is a *waiting* turn changing instance: its queue entry is
        taken from the source and handed to the destination, which is a real
        transfer of pending work. A generating call is never touched -- that
        would be active-call migration, which no path here implements.
        """
        if (not self.saga_stealing or self.routing_value != "saga-placement"
                or not self._schedulers):
            return ()
        self._now_ns = max(self._now_ns, now_ns)

        def move(session, source, destination):
            src, dst = self._schedulers[source], self._schedulers[destination]
            for req in list(src.request):
                owner = req.session_id if req.session_id is not None else req.workflow_id
                if str(owner) != str(session):
                    continue
                if req.admit_seq is not None or req.num_computed_tokens:
                    continue          # started here; stealing it is migration
                src.request.remove(req)
                # Keep the destination's arrival order, which its scheduler
                # relies on. bisect's key= is 3.10+, and this runs under the
                # simulator image's interpreter, so the keys are built here.
                import bisect
                keys = [(r.arrival, r.id) for r in dst.request]
                dst.request.insert(
                    bisect.bisect_right(keys, (req.arrival, req.id)), req)
                self._req_instance[req.id] = destination
                return True
            return False

        return self._live_router.steal(
            self._schedulers, self._s(now_ns), self._instance_load, move)

    def routing_stats(self):
        return dict(self._live_router.stats) if self._live_router is not None else {}

    def prefetch_tick(self, now_ns):
        """Submit any recomputes whose tool result is nearly due.

        The synthetic prefill goes to the instance that last served the
        program, since that is where its successor's prefix must be, and it
        competes for the batch like any other work: a prefetch issued too
        early costs visibly rather than being free.
        """
        if self._prefetcher is None or not self._schedulers:
            return ()
        self._now_ns = max(self._now_ns, now_ns)

        def resident(program_id, ids, instance):
            if not 0 <= instance < len(self._schedulers):
                return False
            cache = self._schedulers[instance].memory.npu_prefix_cache
            return cache.match_prefix(ids).hit_length >= len(ids)

        issued = []
        for program_id, ids, instance in self._prefetcher.due(
                self.programs, now_ns, resident):
            if not 0 <= instance < len(self._schedulers):
                continue
            rid = self._prefetcher.next_id()
            sched = self._schedulers[instance]
            # One generated token, exactly as the engine-side prefetch does.
            sched.add_request([rid, sched.model, len(ids), 1, now_ns, instance,
                               list(ids), []])
            for req in sched.request:
                if req.id == rid:
                    req.prefetch_of = str(program_id)
                    break
            self._prefetcher.note_issued(program_id, rid)
            issued.append((program_id, rid, instance))
        return tuple(issued)

    def prefetch_stats(self):
        return dict(self._prefetcher.stats) if self._prefetcher is not None else {}

    def register_session(self, session):
        """Record a session's AFS inputs when fairness is enabled."""
        if self._fairness is None:
            return
        self._fairness.register(session.get("session_id"), session)

    def fairness_stats(self):
        return dict(self._fairness.stats) if self._fairness is not None else {}

    def _fair_priority(self, program_id, now_ns):
        """Weighted attained service: a tenant with a larger share may accrue
        more service before it yields. Smallest stamp runs first, so the share
        divides rather than multiplies."""
        if self._fairness is None:
            return None
        self._fairness.compute(self.programs, now_ns)
        share = self._fairness.share_of(program_id)
        if not share:
            return None
        pcb = self.programs.get(program_id)
        self.stats["saga_fair_stamps"] += 1
        return int(round(pcb.attained_service_s * 1000.0 / share))

    def attach_schedulers(self, schedulers):
        """Give the live routers the instances they must observe."""
        self._schedulers = tuple(schedulers)

    def _instance_load(self, sched):
        util = self._kv_utilization(sched.memory)
        return 0.0 if util is None else util

    @staticmethod
    def _outstanding(sched):
        return len(sched.request) + sum(len(b.requests) for b in sched.inflight)

    def select_instance(self, req_data, default_select, now_ns):
        """Instance for this request: shared routing policy when one is
        configured, else the router's stock policy via default_select."""
        self._now_ns = max(self._now_ns, now_ns)
        program_id, turn_idx = self._program_identity(req_data)
        if program_id is not None and req_data.get('completed_tool_duration_s') is not None:
            self.programs.observe_completed_tool(
                program_id, req_data['completed_tool_duration_s'])
        if (self._live_router is not None and program_id is not None
                and self._schedulers):
            now_s = self._s(req_data["arrival_time_ns"])
            self.retention_exec.observe_arrival(
                self.programs.get(program_id), now_s)
            if self.routing_value == "autellix-route":
                instance = self._live_router.route(
                    program_id, int(req_data.get("input_toks", 0) or 0),
                    self._schedulers, self._outstanding)
            else:
                instance = self._live_router.route(
                    program_id, self._schedulers, now_s, self._instance_load)
            self._req_instance[req_data["index"]] = instance
            self._instance_of[str(program_id)] = instance
            return instance
        if self.routing_exec is not None and program_id is not None:
            self.retention_exec.observe_arrival(
                self.programs.get(program_id), self._s(req_data["arrival_time_ns"]))
            instance = self.routing_exec.route(
                program_id, turn_idx, self._s(req_data["arrival_time_ns"]))
        else:
            instance = default_select()
        self._req_instance[req_data["index"]] = instance
        if program_id is not None:
            self._instance_of[str(program_id)] = instance
        return instance

    def on_turn_routed(self, req_data):
        """Priority stamp + retention arrival for a routed turn.
        Returns the priority to store on the request (None = unstamped)."""
        program_id, turn_idx = self._program_identity(req_data)
        if program_id is None:
            return None
        if req_data.get('completed_tool_duration_s') is not None:
            self.programs.observe_completed_tool(
                program_id, req_data['completed_tool_duration_s'])
        now_s = self._s(req_data["arrival_time_ns"])
        # Gap observation must precede the stamp: the dispatch transition
        # (on_turn_release inside stamp) clears gap_started_ts, the only
        # record of the just-ended gap's duration.
        self.retention_exec.observe_arrival(
            self.programs.get(program_id), now_s)
        # Stamp first (a pinned-first scheduler must see the protection),
        # then release the program's parked protection for this turn.
        if self._prefetcher is not None:
            self._prefetcher.note_turn_arrival(program_id)
        priority = self.scheduling_exec.stamp(program_id, turn_idx, now_s)
        fair = self._fair_priority(program_id, self._now_ns)
        if fair is not None:
            priority = fair
        native_continuation = (self.retention_value == 'min-waste'
                               and self.min_waste_swap
                               and self.scheduling_value == 'fcfs')
        if native_continuation:
            self._infercept_arrivals.setdefault(program_id, req_data['arrival_time_ns'])
        previous_copy = (None if native_continuation else
                         self._swap_programs.pop(program_id, None))
        if previous_copy is not None:
            for swap in self._host_swaps.values():
                swap.cancel(previous_copy)
        if self._release_at_scheduled:
            # Queue-persistent protection: hold through the queue wait.
            # The parked entry's deadline moves to the hold sentinel so the
            # pressure valve treats it as unexpired and breaks it
            # latest-deadline-first — mirroring the released code, which
            # breaks the pin with the most remaining TTL first and never
            # expires a pin whose next turn is waiting.
            tag = self.programs.get(program_id).kv_request_id
            entry = self._parked.get(tag) if tag is not None else None
            if entry is not None:
                entry.deadline_ns = _HOLD_SENTINEL_NS
            self._pending_release.add(program_id)
        else:
            self.retention_exec.turn_arrival(
                program_id, now=now_s,
                kv_utilization=self._retention_utilization(self._last_memory))
        return priority

    # ------------------------------------------------------------------
    # Wider scheduling hooks (called from Scheduler.schedule)
    # ------------------------------------------------------------------

    @staticmethod
    def _evolved_class(mod, modname, clsname):
        return load_custom_policy(mod, modname, clsname)

    def _pcb_of_req(self, req_obj):
        program_id = req_obj.session_id if req_obj.session_id is not None \
            else req_obj.workflow_id
        if program_id is None:
            return None
        return self.programs.get(program_id)

    def select_victim(self, candidates, now_ns):
        """Policy's choice among running requests to preempt, or None for
        the engine default. Flat (non-program) requests fall back."""
        if not self.custom_hooks or not candidates:
            return None
        if self.scheduling_value == "autellix-mlfq":
            # Algorithm 1 already named the residents it dropped. The
            # simulator preempts by recomputation; Autellix swaps instead,
            # which is the standing gap, not a substitution made here.
            for driver in self._autellix.values():
                chosen = {str(r.id) for r in getattr(driver, "last_preempt", ())}
                for r in candidates:
                    if str(r.id) in chosen:
                        self.stats["victim_overrides"] += 1
                        self._autellix_swap_out(r, now_ns)
                        return r
            return None
        views = []
        for r in candidates:
            pcb = self._pcb_of_req(r)
            if pcb is None:
                return None
            computed = int(r.num_computed_tokens)
            views.append(self._scheduling_mod.VictimView(
                pcb=pcb, priority=int(r.priority),
                prompt_tokens=int(r.original_input),
                computed_tokens=computed,
                generated_tokens=max(0, computed - int(r.original_input)),
                is_prefill=bool(r.is_prefill())))
        idx = self.scheduling_exec.policy.victim(views, self._s(now_ns))
        if idx is None:
            return None
        idx = int(idx)
        if not 0 <= idx < len(candidates):
            return None
        self.stats["victim_overrides"] += 1
        return candidates[idx]

    # ------------------------------------------------------------------
    # SAGA workflow-aware LRU (opt-in: --saga-eviction-order)
    # ------------------------------------------------------------------
    def saga_evictor(self, memory):
        """Install and return this instance's reclaim ranking."""
        key = memory.instance_id
        ev = self._saga_evict.get(key)
        if ev is None:
            from .saga_eviction import SagaEvictionOrder
            ev = SagaEvictionOrder()
            self._saga_evict[key] = ev
            memory.eviction_order = ev
        return ev

    def _saga_note(self, memory, program_id, now_ns, arrival, overlap=None):
        if not self.saga_eviction_order or program_id is None or memory is None:
            return
        ev = self.saga_evictor(memory)
        if arrival:
            ev.note_turn_arrival(program_id, now_ns, overlap)
        else:
            ev.note_turn_complete(program_id, now_ns)
        self.stats["saga_ranked"] = sum(e.stats["ranked"] for e in self._saga_evict.values())
        self.stats["saga_lru_fallback"] = sum(
            e.stats["lru_fallback"] for e in self._saga_evict.values())

    # ------------------------------------------------------------------
    # Autellix MLFQ planner (scheduling_value == "autellix-mlfq")
    # ------------------------------------------------------------------
    def _autellix_driver(self, memory):
        """One AutellixRuntime per instance: queues and clocks are per engine."""
        key = memory.instance_id
        driver = self._autellix.get(key)
        if driver is None:
            from .autellix_driver import AutellixDriver
            driver = AutellixDriver(self._autellix_cfg,
                                    self._autellix_overprovision)
            self._autellix[key] = driver
        return driver

    @staticmethod
    def _autellix_fits(memory, sched):
        """Cumulative feasibility for the planner: the scheduler's own test.

        Read-only, as the runtime requires -- nothing is reserved here. Mirrors
        Scheduler.schedule_with_prefix: a sequence slot, a share of the token
        budget, and enough NPU memory counting what the LRU could free.
        """
        from .memory_model import Device

        def fits(req, selected):
            if len(selected) + 1 > int(sched.max_num_seqs):
                return False
            group = list(selected) + [req]
            budget = int(sched.max_num_batched_tokens)
            tokens, spent = {}, 0
            for r in group:
                if r.is_prefill():
                    want = max(0, int(r.original_input) - int(r.num_computed_tokens))
                    cap = int(sched.long_prefill_token_threshold)
                    if 0 < cap < want:
                        want = cap
                    # Chunked prefill: this step takes what is left of the
                    # budget, not the whole remaining prompt. Demanding the
                    # whole prompt fit made every prefill longer than
                    # max_num_batched_tokens permanently infeasible, so the
                    # planner selected nothing and the engine deadlocked with
                    # a full-but-evictable pool (job 42596388, 16.7k-token
                    # SWE-bench prompts against a 16,384 budget).
                    want = min(want, budget - spent)
                else:
                    want = 1 if budget - spent > 0 else 0
                if want <= 0:
                    return False
                tokens[r.id] = want
                spent += want
            need = memory.get_block_kv(group, len(group), tokens)
            usable = (memory.avail_size(Device.NPU)
                      + memory.evictable_size(Device.NPU))
            return need <= usable

        return fits

    def autellix_plan(self, waiting, running, memory, sched, now_ns):
        """Replace the waiting order with Algorithm 1's selection."""
        driver = self._autellix_driver(memory)
        out = driver.plan(waiting, running, self._pcb_of_req, self._s(now_ns),
                          self._autellix_fits(memory, sched))
        self.stats["autellix_demoted"] = sum(
            d.stats["demoted"] for d in self._autellix.values())
        self.stats["autellix_promoted"] = sum(
            d.stats["promoted"] for d in self._autellix.values())
        self.stats["autellix_overprovisioned"] = sum(
            d.stats["overprovisioned"] for d in self._autellix.values())
        return out

    def on_batch_started(self, memory, requests, now_ns):
        """Actual batch membership, which may differ from the plan."""
        if self.scheduling_value != "autellix-mlfq":
            return
        self._autellix_driver(memory).batch_started(requests, self._s(now_ns))

    def on_batch_done(self, memory, finished_ids, now_ns, execution_ns):
        """Measured execution interval, which is what demotes a call."""
        if self.scheduling_value != "autellix-mlfq":
            return
        self._autellix_driver(memory).batch_finished(
            finished_ids, self._s(now_ns), execution_ns / 1e9)

    def on_request_dropped(self, memory, req_id, now_ns):
        if self.scheduling_value != "autellix-mlfq":
            return
        self._autellix_driver(memory).drop(req_id, self._s(now_ns))

    #: Preemptions the plan may force per scheduling tick. The runtime states
    #: that it "does not substitute recomputation for required swapping", and
    #: the simulator's only preemption IS recomputation, so applying every
    #: plan.preempt each tick livelocks: a resident the plan did not select is
    #: preempted, re-admitted, and preempted again (job 42631045: 13,468
    #: preemptions on 50 programs, against 1,254 before). A quantum yield is
    #: one call stepping aside for a better-placed waiter, so bound it at that.
    _PLAN_PREEMPTS_PER_TICK = 1

    def apply_scheduling_plan(self, sched, running, now_ns):
        """Enforce quantum yields before native running-first admission.

        Only a resident the plan dropped *and* whose MLFQ queue is worse than
        some waiting call's yields: that is the paper's rule, a call giving way
        when its quantum has expired and better-placed work is queued. A
        resident the plan merely could not fit this step keeps its KV and waits
        its turn, because preempting it here costs a full re-prefill.
        """
        if self.scheduling_value != "autellix-mlfq":
            return running
        driver = self._autellix_driver(sched.memory)
        calls = driver.runtime.calls
        victims = {r.id for r in driver.last_preempt}
        if victims:
            waiting_q = [calls[driver._rid(r)].queue for r in sched.request
                         if driver._rid(r) in calls]
            best_waiting = min(waiting_q) if waiting_q else None
            if best_waiting is None:
                victims = set()
            else:
                victims = {r.id for r in driver.last_preempt
                           if driver._rid(r) in calls
                           and calls[driver._rid(r)].queue > best_waiting}
        self._last_memory = sched.memory
        self._now_ns = now_ns
        kept, yielded = [], 0
        for req in running:
            if req.id not in victims or yielded >= self._PLAN_PREEMPTS_PER_TICK:
                kept.append(req)
                continue
            yielded += 1
            self._autellix_swap_out(req, now_ns)
            from .memory_model import Device
            sched.preemption_log.append((
                now_ns, req.id, req.num_computed_tokens,
                max(0, req.num_computed_tokens - req.original_input), 0,
                sched.memory.avail_size(Device.NPU),
                sched.memory.evictable_size(Device.NPU), req.admit_seq,
                sched._admit_counter, len(running), len(kept),
                req.original_input, req.npu_cache_hit))
            sched._preempt_recompute(req)
            sched.num_preemptions += 1
        return kept

    def filter_waiting(self, waiting, running, memory, n_inflight, now_ns,
                       pending_reserve=0, sched=None):
        """Apply the admission gate to the ordered waiting queue. Held
        requests keep their place for the next tick. Preempted requests
        are engine-internal and always pass. With the engine idle and
        everything held, the head passes (no gateway may idle the
        engine forever: the simulator would otherwise raise stuck)."""
        if not self.custom_hooks or (not waiting and self.scheduling_value != "autellix-mlfq"):
            return waiting
        if (waiting and waiting[0].infercept_session and not running
                and not n_inflight and sched is not None):
            from .memory_model import Device
            head = waiting[0]
            if not head._prefix_locked:
                head.num_computed_tokens = 0
                memory.prefix_match(head)
                if head.npu_cache_hit:
                    # Admission needs this prefix to survive. Exclude it from
                    # reclaimable capacity before deciding whether to preempt
                    # another owner, including when the two share an ancestor.
                    # This is a speculative admission lock, not newly retained
                    # session ownership; _drop_prefill releases it on failure.
                    memory.lock_prefix(head, Device.NPU)
                    head._prefix_locked = True
            # Native InferCept explicitly preempts younger ready owners when
            # the idle queue head cannot allocate its full prompt. Do not
            # silently drop the returning owner's lock just to try admission.
            for victim in reversed(waiting[1:]):
                need = memory.get_block_kv(
                    [head], 1, {head.id: max(1, head.original_input - head.num_computed_tokens)})
                usable = memory.avail_size(Device.NPU) + memory.evictable_size(Device.NPU)
                if need <= usable:
                    break
                if victim.infercept_session and victim._prefix_locked:
                    sched._preempt_recompute(victim)
                    sched.num_preemptions += 1
        if self.min_waste_fcfs_restore and memory.host_swap is not None:
            gate = self._restore_gates.get(memory.instance_id)
            if gate is None:
                from .host_swap import RestoreGate
                gate = RestoreGate(memory, memory.host_swap.profile, self.stats)
                self._restore_gates[memory.instance_id] = gate
            restore_budget = sched.max_num_batched_tokens if sched is not None else 2048
            if self.infercept_paper_token_budget is not None:
                restore_budget = min(restore_budget, self.infercept_paper_token_budget)
            waiting = gate.admit(waiting, restore_budget)
            if not waiting:
                return waiting
        if self.scheduling_value == "autellix-mlfq":
            # The planner decides membership and order; there is no separate
            # admission gate to run on top of it.
            if sched is None:
                raise RuntimeError(
                    "the Autellix planner needs the scheduler's token budget; "
                    "Scheduler must pass sched= to filter_waiting")
            return self.autellix_plan(waiting, running, memory, sched, now_ns)
        now_s = self._s(now_ns)
        from .memory_model import Device  # local: avoid import cycle at load
        bpt = max(1, memory._bytes_per_token)
        # Minus what the running set takes this step: the engine has already
        # allocated those blocks by the time its own gate runs, so counting
        # them free here admits turns vLLM would have held (see
        # Scheduler._running_step_kv).
        free_tok = max(0, memory.avail_size(Device.NPU) - pending_reserve) // bpt
        evict_tok = memory.evictable_size(Device.NPU) // bpt
        util = self._kv_utilization(memory)
        util = 0.0 if util is None else util
        # Running work has not reserved this step's KV yet. Use the same
        # projected allocation for utilization as for free capacity above.
        pool = memory.npu_mem - memory.weight
        if pool > 0:
            util = min(1.0, util + pending_reserve / pool)
        kept, held = [], 0
        trace = self._gate_trace
        if trace is not None:
            used_tok = max(0, memory.npu_used - memory.weight) // bpt
            prot_tok = sum(e.tokens for e in self._parked.values()
                           if e.memory is memory)
            trace[1].write(json.dumps({
                "event": "step", "ts": now_s, "n_running": len(running),
                "n_waiting": len(waiting), "n_inflight": n_inflight,
                "kv_utilization": util, "kv_free_tokens": int(free_tok),
                "kv_evictable_tokens": int(evict_tok),
                "kv_protected_tokens": int(prot_tok),
                "kv_used_tokens": int(used_tok),
                "kv_running_tokens": int(max(0, used_tok - evict_tok - prot_tok)),
                "kv_reserved_tokens": int(memory.npu_reserved // bpt)}) + "\n")
        for r in waiting:
            if r.preempt_seq is not None:
                kept.append(r)
                continue
            pcb = self._pcb_of_req(r)
            if pcb is None:
                kept.append(r)
                continue
            view = self._scheduling_mod.QueueView(
                n_running=len(running), n_waiting=len(waiting),
                n_inflight=n_inflight, kv_utilization=util,
                kv_free_tokens=int(free_tok), kv_evictable_tokens=int(evict_tok),
                prompt_tokens=int(r.original_input),
                cached_tokens=(int(memory.peek_prefix_hit(r)) if _GATE_PREFIX_PROBE
                               else int(getattr(r, "npu_cache_hit", 0) or 0)))
            if self.scheduling_exec.policy.admit(pcb, now_s, view):
                kept.append(r)
            else:
                held += 1
                if trace is not None:
                    trace[0].write(json.dumps({
                        "event": "hold", "ts": now_s,
                        "program_id": pcb.program_id, "request_id": r.id,
                        "kv_utilization": view.kv_utilization,
                        "kv_free_tokens": view.kv_free_tokens,
                        "kv_evictable_tokens": view.kv_evictable_tokens,
                        "prompt_tokens": view.prompt_tokens,
                        "cached_tokens": view.cached_tokens,
                        "n_running": view.n_running,
                        "n_waiting": view.n_waiting}) + "\n")
        if not kept and not running and n_inflight == 0:
            kept = [waiting[0]]
            held -= 1
        self.stats["admission_holds"] += max(0, held)
        return kept

    # ------------------------------------------------------------------
    # Event: a turn is admitted to the running batch (first schedule)
    # ------------------------------------------------------------------

    def bind_infercept_continuation(self, req, memory):
        """Hand resident KV to the successor before dropping its paused owner.

        Keep release time separate from FCFS session order. Swapped/discarded
        context still follows the existing restore/recompute paths.
        """
        if not (self.retention_value == 'min-waste' and self.min_waste_swap
                and self.scheduling_value == 'fcfs'):
            return
        program = req.session_id if req.session_id is not None else req.workflow_id
        if program is None:
            return
        req.infercept_session = True
        req.queue_arrival = self._infercept_arrivals.setdefault(program, req.arrival)
        previous = self._swap_programs.pop(program, None)
        if previous is None:
            return
        memory.prefix_match(req)
        history = self._infercept_history.pop(program, ())
        common = 0
        for old, new in zip(history, req.input_hash_ids):
            if old != new:
                break
            common += 1
        # Full-prompt replacement discards a divergent partial block. A
        # streaming continuation retains the previous computed boundary.
        if common < len(history):
            common = common // memory.block_size * memory.block_size
        end = min(common, req.original_input - 1)
        resident = max(req.npu_cache_hit, req.storage_cache_hit)
        if resident < end // memory.block_size * memory.block_size:
            req.infercept_recompute_end = end
            req.infercept_recompute_chunk = memory.host_swap.profile.S
        if req.npu_cache_hit:
            from .memory_model import Device
            memory.lock_prefix(req, Device.NPU)
            req._prefix_locked = True
            req.infercept_retained_prefix = True
        if req.storage_cache_hit:
            req.infercept_cpu_node = req.storage_last_node
            memory.second_tier_prefix_cache.inc_lock_ref(req.infercept_cpu_node)
        # An unfinished DMA keeps its own reference until acknowledgement.
        # The successor now owns the resident matching prefix independently.
        for swap in self._host_swaps.values():
            swap.cancel(previous)

    def on_turn_admitted(self, req_obj, now_ns):
        """Deferred release for queue-persistent retention: the program's
        parked protection is released only when its next turn actually
        starts running (Continuum released-code semantics). Idempotent —
        re-admission after preemption is a no-op."""
        if not self._pending_release:
            return
        program_id = req_obj.session_id if req_obj.session_id is not None \
            else req_obj.workflow_id
        if program_id is None or program_id not in self._pending_release:
            return
        self._pending_release.discard(program_id)
        self._now_ns = max(self._now_ns, now_ns)
        self.stats["admission_releases"] = self.stats.get("admission_releases", 0) + 1
        self.retention_exec.turn_arrival(
            program_id, now=self._s(now_ns),
            kv_utilization=self._retention_utilization(self._last_memory))

    # ------------------------------------------------------------------
    # Event: a turn completed (gap start)
    # ------------------------------------------------------------------

    def on_turn_complete(self, req_obj, program_id, turn_idx, tool_name,
                         context_tokens, memory, completion_time_ns):
        if req_obj is None:
            return
        self._now_ns = max(self._now_ns, completion_time_ns)
        instance = self._req_instance.pop(req_obj.id, None)
        if instance is not None and self.routing_exec is not None:
            self.routing_exec.turn_complete(instance)
        if program_id is None:
            return
        # PLAS: attained service = first schedule -> last token, i.e.
        # PREFILL + DECODE. This must mirror the real driver exactly
        # (bench/core/runner.py: `service_s = max(0.0, lt - st)` with
        # lt=last_token_ts, st=scheduled_ts).
        #
        # It previously used `latency - queuing_delay`, which is NOT that:
        # `set_que_delay` is re-called on every prefill chunk while `is_init`
        # holds, so queuing_delay measures arrival -> LAST CHUNK, and the
        # difference collapses to decode time with prefill excluded. Since
        # prefill scales with prompt size, that under-charged exactly the
        # large-prompt programs PLAS is meant to deprioritize.
        if req_obj.end_time >= 0 and req_obj.first_sched_ts >= 0:
            service_s = max(0.0, (req_obj.end_time - req_obj.first_sched_ts) / 1e9)
        elif req_obj.latency >= 0 and req_obj.queuing_delay >= 0:
            service_s = max(0.0, (req_obj.latency - req_obj.queuing_delay) / 1e9)
        else:
            service_s = 0.0
        self.scheduling_exec.turn_complete(
            program_id, service_s, turn_idx=turn_idx)

        self._last_memory = memory
        self._ctx = (memory, self._finished_chain_node(req_obj, memory),
                     context_tokens)
        self._ctx_req = req_obj
        # SAGA's ranking needs a last-touch time and, on a successor, the
        # overlap its predecessor's context actually supplied.
        if turn_idx:
            hit = int(getattr(req_obj, "first_cache_hit", 0) or 0)
            total = max(1, int(getattr(req_obj, "original_input", 0) or 1))
            self._saga_note(memory, program_id, completion_time_ns, True,
                            overlap=min(1.0, hit / total))
        self._saga_note(memory, program_id, completion_time_ns, False)
        if self._prefetcher is not None:
            self._prefetcher.note_gap_start(
                program_id, req_obj, self._instance_of.get(str(program_id), 0),
                completion_time_ns)
        if self.retention_value == 'min-waste' and self.min_waste_swap:
            self._swap_programs[program_id] = f'{program_id}:{turn_idx}'
            if self.scheduling_value == 'fcfs':
                self._infercept_history[program_id] = (
                    req_obj.input_hash_ids + req_obj.output_hash_ids
                )[:req_obj.num_computed_tokens]
        try:
            self.retention_exec.turn_complete(
                program_id, turn_idx, f"{program_id}:{turn_idx}",
                tool_name, now=self._s(completion_time_ns),
                context_tokens=context_tokens,
                kv_utilization=self._retention_utilization(memory),
            )
        finally:
            self._ctx = None
            self._ctx_req = None

    def _retention_utilization(self, memory):
        # SAGA replay callbacks share the router's observation snapshot.
        # Use that same sampled load for TTL inputs, not fresher live state.
        if self.routing_value == 'saga-placement' and self._live_router is not None:
            for worker in self._live_router._routing_workers:
                if self._schedulers[worker.worker].memory is memory:
                    return worker.load
            return None
        return self._kv_utilization(memory)

    @staticmethod
    def _kv_utilization(memory):
        """Fraction of the instance's KV pool in use (weights excluded),
        the same quantity vLLM reports as kv_cache_usage."""
        if memory is None:
            return None
        pool = memory.npu_mem - memory.weight
        if pool <= 0:
            return None
        used = memory.npu_used - memory.weight
        # SIM_KV_UTIL_SEMANTICS=vllm (default): match vLLM's kv_cache_usage,
        # where cached-evictable blocks (free queue) are NOT in use. "sim"
        # counts them as used (npu_used includes the prefix cache). Parked
        # (lock_ref>0) KV is USED under both -- see docs/sim-vllm-parity.md.
        if _KV_UTIL_SEMANTICS == "vllm":
            from .memory_model import Device  # local: avoid import cycle
            used -= memory.evictable_size(Device.NPU)
            # Native blocks leave the free queue when a batch is allocated,
            # before its KV is computed. Our separate reservation account
            # represents that occupancy until publication at completion.
            used += memory.npu_reserved
        return max(0.0, min(1.0, used / pool))

    def finish(self):
        for swap in self._host_swaps.values():
            swap.close()
        self.retention_exec.finish()
        for f in self._log_files:
            f.close()

    # ------------------------------------------------------------------
    # KVControl interface (called by RetentionExecutor)
    # ------------------------------------------------------------------

    def protect(self, request_id, deadline_ts):
        memory, node, tokens = self._ctx
        if memory is None or node is None or node == self._root(memory):
            return 0
        locked = -memory.npu_prefix_cache.inc_lock_ref(node)
        if locked <= 0:
            # Nothing newly protected (chain already locked elsewhere);
            # undo our reference so the lock count stays balanced.
            memory.npu_prefix_cache.dec_lock_ref(node)
            return 0
        self._parked[request_id] = _ProtEntry(
            memory, node, int(deadline_ts * 1e9),
            self._pin_chain_tokens(node, self._root(memory)))
        blocks = max(1, locked // self.block_size)
        self.stats["protected"] += blocks
        return blocks

    def release(self, request_id):
        entry = self._parked.pop(request_id, None)
        if entry is None:
            return 0
        freed = entry.memory.npu_prefix_cache.dec_lock_ref(entry.node)
        blocks = max(1, freed // self.block_size) if freed else 0
        self.stats["released"] += blocks
        return blocks

    def evict(self, request_id):
        entry = self._parked.pop(request_id, None)
        if entry is not None:
            entry.memory.npu_prefix_cache.dec_lock_ref(entry.node)
            memory, node = entry.memory, entry.node
        else:
            if self._ctx is None:
                return 0
            memory, node, _ = self._ctx
            if memory is None or node is None:
                return 0
        evicted = memory.npu_prefix_cache.evict_chain(node)
        memory.apply_kv_cache_events()
        return max(1, evicted // self.block_size) if evicted else 0

    # ------------------------------------------------------------------
    # Experimental host copies: queue now, transfer on a subsequent batch
    # ------------------------------------------------------------------
    def swap(self, request_id):
        """Queue a copy; no GPU memory is freed by a completion callback."""
        if self._ctx is None or self._ctx_req is None:
            return 0
        memory = self._ctx[0]
        if memory is None or memory.host_swap is None:
            raise RuntimeError('min-waste host copies require a configured transfer controller')
        return memory.host_swap.enqueue(request_id, self._ctx_req, self._now_ns)

    def configure_host_swap(self, memory, bandwidth_gbs, profile=None,
                            policy=None, flag='--min-waste-swap'):
        """Give this instance a host tier and a copy controller.

        `profile` is the measured forward-time model the budget is sized from
        (T_swap(N) = T_fwd(B)); it defaults to the retention policy's, which is
        where min-waste already keeps it. `policy` decides ranking and whether
        a queued source may be dropped -- InferCept scores waste, Autellix
        never drops because its victim would otherwise re-prefill.
        """
        from .host_swap import HostSwap
        import math
        if (bandwidth_gbs is None or not math.isfinite(bandwidth_gbs)
                or bandwidth_gbs <= 0):
            raise ValueError(f'{flag} requires a measured per-rank link rate '
                             'in GB/s, not CPU DRAM bandwidth')
        if memory.prefix_storage is not None or memory.enable_prefix_sharing or memory.pp_size != 1:
            raise ValueError('experimental host copies require PP1 and a private CPU tier')
        if memory.host_swap is not None:
            raise ValueError('host swap already configured for this instance')
        if profile is None:
            profile = getattr(self.retention_exec.policy, 'profile', None)
        if profile is None:
            raise ValueError(f'{flag} needs a measured forward-time profile to '
                             'size the transfer budget')
        memory.enable_swap_tier()
        memory.cpu_mem_bw_gbs = bandwidth_gbs
        memory.host_swap = HostSwap(memory, profile, self.stats, policy=policy)
        self._host_swaps[id(memory)] = memory.host_swap

    def _swap_queue_limit(self, memory):
        """Tokens the swap queue may hold before a victim recomputes instead.

        A queued copy owns its source chain until its last chunk lands, so
        those tokens are unreclaimable. The bound is therefore whatever the
        pool has *beyond* what the engine needs to keep scheduling: one full
        token budget plus a block for every sequence slot, which is the most
        a single step can newly allocate. Derived from the engine's own
        limits rather than a fraction picked by hand, because this cap
        silently turns swaps into recomputes -- the thing --autellix-swap
        exists to avoid -- so it must bind as rarely as the deadlock allows.
        """
        pool = max(1, (memory.npu_mem - memory.weight)
                   // max(1, memory._bytes_per_token))
        sched = None
        idx = memory.instance_id
        if self._schedulers and 0 <= idx < len(self._schedulers):
            sched = self._schedulers[idx]
        if sched is None:
            headroom = 2 * memory.block_size
        else:
            # A chunked prefill accumulates KV across every chunk, so the
            # space the engine must keep is one *request's* worst-case
            # footprint, not one step's tokens. Job 42632952 wedged with a
            # head needing 41,376 tokens of cumulative KV while the headroom
            # reserved 18,432: the cap was sized for a chunk.
            longest = int(sched.config.get('max_position_embeddings',
                                           sched.max_num_batched_tokens))
            headroom = longest + int(sched.max_num_seqs) * int(memory.block_size)
        return max(0, pool - headroom)

    def _autellix_swap_out(self, req, now_ns):
        """Autellix preempts to host memory, not to recomputation.

        Called while the victim still has its computed tokens: the scheduler
        resets them immediately afterwards. The copy owns the source chain
        until it lands, so the successor restores instead of re-prefilling.

        That ownership is the catch. In this radix host-copy model the source
        stays *resident and locked* until its last chunk is transferred, so
        queueing faster than the link drains walks the pool to fully locked
        and the engine deadlocks with nothing evictable (job 42610546:
        115,824 tokens locked, 0 evictable, `valve_fail`). Real Autellix does
        not have this problem because a swapped-out call's pages are released
        as they move. Here the queue is bounded instead: once locking this
        chain would leave less than a couple of batches' worth evictable, the
        victim is preempted by recomputation, which is what would have
        happened without swapping at all.
        """
        if not self.autellix_swap:
            return
        memory = self._last_memory
        if memory is None or memory.host_swap is None:
            return
        # Tokens already pinned by queued copies, plus what this one would
        # pin. Measured against the pool rather than against evictable_size(),
        # which is 0 whenever the running set holds the cache -- exactly when
        # a victim is being chosen.
        swap = memory.host_swap
        # Copy progress does not release any source nodes in this substrate.
        # Count full chains until completion (conservative for shared prefixes).
        sources = {id(c): c for c in swap.queued.values()}
        for items in swap.pending.values():
            for copy, _, _, _ in items:
                sources[id(copy)] = copy
        outstanding = sum(len(c.ids) for c in sources.values())
        if outstanding + int(req.num_computed_tokens) > self._swap_queue_limit(memory):
            self.stats["autellix_swap_declined"] += 1
            return 0
        blocks = memory.host_swap.enqueue(str(req.id), req, self._now_ns)
        self.stats["autellix_swapped"] += 1
        return blocks

    def kv_stats(self):
        out = dict(self.stats)
        out["currently_protected"] = len(self._parked)
        return out

    # ------------------------------------------------------------------
    # Safety valve (called from MemoryModel.evict_prefix_cache)
    # ------------------------------------------------------------------

    def parked_tokens(self, memory):
        """Tokens currently parked (protected) on this memory."""
        return sum(e.tokens for e in self._parked.values() if e.memory is memory)

    def admission_parked_tokens(self, memory):
        """Nominal parked coverage for legacy diagnostics, including sharing.

        Allocation feasibility uses reclaimable_parked_tokens instead: shared
        references are counted once and running owners cannot be reclaimed.
        """
        return self.parked_tokens(memory)

    @staticmethod
    def _pin_chain_tokens(node, root):
        total = 0
        while node is not root:
            total += len(node.key)
            node = node.parent
        return total

    def reclaimable_parked_tokens(self, memory):
        """Count each node whose references are all retention pins once."""
        refs = {}
        root = self._root(memory)
        for entry in self._parked.values():
            if entry.memory is not memory:
                continue
            node = entry.node
            while node is not root:
                refs[node] = refs.get(node, 0) + 1
                node = node.parent
        return sum(len(node.key) for node, count in refs.items()
                   if node.lock_ref == count)

    def _plan_pin_reclaim(self, memory, tokens_needed):
        """Simulate tail releases without changing pins, nodes, or counters.

        Each (node, page offset) identifies a physical-sized slice of the
        current compressed tree. Two pins dropping the same slice free it only
        when its simulated reference count reaches zero. The plan contains
        token counts per pin, so later radix splits do not invalidate it.
        """
        cache = memory.npu_prefix_cache
        deficit = tokens_needed - cache.evictable_size()
        if deficit <= 0:
            return []
        if deficit > self.reclaimable_parked_tokens(memory):
            return None
        mine = [(tag, e) for tag, e in self._parked.items()
                if e.memory is memory]
        expired = sorted((x for x in mine if x[1].deadline_ns <= self._now_ns),
                         key=lambda x: x[1].deadline_ns)
        unexpired = sorted((x for x in mine if x[1].deadline_ns > self._now_ns),
                           key=lambda x: -x[1].deadline_ns)
        remaining_refs = {}
        plan = []
        freed_total = 0
        for tag, entry in expired + unexpired:
            node = entry.node
            drop = freed = 0
            while node is not cache.root_node:
                for end in range(len(node.key), 0, -cache.page_size):
                    size = min(cache.page_size, end)
                    page = (node, end)
                    refs = remaining_refs.get(page, node.lock_ref)
                    if refs <= 0:
                        raise RuntimeError("Retention pin has no matching cache reference")
                    remaining_refs[page] = refs - 1
                    drop += size
                    if refs == 1:
                        freed += size
                        freed_total += size
                    if freed_total >= deficit:
                        plan.append((tag, drop, freed))
                        return plan
                node = node.parent
            if drop:
                plan.append((tag, drop, freed))
        # A failed plan must never consume even a zero-yield pin reference.
        return None

    def ensure_evictable_tokens(self, memory, tokens_needed, site="chunk"):
        """Apply a feasible retention-reclamation plan, or leave pins intact.

        Expired pins precede unexpired pins; unexpired deadlines are ordered
        latest first. References are removed tail-first in whole pages. Shared
        prefixes become evictable only after their last pin reference is gone,
        and a running reference continues to protect its blocks.
        """
        cache = memory.npu_prefix_cache
        plan = self._plan_pin_reclaim(memory, tokens_needed)
        if plan is None:
            return False
        for tag, drop, expected_freed in plan:
            entry = self._parked[tag]
            key = ("reclaimed_expired" if entry.deadline_ns <= self._now_ns
                   else "reclaimed_forced")
            node, unlocked, freed = cache.dec_lock_ref_tail(entry.node, drop)
            if unlocked != drop or freed != expected_freed:
                raise RuntimeError("Retention reclamation diverged from its plan")
            if freed:
                blocks = max(1, freed // self.block_size)
                self.stats[key] += blocks
                site_key = key + "_" + site
                self.stats[site_key] = self.stats.get(site_key, 0) + blocks
            if node is cache.root_node:
                del self._parked[tag]
            else:
                entry.node = node
                # tokens describes the remaining reference chain, not the
                # uniquely protected footprint when the pin was first taken.
                entry.tokens = self._pin_chain_tokens(node, cache.root_node)
        return True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finished_chain_node(req_obj, memory):
        """Locate the finished request's cached chain in the radix tree.
        cache_finished_req nulls req.npu_last_node at completion, so the
        chain is re-matched from the request's hashed tokens, exactly
        the key cache_finished_req inserted ((input+output)[:-1],
        page-aligned). None when prefix caching is off or nothing is
        cached."""
        if req_obj is None or memory is None:
            return None
        cache = getattr(memory, "npu_prefix_cache", None)
        if cache is None or not req_obj.input_hash_ids:
            return None
        token_ids = (req_obj.input_hash_ids + (req_obj.output_hash_ids or []))[:-1]
        aligned = len(token_ids) // cache.page_size * cache.page_size
        if aligned <= 0:
            return None
        res = cache.match_prefix(token_ids[:aligned])
        if res.hit_length <= 0 or res.last_device_node == cache.root_node:
            return None
        return res.last_device_node

    def _oracle_table(self):
        """Trace ground truth for the clairvoyant probes, built once and
        shared by both axes. Requires the dataset path (wired from
        --dataset in __main__)."""
        if self._oracle_tbl is None:
            if not self._oracle_table_path:
                raise ValueError(
                    "oracle policies need the trace: no dataset path was "
                    "passed to UnifiedPolicyAdapter (oracle_table_path)")
            from policies.oracle import OracleTable
            path = self._oracle_table_path
            # Same resolution as Router.load_requests: the simulator
            # chdir's into astra-sim/, so repo-relative paths need '../'.
            if not os.path.isabs(path):
                path = f'../{path}'
            self._oracle_tbl = OracleTable.from_jsonl(path)
        return self._oracle_tbl

    @staticmethod
    def _program_identity(req_data):
        """(program_id, turn_idx) for a request row, or (None, None) for
        flat requests. Sessions use session_id/sub_request_index; DAG
        workflows use workflow_id/node_id."""
        if req_data.get("session_id") is not None:
            return req_data["session_id"], req_data["sub_request_index"]
        if req_data.get("workflow_id") is not None:
            return req_data["workflow_id"], req_data["node_id"]
        return None, None

    @staticmethod
    def _s(ns):
        return ns / 1e9

    @staticmethod
    def _root(memory):
        return memory.npu_prefix_cache.root_node
