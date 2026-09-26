"""Running the harness's policies on the program-aware planes.

The policies themselves are not reimplemented here and must not be: this
imports the SAME classes the real GPU harness runs (`policies/continuum.py`,
`policies/saga.py`, ...). A policy that is a different object on the two
hosts is not one policy, and the arena's whole claim is that it compares the
same decision rule against a real measurement.

What this module is, then, is a translator, and it translates in exactly one
direction: plane state -> the record a policy reads. There is deliberately no
`ProgramTable` here. The old adapter kept one, and had to, because the old
engine had no program record of its own; keeping a second one now would put two
copies of `turns_completed`, `in_gap` and `attained_service_s` in the same
process, updated by different code, to be found disagreeing later. The
orchestrator owns those facts and a `ProgramControlBlock` is built from it on
demand.

Three things the planes hand over that a `ProgramState` does not carry -- a
queue snapshot for the admission gate, per-request token counts for the victim
rule, and per-instance load for routing -- arrive as small structs defined by
the planes themselves, so `program_scheduler` and `program_router` never import
`harness`. The dependency points one way: policies know nothing about planes,
planes know nothing about policies, and this module knows both.

Declining is always safe. Every hook returns the engine's default when a policy
returns None or raises, because under policy search candidates misbehave, and a
bad candidate should cost a fallback rather than a run.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .program_orchestrator import ProgramOrchestrator, ProgramState

# ---------------------------------------------------------------- policies

#: flag value -> (module attribute, kwargs builder). One table, so that adding
#: a value is a data change and so that the mapping can be compared against
#: `unified_policy`'s -- see `tests/test_program_policy.py`, which asserts the
#: two agree for every flag. Two adapters that disagree about what
#: `--retention continuum` means would each be self-consistent and produce two
#: different published numbers under one name.
RETENTION_VALUES = (
    "cache-lru", "evict-always", "ttl", "min-waste", "saga-ttl", "continuum",
    "evolved", "oracle-ttl",
)
SCHEDULING_VALUES = (
    "fcfs", "program-fcfs", "plas", "continuum", "evolved", "oracle-srpt",
)
ROUTING_VALUES = ("round-robin", "least-loaded", "session-affinity")


def resolve_policy(axis: str, value: Optional[str], cfg, *, default: str):
    """The policy a flag names: a registry key, or `module:Class`.

    One lookup for all three axes. Each axis used to have an if-chain naming
    every value and its constructor arguments, which meant the registry mapping
    a name to a class could not be the thing that built it -- so adding a
    policy took a table entry AND a branch, and the two could disagree.
    Construction now lives with the policy, as `from_config`.
    """
    import policies
    name = value or default
    if ":" in name:
        flag = {"kv": "retention"}.get(axis, axis)
        return load_unified(name, flag=f"--{flag}",
                            instantiate=False).from_config(cfg)
    if name == "evolved":
        from .unified_policy_adapter import load_custom_policy
        which = {"kv": ("harness.evolved_retention", "EvolvedRetention"),
                 "scheduling": ("harness.evolved_scheduling",
                                "EvolvedScheduling")}[axis]
        return load_custom_policy(policies, *which)()
    if name.startswith("oracle-"):
        from policies import oracle
        cls = {"oracle-ttl": oracle.OracleTTLRetention,
               "oracle-srpt": oracle.OracleSRPTScheduling}[name]
        return cls(cfg.oracle_table)
    cls = policies.VALUES.get(axis, {}).get(name)
    if cls is None:
        flag = {"kv": "retention"}.get(axis, axis)
        raise ValueError(
            f"unknown {flag} value: {name!r}. Known: "
            f"{sorted(policies.VALUES[axis])}, or module:Class for your own.")
    return cls.from_config(cfg)


def build_retention(value, mod=None, *, tau_s=None, min_waste_profile=None,
                    default_gap_s=0.0, waste_mod=None, oracle_table=None):
    """Kept for callers that pass tuning as keywords."""
    from policies.base import PolicyConfig
    return resolve_policy("kv", value, PolicyConfig(
        tau_s=tau_s, min_waste_profile=min_waste_profile,
        default_gap_s=default_gap_s, oracle_table=oracle_table),
        default="cache-lru")


def build_scheduling(value, mod=None, *, oracle_table=None):
    from policies.base import PolicyConfig
    return resolve_policy("scheduling", value,
                          PolicyConfig(oracle_table=oracle_table),
                          default="fcfs")


def build_routing(value, mod=None, *, num_instances=1, capacity_limit=None):
    from policies.base import PolicyConfig
    return resolve_policy("routing", value, PolicyConfig(
        num_instances=num_instances, capacity_limit=capacity_limit),
        default="least-loaded")


# -------------------------------------------------------- unified policies


def load_unified(spec: str, flag: str = "--policy", *,
                 instantiate: bool = True):
    """Instantiate ONE object that plays as many of the three axes as it likes.

    `spec` is `module:Class` (`harness.my_policy:MyPolicy`). The object is
    assigned to every axis whose interface it implements, so a policy that
    decides retention AND scheduling from one piece of state is one object with
    one piece of state -- not two objects that happen to share a file.

    That distinction is the reason this exists. The evolve harness could
    already put both classes in `evolved_joint.py` and shim each axis onto it,
    but a shim shares a MODULE: the two instances still had to coordinate
    through class attributes or globals, which is a shared mutable the search
    could corrupt and nothing would report. And routing was never in it at all,
    so a policy could not say "hold this program's context BECAUSE I am about
    to route its next turn back here" -- the one sentence a genuinely unified
    policy exists to say.

    Implementing an axis means defining its method, not inheriting its base: a
    policy may implement one, two or three, and whatever it leaves out falls
    back to the engine's own rule for that axis.
    """
    import importlib

    if ":" not in spec:
        raise ValueError(
            f"{flag} wants module:Class, got {spec!r} "
            "(e.g. harness.my_policy:MyPolicy)")
    mod_name, cls_name = spec.rsplit(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ImportError(
            f"{flag} {spec}: cannot import {mod_name}. It must be importable "
            f"-- on PYTHONPATH, or in a directory passed with "
            f"--policy-root.") from e
    try:
        cls = getattr(mod, cls_name)
    except AttributeError:
        raise AttributeError(
            f"{flag} {spec}: {mod_name} has no {cls_name}. Defined: "
            f"{[n for n in vars(mod) if not n.startswith('_')]}")
    # `instantiate=False` hands back the CLASS, for callers that build it from
    # a PolicyConfig. Constructing here as well built every policy twice, the
    # first time with no arguments -- which any policy with a required
    # parameter (TTLRetention's tau_s) cannot survive.
    return cls() if instantiate else cls


def axes_of(policy, bases) -> Dict[str, bool]:
    """Which axes this object actually decides.

    By method presence, because the whole point of a unified policy is that it
    need not be three subclasses. A method inherited unchanged from the base is
    NOT an implementation -- it is the base's "no opinion", and attaching it
    would replace the engine's rule with a function that always declines while
    the counters said a policy was running.
    """
    r_base, s_base, rt_base = bases
    def owns(name, base):
        fn = getattr(type(policy), name, None)
        return fn is not None and fn is not getattr(base, name, None)
    return {
        "retention": owns("on_turn_complete", r_base) or owns("on_turn_arrival", r_base),
        "scheduling": (owns("priority", s_base) or owns("admit", s_base)
                       or owns("victim", s_base)),
        "routing": owns("route", rt_base),
    }


# ------------------------------------------------------------- the adapter


def _sec(ns):
    """Nanoseconds to seconds, passing None through.

    The engine's clock is nanoseconds on this plane; every quantity handed to a
    policy is seconds, because that is what the policies are written against and
    what the request plane gives them.
    """
    return None if ns is None else ns / 1e9


def _ns(sec):
    """Seconds back to engine nanoseconds, passing None through.

    The inverse of `_sec`. A policy is handed seconds and hands back a deadline
    in the same units; the KV plane stores engine ns. Converting one direction
    and not the other would put a 2-second TTL two nanoseconds in the future.
    """
    return None if sec is None else sec * 1e9


class ProgramPolicyAdapter:
    """One policy set, wired to one cluster's planes."""

    def __init__(self, orchestrator: ProgramOrchestrator, harness_mods,
                 retention: Optional[str] = None,
                 scheduling: Optional[str] = None,
                 routing: Optional[str] = None,
                 num_instances: int = 1,
                 tau_s: Optional[float] = None,
                 default_gap_s: float = 0.0,
                 min_waste_profile: Optional[str] = None,
                 capacity_limit: Optional[float] = None,
                 oracle_table=None,
                 unified=None,
                 paper: Optional[str] = None,
                 log_dir: Optional[str] = None) -> None:
        r_mod, s_mod, rt_mod, w_mod, p_mod = harness_mods
        self.orch = orchestrator
        self._pcb_cls = p_mod.ProgramControlBlock
        self._queue_cls = s_mod.QueueView
        self._victim_cls = s_mod.VictimView
        self._sched_base = s_mod.SchedulingPolicy

        self.unified = None
        self.paper = None
        if paper is not None:
            # One published system, every axis it decides. The axis flags stay
            # the way in for a single value; this is the way in for a paper,
            # and it is what connects halves that were only ever reachable one
            # at a time -- SAGA's routing rule has existed since before the
            # arena did and nothing selected it alongside SAGA's retention.
            import policies as _P
            built = _P.build(paper, tau_s=tau_s, pin_s=tau_s,
                             num_instances=num_instances,
                             capacity_limit=capacity_limit,
                             default_gap_s=default_gap_s,
                             min_waste_profile=min_waste_profile)
            self.paper = paper
            self.axes = {"retention": "kv" in built,
                         "scheduling": "scheduling" in built,
                         "routing": "routing" in built}
            self.retention = built.get("kv") or r_mod.CacheLRURetention()
            self.scheduling = built.get("scheduling") or s_mod.FCFSScheduling()
            self.routing = built.get("routing") or build_routing(
                routing, rt_mod, num_instances=num_instances,
                capacity_limit=capacity_limit)
        elif unified is not None:
            # One object, as many axes as it implements. Assigned to each axis
            # it decides; the rest keep the engine's default, so a unified
            # policy that only speaks to two of the three is a normal thing to
            # write rather than a special case to configure.
            obj = unified if not isinstance(unified, str) else load_unified(unified)
            self.unified = obj
            owns = axes_of(obj, (r_mod.RetentionPolicy, s_mod.SchedulingPolicy,
                                 rt_mod.RoutingPolicy))
            if not any(owns.values()):
                raise ValueError(
                    f"{type(obj).__name__} implements none of the three axes. "
                    "Define on_turn_complete/on_turn_arrival (retention), "
                    "priority/admit/victim (scheduling), or route (routing).")
            self.axes = owns
            self.retention = obj if owns["retention"] else r_mod.CacheLRURetention()
            self.scheduling = obj if owns["scheduling"] else s_mod.FCFSScheduling()
            if owns["routing"]:
                # The routing base keeps the per-instance in-flight vector,
                # `_least_loaded`, and the submit/complete bookkeeping. A
                # unified policy need not inherit from it -- and usually will
                # not, since it cannot inherit from all three -- so whatever it
                # did not bring, it is lent. Without this a policy whose own
                # `route` calls `self._least_loaded()` raises AttributeError on
                # its first placement.
                import types
                if not hasattr(obj, "inflight"):
                    rt_mod.RoutingPolicy.__init__(obj, num_instances)
                for name in ("_least_loaded", "on_submit", "on_complete"):
                    if not hasattr(obj, name):
                        setattr(obj, name,
                                types.MethodType(
                                    getattr(rt_mod.RoutingPolicy, name), obj))
                self.routing = obj
            else:
                self.routing = build_routing(routing, rt_mod,
                                             num_instances=num_instances,
                                             capacity_limit=capacity_limit)
        else:
            self.axes = {"retention": retention is not None,
                         "scheduling": scheduling is not None,
                         "routing": routing is not None}
            self.retention = build_retention(
                retention, r_mod, tau_s=tau_s,
                min_waste_profile=min_waste_profile,
                default_gap_s=default_gap_s, waste_mod=w_mod,
                oracle_table=oracle_table)
            self.scheduling = build_scheduling(scheduling, s_mod,
                                               oracle_table=oracle_table)
            self.routing = build_routing(routing, rt_mod,
                                         num_instances=num_instances,
                                         capacity_limit=capacity_limit)

        from policies.arrival_adapter import ArrivalAdapter
        self._arrival = ArrivalAdapter()

        #: Whether a protection outlives the next turn's ARRIVAL and is held
        #: until that turn is actually admitted. Continuum's released code does
        #: the latter, and the difference is the whole queue wait.
        self.release_at_scheduled = (
            getattr(self.retention, "release_event", "arrival") == "scheduled")
        #: Only the values that actually override a hook get it. A policy that
        #: stamps priorities and nothing else must run the engine's own
        #: admission and victim code, unchanged, or "same engine, one knob
        #: turned" stops being true.
        # `getattr(..., None)`, not attribute access: a unified policy need not
        # subclass anything, so the method may simply be absent -- which means
        # "no opinion on this hook", exactly like inheriting the base's.
        def _overrides(name: str) -> bool:
            fn = getattr(type(self.scheduling), name, None)
            return fn is not None and fn is not getattr(self._sched_base, name, None)

        self.wants_admit = _overrides("admit")
        self.wants_victim = _overrides("victim")
        self.wants_priority = _overrides("priority")

        self.counters: Dict[str, int] = {
            "pcb_projections": 0, "protect": 0, "release": 0, "evict": 0,
            "declined": 0, "policy_errors": 0, "placements": 0,
            "pressure_choices": 0,
        }
        #: instance -> turns this adapter has announced as placed and not yet
        #: as complete. Mirrors what the policy believes, so the two cannot
        #: drift into a failed assert.
        self._routed: Dict[int, int] = {}
        self._log_dir = log_dir
        #: One log per knob, because `utils/parity.py` diffs per knob and its
        #: field set differs for each. The executors produce three on the old
        #: and real paths; this plane does not use executors -- their
        #: record-then-ask step belongs to the orchestrator now -- but it still
        #: owes the three logs, or a run on these planes cannot be
        #: decision-compared against the GPU at all.
        self._decisions: Dict[str, List[dict]] = {
            "retention": [], "scheduling": [], "routing": []}
        self._seq = {"retention": 0, "scheduling": 0, "routing": 0}

    # ------------------------------------------------------- projection

    def pcb(self, state: Optional[ProgramState], *, context_tokens: int = 0):
        """A `ProgramControlBlock` view of plane state.

        Built fresh, never stored. The orchestrator is the owner of every field
        below; a cached copy would be a second record that drifts the first
        time one of them is updated without the other.
        """
        if state is None:
            return None
        self.counters["pcb_projections"] += 1
        pin = state.pins[0] if state.pins else None
        return self._pcb_cls(
            program_id=state.program_id,
            # Seconds. This plane carries the clock in nanoseconds (see
            # program_scheduler._ns) while the request plane converts at every
            # policy call site (UnifiedPolicyAdapter._s). The policies are the
            # SAME objects, and they compare against second-valued knobs --
            # TTLRetention(tau_s=2.0), Continuum's 2.0 s tool threshold. Handing
            # them nanoseconds made `now + tau` indistinguishable from `now` and
            # `tool_mean_gap_s <= 2.0` unsatisfiable, so a policy that ran on
            # both planes was two different policies.
            arrival_ts=_sec(state.arrival_ts),
            turn_idx=state.turn_idx,
            turns_completed=state.turns_completed,
            # Already seconds: program_scheduler measures it as
            # (end_time - first_sched_ts) / 1e9 and the orchestrator
            # accumulates that unchanged. Converting here too put a
            # 2-second service at 2 nanoseconds. Not every float on this
            # plane is a nanosecond timestamp.
            attained_service_s=state.attained_service_s,
            kv_instance=state.live_instance,
            context_tokens=context_tokens or state.context_tokens,
            kv_protected=bool(state.pins),
            kv_deadline_ts=_sec(pin.deadline_ts) if pin else None,
            kv_request_id=pin.request_id if pin else None,
            in_gap=state.in_gap,
            tool_name=state.tool_name,
            # Cluster-scoped, by tool name, from the orchestrator -- not
            # accumulated here. See ProgramOrchestrator.on_turn_arrival.
            tool_mean_gap_s=_sec(self.orch.tool_mean_gap_s(state.tool_name)),
            gap_started_ts=_sec(state.gap_started_ts),
            gap_n=state.gap_count,
            gap_sum_s=_sec(state.gap_sum_s),
        )

    # ------------------------------------------------------------ hooks

    def _call(self, obj, name: str, default: Any, *args) -> Any:
        """Call an OPTIONAL hook, if the policy has one.

        The published values inherit no-op defaults from their base, so the
        method is always there. A unified policy subclasses nothing -- it
        cannot subclass all three bases -- so an optional hook it did not
        write is simply absent. That is "no opinion", not misbehaviour, and
        counting it as a policy error would report 141 failures on a policy
        that is working exactly as written.
        """
        fn = getattr(obj, name, None)
        if fn is None:
            return default
        return self._guard(lambda: fn(*args), default)

    def _guard(self, fn: Callable, default: Any) -> Any:
        try:
            out = fn()
        except Exception:
            self.counters["policy_errors"] += 1
            return default
        if out is None:
            self.counters["declined"] += 1
            return default
        return out

    def priority_fn(self, state: ProgramState, now: float) -> Optional[int]:
        """`ProgramBatchScheduler.priority_fn`. `now` arrives in engine ns."""
        now = _sec(now)
        if not self.wants_priority:
            return None
        pcb = self.pcb(state)
        if pcb is None:
            return None
        priority = self._call(self.scheduling, "priority", None, pcb, now)
        if priority is not None:
            self._emit("scheduling", ts=now, program_id=state.program_id,
                       turn_idx=state.turn_idx, priority=int(priority))
        return priority

    def admit_fn(self, state: ProgramState, snapshot, now: float) -> bool:
        now = _sec(now)
        """`ProgramBatchScheduler.admit_fn`.

        `snapshot` is the plane's own `QueueSnapshot`; the harness's
        `QueueView` is built from it here so the plane needs no harness import.
        """
        if not self.wants_admit:
            return True
        pcb = self.pcb(state, context_tokens=snapshot.prompt_tokens)
        if pcb is None:
            return True
        view = self._queue_cls(
            n_running=snapshot.n_running,
            n_waiting=snapshot.n_waiting,
            n_inflight=snapshot.n_inflight,
            kv_utilization=snapshot.kv_utilization,
            kv_free_tokens=snapshot.kv_free_tokens,
            kv_evictable_tokens=snapshot.kv_evictable_tokens,
            prompt_tokens=snapshot.prompt_tokens,
            cached_tokens=snapshot.cached_tokens,
        )
        return bool(self._call(self.scheduling, "admit", True, pcb, now, view))

    def victim_fn(self, views: List, now: float) -> Optional[str]:
        now = _sec(now)
        """`ProgramBatchScheduler.victim_fn`.

        The contract returns an INDEX into the candidate list; the plane wants
        a request id. Translated here rather than changing either side: an
        index is what the published policies were written against, and a
        request id is what the plane can act on without a positional agreement
        that nothing checks.
        """
        if not self.wants_victim or not views:
            return None
        cands = [
            self._victim_cls(
                pcb=self.pcb(v.state, context_tokens=v.computed_tokens),
                priority=v.priority,
                prompt_tokens=v.prompt_tokens,
                computed_tokens=v.computed_tokens,
                generated_tokens=v.generated_tokens,
                is_prefill=v.is_prefill,
            )
            for v in views
        ]
        idx = self._call(self.scheduling, "victim", None, cands, now)
        if idx is None:
            return None
        try:
            return views[int(idx)].request_id
        except (ValueError, TypeError, IndexError):
            self.counters["policy_errors"] += 1
            return None

    def route_fn(self, state: ProgramState, instance_views: List,
                 now: float) -> Optional[int]:
        """`ProgramRouter.route_fn`.

        The contract returns `(instance, info)` -- the placement and why -- and
        the plane wants just the instance. Unpacked here, because the plane
        passing a tuple where it expects an int is not a decline the router can
        fall back from; it is a TypeError outside the try block.
        """
        pcb = self.pcb(state)
        if pcb is None:
            return None
        out = self._call(self.routing, "route", None, pcb, now)
        if out is None:
            return None
        instance = out[0] if isinstance(out, tuple) else out
        info = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        try:
            instance = int(instance)
        except (TypeError, ValueError):
            self.counters["policy_errors"] += 1
            return None
        self._emit("routing", ts=now, program_id=state.program_id,
                   turn_idx=state.turn_idx, instance=instance, info=info)
        return instance

    def on_placed(self, instance: int) -> None:
        """A turn was placed. The routing policies keep their own per-instance
        in-flight count and choose from it, so a host that never tells them
        leaves every policy deciding from a vector of zeros -- least-loaded
        then means "instance 0", always, and looks like a working policy."""
        self.routing.on_submit(instance)
        self._routed[instance] = self._routed.get(instance, 0) + 1
        self.counters["placements"] += 1

    # -------------------------------------------------------- retention

    def on_turn_complete(self, kv, state: ProgramState, request_id: str,
                         now: float) -> Optional[str]:
        """A turn ended and its tool gap began: keep, pin, or throw away.

        The action is applied to the KV plane here rather than returned,
        because "protect" and "evict" are different mechanisms with different
        costs and the caller should not have to know which is which.
        """
        # Balance the routing policy's in-flight count first: it is keyed on
        # the instance, and `on_complete` asserts the count is positive, so a
        # completion we never announced as a placement must not be announced
        # as a completion either.
        inst = state.live_instance
        if inst is not None and self._routed.get(inst, 0) > 0:
            self.routing.on_complete(inst)
            self._routed[inst] -= 1
        pcb = self.pcb(state)
        if pcb is None:
            return None
        self._signals(kv)
        # Seconds in, seconds out: the policy is given the clock it expects and
        # returns a deadline in the same units, which the KV plane stores as
        # engine nanoseconds.
        out = self._call(self.retention, "on_turn_complete", None,
                         pcb, request_id, _sec(now))
        if not out:
            return None
        action = out[0]
        deadline = _ns(out[1]) if len(out) > 1 else None
        info = out[2] if len(out) > 2 else None
        if action == "protect":
            kv.pin(state.program_id, request_id, now, deadline_ts=deadline)
            self.counters["protect"] += 1
        elif action == "evict":
            kv.evict(state.program_id)
            self.counters["evict"] += 1
        elif action == "release":
            kv.unpin(state.program_id, request_id)
            self.counters["release"] += 1
        self._record(action, state, request_id, _sec(now), _sec(deadline), info)
        return action

    def pressure_fn(self, programs: List, need_tokens: int,
                    now: float) -> Optional[List[str]]:
        """Which programs lose their context when memory is short.

        The fourth retention event, and the one the published policies do not
        have: they decide at turn boundaries -- pin now, release now -- and say
        nothing about who pays when the pool fills. That decision is the valve's
        by default, which is LRU, which is the engine's opinion and not a
        policy's.

        Optional by presence, like every other hook here: a policy that does not
        define `on_pressure` never sees this and the valve runs exactly as
        before. So no published policy changes behaviour by this existing,
        which is the property that lets it be added at all.

        Returns program ids in the order they should be given up. Returning
        None or nothing is a decline, and declining is safe -- a pressure hook
        that can deadlock the engine when a policy misbehaves is worse than no
        hook.
        """
        pcbs = [self.pcb(p) for p in programs]
        pcbs = [p for p in pcbs if p is not None]
        out = self._call(self.retention, "on_pressure", None,
                         pcbs, need_tokens, now)
        if not out:
            return None
        self.counters["pressure_choices"] += 1
        return [str(x) for x in out]

    def observe_arrival(self, state: ProgramState, now: float) -> None:
        """Called at the next turn's arrival BEFORE the gap record is cleared
        -- the only moment the just-ended gap's duration is observable."""
        now = _sec(now)
        pcb = self.pcb(state)
        if pcb is None:
            return
        if self._arrival.handles(self.retention):
            self._guard(lambda: self._arrival.observe(self.retention, pcb, now), None)
        else:
            self._call(self.retention, "observe_arrival", None, pcb, now)

    def on_turn_arrival(self, kv, state: ProgramState, now: float) -> None:
        """Release at arrival, unless the policy holds through the queue wait."""
        now = _sec(now)
        if self.release_at_scheduled:
            return
        self._maybe_release(kv, state, now)

    def on_turn_scheduled(self, kv, state: ProgramState, now: float) -> None:
        """Release at admission, for queue-persistent policies."""
        now = _sec(now)
        if not self.release_at_scheduled:
            return
        self._maybe_release(kv, state, now)

    def _maybe_release(self, kv, state: ProgramState, now: float) -> None:
        pcb = self.pcb(state)
        if pcb is None:
            return
        if self._arrival.handles(self.retention):
            out = self._guard(lambda: self._arrival.action(self.retention, pcb, now), None)
        else:
            out = self._call(self.retention, "on_turn_arrival", None, pcb, now)
        if out == "release":
            kv.unpin(state.program_id)
            self.counters["release"] += 1
            self._record("release", state, pcb.kv_request_id, now, None, None)

    def _signals(self, kv) -> None:
        """Hand the policy the system reading it is allowed to see."""
        sig = getattr(self.retention, "signals", None)
        if sig is not None:
            try:
                sig.kv_utilization = kv.pressure()
            except Exception:
                pass

    # ---------------------------------------------------------- logging

    def _emit(self, knob: str, **fields) -> None:
        rec = {"seq": self._seq[knob], "ts": fields.pop("ts")}
        rec.update(fields)
        self._seq[knob] += 1
        self._decisions[knob].append(rec)

    def _record(self, action, state, request_id, now, deadline, info) -> None:
        # `now` and `deadline` arrive in SECONDS -- every caller converts before
        # it gets here. Converting again inside was a second application for the
        # release path, whose hook had already done it, so a release at engine
        # 2 s logged 0.000000002. Seconds is what the request plane's
        # retention.jsonl and the real harness both write, and the audits
        # compare those columns line-for-line.
        self._emit("retention", ts=now, action=action,
                   program_id=state.program_id, turn_idx=state.turn_idx,
                   request_id=request_id, deadline_ts=deadline,
                   tool_name=state.tool_name, info=info)

    def finish(self) -> Optional[str]:
        """Flush the decision log. Same JSONL shape the real harness writes, so
        `harness/parity.py` can diff the two runs decision by decision."""
        if self._log_dir is None:
            return None
        import json
        import os
        os.makedirs(self._log_dir, exist_ok=True)
        written = []
        for knob, recs in self._decisions.items():
            if not recs:
                continue          # an axis this policy has no opinion on
            path = os.path.join(self._log_dir, f"{knob}.jsonl")
            with open(path, "w") as f:
                for d in recs:
                    f.write(json.dumps(d, sort_keys=True) + "\n")
            written.append(path)
        return ", ".join(written) if written else None

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counters)
