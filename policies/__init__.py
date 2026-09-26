"""Serving policies, one module per published system.

    base.py        KVPolicy | SchedulingPolicy | RoutingPolicy + the records
    executors.py   the machinery that applies a decision and logs it
    stock.py       vLLM v1 as shipped
    continuum.py   ContinuumKV + ContinuumScheduling
    saga.py        SagaKV + SagaRouting
    autellix.py    AutellixScheduling
    infercept.py   InferceptKV
    gate.py        the evolved joint champion (copy of evolve/champion_joint_41008503.py), named
    generic.py     primitives that are not papers (plain TTL, program FCFS)

The files used to be cut by AXIS, so one paper's policy was assembled from
pieces in two or three of them under names that did not say which paper they
came from. It cost something real: SAGA's affinity rule sat in `routing.py`
unassociated with SAGA, and the arena recorded its absence as a missing
feature. It was not missing, it was unlabelled.

`PAPERS` is the mapping the CLI reads. `VALUES` keeps the axis flag strings
working -- `--retention continuum` must still resolve to the same class,
because the 24-leg answer key was recorded under those strings and
`evaluation/drift.py` checks every spec against the reference meta.json.
"""
from . import autellix, base, continuum, executors, generic, infercept, stock
from . import utils
from .base import (                                              # noqa: F401
    KVPolicy, PriorityStamp, ProgramControlBlock, ProgramTable, QueueView,
    RetentionDecision, RetentionPolicy, RoutingDecision, RoutingPolicy,
    SchedulingPolicy, SystemSignals, VictimView)
from .executors import (                                         # noqa: F401
    RetentionExecutor, RoutingExecutor, SchedulingExecutor)
from . import gate, saga, search_seed

AXES = ("kv", "scheduling", "routing")

#: paper -> the planes it decides. A missing key means no opinion on that plane.
PAPERS = {
    "stock": {"kv": stock.StockKV},
    "continuum": {"kv": continuum.ContinuumKV,
                  "scheduling": continuum.ContinuumScheduling},
    "saga": {"kv": saga.SagaKV, "routing": saga.SagaRouting},
    "autellix": {"scheduling": autellix.AutellixScheduling},
    "infercept": {"kv": infercept.InferceptKV},
}

CITATIONS = {
    "stock": stock.PAPER, "continuum": continuum.PAPER, "saga": saga.PAPER,
    "autellix": autellix.PAPER, "infercept": infercept.PAPER,
}

#: axis flag string -> class. The strings are a published interface: the answer
#: key was recorded with them and the arena emits them. The classes behind them
#: moved; the strings did not.
VALUES = {
    "kv": {
        "cache-lru": stock.StockKV,
        "evict-always": stock.NoCacheKV,
        "ttl": generic.TTLRetention,
        "saga-ttl": saga.SagaKV,
        "saga-tool-ttl": saga.SagaToolTTL,
        "continuum": continuum.ContinuumKV,
        "min-waste": infercept.InferceptKV,
        "gate": gate.EvolvedRetention,
        "search-seed": search_seed.EvolvedRetention,
    },
    "scheduling": {
        "fcfs": stock.StockScheduling,
        "program-fcfs": generic.ProgramFCFSScheduling,
        "plas": autellix.AutellixScheduling,
        "continuum": continuum.ContinuumScheduling,
        "gate": gate.EvolvedScheduling,
        "search-seed": search_seed.EvolvedScheduling,
    },
    "routing": {
        # "rr" is what the CLI has always accepted and what the recorded runs
        # say; "round-robin" is the name in this table. Both resolve here so
        # neither spelling is wrong.
        "rr": stock.RoundRobinRouting,
        "round-robin": stock.RoundRobinRouting,
        "least-loaded": stock.LeastLoadedRouting,
        "session-affinity": saga.SagaRouting,
    },
}

#: Values the CLI accepts but that are not a class in VALUES: the search
#: candidate, and the clairvoyant probes that need the trace to construct.
SPECIAL_VALUES = {
    "kv": ("evolved", "oracle-ttl"),
    # autellix-mlfq is Algorithm 1 driven by policies.autellix_runtime through
    # serving/core/autellix_driver.py: a runtime plus an engine executor, not a
    # single SchedulingPolicy class, so it lives here rather than in VALUES.
    "scheduling": ("evolved", "oracle-srpt", "autellix-mlfq"),
    # Both need live instance state (queue contents, utilization), so they
    # are driven by serving/core/routing_drivers.py rather than being a
    # RoutingPolicy class the shared executor can call.
    "routing": ("saga-placement", "autellix-route"),
}


def choices_for(axis: str):
    """Every value `--<axis>` accepts. Derived, so adding a policy to VALUES
    is the only edit needed to make the flag accept it."""
    return sorted(VALUES[axis]) + list(SPECIAL_VALUES[axis])


def axes_of(paper: str):
    try:
        return PAPERS[paper]
    except KeyError:
        raise ValueError(
            f"unknown paper {paper!r}. Known: {sorted(PAPERS)}") from None


def build(paper: str, **kwargs):
    """Instantiate every plane this paper decides.

    A paper may build an axis itself (`make_kv`): InferCept reads a measured
    waste profile from disk where Continuum takes a float, and that is the
    paper's business rather than the registry's.
    """
    import importlib
    import inspect

    module = importlib.import_module(f"{__name__}.{paper}")
    out = {}
    for axis, cls in axes_of(paper).items():
        maker = getattr(module, f"make_{axis}", None)
        if maker is not None:
            out[axis] = maker(**kwargs)
            continue
        init = cls.__init__
        if init is object.__init__:
            out[axis] = cls()
            continue
        params = inspect.signature(init).parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            accepted = dict(kwargs)
        else:
            accepted = {k: v for k, v in kwargs.items() if k in params}
        out[axis] = cls(**accepted)
    return out


# ----------------------------------------------------------- legacy names
#
# The old axis modules exposed these by class name and callers look them up as
# attributes (`retention.ContinuumTTLRetention`). The classes moved to their
# paper modules; the names are kept here, bound to the SAME objects, because
# the answer key and the evolve shims were written against them. They are
# aliases in the only sense that costs nothing: one class, two names, no second
# definition.
CacheLRURetention = stock.StockKV
EvictAlwaysRetention = stock.NoCacheKV
TTLRetention = generic.TTLRetention
PressureTTLRetention = saga.SagaKV
ContinuumTTLRetention = continuum.ContinuumKV
MinWasteRetention = infercept.InferceptKV

FCFSScheduling = stock.StockScheduling
ProgramFCFSScheduling = generic.ProgramFCFSScheduling
PLASScheduling = autellix.AutellixScheduling
ContinuumScheduling = continuum.ContinuumScheduling

RoundRobinRouting = stock.RoundRobinRouting
LeastLoadedRouting = stock.LeastLoadedRouting
SessionAffinityRouting = saga.SagaRouting


# ------------------------------------------------- naming a policy by spec
#
# Both sides resolve a policy value the same way, so a policy someone writes
# runs on the simulator AND on hardware. It lived only on the simulator side
# until 2026-09-15: `serving/__main__.py` built its choices from
# `choices_for()` and accepted `module:Class`, while `bench/core/runner.py`
# carried a hand-written `choices=[...]` and `policy_driver` ended its
# if-chain in `raise ValueError`. So a policy added to VALUES, or named as
# module:Class, was runnable in simulation and could not be replayed on a
# GPU -- which is most of the point of having both.
#
# This lives here rather than in `serving/` because `bench/` must not import
# the simulator: the benchmark is meant to run for someone who has no
# simulator installed.

def load_spec(spec, *, flag="--policy"):
    """The class named by `module:Class`. Import errors name the flag."""
    import importlib
    if ":" not in (spec or ""):
        raise ValueError(f"{flag} {spec!r} is not module:Class")
    modname, _, clsname = spec.partition(":")
    try:
        mod = importlib.import_module(modname)
    except ImportError as e:
        raise ValueError(
            f"{flag} {spec!r}: cannot import {modname!r} ({e}). It must be on "
            f"sys.path -- pass --harness-root, or run from its directory."
        ) from None
    cls = getattr(mod, clsname, None)
    if cls is None:
        raise ValueError(
            f"{flag} {spec!r}: {modname!r} defines no {clsname!r}")
    return cls


def resolve(axis, value, cfg=None, *, flag=None):
    """A registry name or `module:Class` on `axis` -> a policy INSTANCE.

    `cfg` is a `policies.base.PolicyConfig`; a class that does not define
    `from_config` is constructed with no arguments, which is what a policy
    with no knobs wants.
    """
    flag = flag or f"--{axis}"
    cls = (VALUES[axis][value] if value in VALUES[axis]
           else load_spec(value, flag=flag))
    if cfg is not None and "from_config" in getattr(cls, "__dict__", {}):
        return cls.from_config(cfg)
    return cls()


def policy_value(axis):
    """argparse `type` for a policy flag: a registry name, or module:Class.

    `choices=` cannot express "one of these, or any importable class", and the
    open half is the point -- a policy someone writes should not need an edit
    to this repository to be runnable.
    """
    import argparse
    known = choices_for(axis)

    def parse(value):
        if value in known or ":" in value:
            return value
        raise argparse.ArgumentTypeError(
            f"{value!r} is not one of {known}, and is not module:Class "
            f"(e.g. mypolicy:MyRetention)")
    return parse
