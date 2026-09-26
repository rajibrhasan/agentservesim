"""Program-aware placement.

Every cell we run today has one instance, so this plane cannot be exercised by
any cell — the same reason the arena reports `routing` as unproven rather than
absent. These tests are therefore the only thing standing behind it until a
multi-instance cell exists, which makes the invariant checks the important ones
rather than the load-balancing ones.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from serving.core.program_orchestrator import ProgramOrchestrator   # noqa: E402
from serving.core.program_router import (                           # noqa: E402
    InstanceView, ProgramRouter)


def chain(*specs):
    """Build a path graph the way the loader does: turn i parented on i-1 with
    that turn's tool gap as the edge delay. `specs` are (input, output, gap_ns).
    """
    from serving.core.program_orchestrator import PlannedTurn
    return tuple(
        PlannedTurn(node_id=i, input_toks=inp, output_toks=out,
                    parents=() if i == 0 else ((i - 1, specs[i - 1][2]),))
        for i, (inp, out, _) in enumerate(specs))


def build(n=2, policy="AFFINITY", **kw):
    """Default AFFINITY here because most of these tests are about program
    locality. The router's own default is LOAD, matching the CLI and the old
    router -- affinity is the addition, and it is opt-in so it can be measured
    against the thing it replaces."""
    orch = ProgramOrchestrator()
    return orch, ProgramRouter(orch, n_instances=n, policy=policy, **kw)


def arrive(orch, pid, now=0.0, turn=0):
    orch.on_turn_arrival(pid, turn, now=now)


# ------------------------------------------------------------- affinity

def test_a_program_returns_to_the_instance_holding_its_context():
    """The expensive thing in an agent workload is the prefix; a turn placed
    away from it pays the whole prompt again."""
    orch, r = build()
    arrive(orch, "a")
    first = r.route("a", now=0.0)
    assert r.route("a", now=1.0) == first
    assert r.counters["affinity_hits"] == 1


def test_a_cold_program_is_placed_not_counted_as_affinity():
    orch, r = build()
    arrive(orch, "a")
    r.route("a", now=0.0)
    assert r.counters["cold_placements"] == 1
    assert r.counters["affinity_hits"] == 0


def test_affinity_holds_even_when_the_home_instance_is_busier():
    """Load imbalance is cheaper to tolerate than a lost prefix, which is why
    SAGA pins a session rather than balancing it."""
    orch, r = build()
    arrive(orch, "a")
    home = r.route("a", now=0.0)
    for i in range(5):                       # pile other programs onto home
        arrive(orch, f"x{i}")
        orch.place(f"x{i}", home)
    assert r.route("a", now=2.0) == home


def test_affinity_yields_above_the_capacity_limit_and_is_counted():
    """A break is a real cost, not a routine choice: the program loses its
    cached prefix. Counting it is what stops it looking like a cache miss."""
    class FakeKV:
        def __init__(self, p): self._p = p
        def pressure(self): return self._p

    class FakeSched:
        def __init__(self, p):
            self.kv, self.running, self.waiting = FakeKV(p), [], []
            self.max_num_seqs = 128

    orch = ProgramOrchestrator()
    r = ProgramRouter(orch, n_instances=2, policy="AFFINITY",
                      schedulers=[FakeSched(0.95), FakeSched(0.10)],
                      capacity_limit=0.8)
    arrive(orch, "a")
    orch.place("a", 0)                        # home is instance 0, at 95%
    assert r.route("a", now=1.0) == 1
    assert r.counters["affinity_breaks"] == 1


# ------------------------------------------------------- the invariant

def test_routing_keeps_the_single_live_instance_invariant():
    """program_kv assumes a program's live context is on one instance. Routing
    is the only thing that can move a program, so it is where that holds."""
    orch, r = build()
    arrive(orch, "a")
    r.route("a", now=0.0)
    orch.set_footprint("a", context_tokens=900, charged_tokens=450)
    r._commit("a", 1, orch.get("a"))          # forced move
    p = orch.get("a")
    assert p.live_instance == 1
    assert p.context_tokens == 0              # stale footprint dropped
    assert p.instance_history == frozenset({0, 1})


def test_placement_is_always_committed_to_the_orchestrator():
    """A router that chose without committing would leave the KV plane charging
    a program for context on an instance it has left."""
    orch, r = build()
    arrive(orch, "a")
    chosen = r.route("a", now=0.0)
    assert orch.get("a").live_instance == chosen


# --------------------------------------------------------- distribution

def test_cold_programs_spread_across_instances():
    orch, r = build(n=3)
    for i in range(6):
        arrive(orch, f"p{i}")
        r.route(f"p{i}", now=0.0)
    counts = {}
    for p in orch.all():
        counts[p.live_instance] = counts.get(p.live_instance, 0) + 1
    assert set(counts) == {0, 1, 2} and max(counts.values()) == 2


def test_a_single_instance_cluster_is_degenerate_but_legal():
    """Every cell today is one instance, which is why this plane is unproven
    rather than wrong."""
    orch, r = build(n=1)
    arrive(orch, "a")
    assert r.route("a", now=0.0) == 0


# -------------------------------------------------------------- policy

def test_policy_choice_is_honoured_and_counted():
    orch, r = build(route_fn=lambda st, views, now: 1)
    arrive(orch, "a")
    assert r.route("a", now=0.0) == 1
    assert r.counters["policy_overrides"] == 1


def test_a_declining_policy_falls_back_to_affinity():
    orch, r = build(route_fn=lambda st, views, now: None)
    arrive(orch, "a")
    home = r.route("a", now=0.0)
    assert r.route("a", now=1.0) == home
    assert r.counters["policy_overrides"] == 0


def test_a_raising_policy_does_not_stall_routing():
    def boom(*a):
        raise ValueError("bad candidate")

    orch, r = build(route_fn=boom)
    arrive(orch, "a")
    assert r.route("a", now=0.0) in (0, 1)


def test_an_out_of_range_policy_choice_is_ignored():
    """A searched policy will return nonsense; it must not index off the end."""
    orch, r = build(route_fn=lambda st, views, now: 99)
    arrive(orch, "a")
    assert r.route("a", now=0.0) in (0, 1)
    assert r.counters["policy_overrides"] == 0


def test_a_policy_sees_load_but_not_engine_internals():
    """What the real gateway cannot observe, a policy must not read -- a
    decision it could not reproduce there is not one the benchmark can score."""
    seen = {}
    orch, r = build(route_fn=lambda st, views, now: seen.setdefault("v", views) and None)
    arrive(orch, "a")
    r.route("a", now=0.0)
    assert all(isinstance(v, InstanceView) for v in seen["v"])
    assert set(vars(seen["v"][0])) == {"instance", "programs", "running",
                                       "waiting", "pressure", "capacity"}


# ------------------------------------------------------ routing policies

def test_the_default_policy_is_load_not_affinity():
    """`--planes program` must not silently change placement: LOAD is the CLI
    default and the old router's, so it is this router's too. Affinity is the
    addition, and being opt-in is what makes it measurable."""
    orch = ProgramOrchestrator()
    assert ProgramRouter(orch, n_instances=2).policy == "LOAD"


def test_round_robin_cycles():
    orch = ProgramOrchestrator()
    r = ProgramRouter(orch, n_instances=3, policy="RR")
    for name in ("a", "b", "c", "d"):
        orch.on_turn_arrival(name, 0, now=0.0)
    assert [r.route(n, 0.0) for n in ("a", "b", "c", "d")] == [0, 1, 2, 0]


def test_random_is_seeded_so_a_run_is_reproducible():
    """An unseeded router would make two runs of one cell differ for reasons
    no counter records."""
    def rt(seed):
        orch = ProgramOrchestrator()
        r = ProgramRouter(orch, n_instances=4, policy="RAND", seed=seed)
        for i in range(8):
            orch.on_turn_arrival(f"p{i}", 0, now=0.0)
        return [r.route(f"p{i}", 0.0) for i in range(8)]

    assert rt(7) == rt(7)
    assert set(rt(7)) <= {0, 1, 2, 3}


def test_load_uses_the_engine_weighting():
    """waiting counts four times a running turn, normalised by capacity --
    copied from the old router rather than improved, so the flag means the same
    thing on both paths."""
    class FakeKV:
        def pressure(self): return 0.0

    class S:
        def __init__(self, w, r, cap=128):
            self.waiting = list(range(w))
            self.running = list(range(r))
            self.max_num_seqs = cap
            self.kv = FakeKV()

    orch = ProgramOrchestrator()
    # instance 0: 2 waiting -> 8;  instance 1: 7 running -> 7. 1 wins.
    r = ProgramRouter(orch, n_instances=2, policy="LOAD",
                      schedulers=[S(2, 0), S(0, 7)])
    orch.on_turn_arrival("a", 0, now=0.0)
    assert r.route("a", 0.0) == 1


def test_affinity_is_not_load():
    """The two disagree, which is the whole reason affinity is a separate
    policy rather than a tweak to LOAD."""
    class FakeKV:
        def pressure(self): return 0.0

    class S:
        def __init__(self, w):
            self.waiting, self.running = list(range(w)), []
            self.max_num_seqs, self.kv = 128, FakeKV()

    for policy, expected in (("LOAD", 1), ("AFFINITY", 0)):
        orch = ProgramOrchestrator()
        r = ProgramRouter(orch, n_instances=2, policy=policy,
                          schedulers=[S(9), S(0)])
        orch.on_turn_arrival("a", 0, now=0.0)
        orch.place("a", 0)                    # context is on the busy instance
        assert r.route("a", 1.0) == expected


# ------------------------------------------------- prefill/decode split

def test_narrowing_returns_an_instance_id_not_a_position():
    """The decode half of a P/D split is instances {2,3}, not positions {0,1}.
    A router that returned a position would place every transferred request on
    the prefill instances and record the lie in the orchestrator."""
    orch, r = build(n=4, policy="LOAD")
    orch.define_program("a", chain((10, 20, 0)), arrival_ts=0.0)
    for _ in range(4):
        assert r.route("a", 0.0, only=[2, 3]) in (2, 3)


def test_narrowing_holds_for_every_policy():
    for policy in ("LOAD", "RR", "RAND", "AFFINITY"):
        orch, r = build(n=4, policy=policy)
        orch.define_program("a", chain((10, 20, 0)), arrival_ts=0.0)
        orch.place("a", 0)                      # home is OUTSIDE the subset
        chosen = {r.route("a", 0.0, only=[2, 3]) for _ in range(6)}
        assert chosen <= {2, 3}, policy


def test_a_policy_may_not_escape_the_narrowing():
    orch, r = build(n=4, policy="LOAD", route_fn=lambda s, v, n: 0)
    orch.define_program("a", chain((10, 20, 0)), arrival_ts=0.0)
    assert r.route("a", 0.0, only=[2, 3]) in (2, 3)
    assert r.counters["policy_overrides"] == 0
