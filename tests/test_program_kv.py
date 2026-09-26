"""The KV plane over a real radix tree.

These build an actual `RadixCache` and insert real token keys, because the whole
point of this plane is that the tree is the block store and the plane is a view
over it. A test that fabricated its own blocks would be testing a ledger that
does not exist in the running system.

The accounting tests matter most. A plane that evicts correctly but reports
occupancy differently from the other host produces plausible JCTs for policies
keyed on a number that means two different things -- invisible in every
aggregate.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from serving.core.program_kv import (                    # noqa: E402
    BlockClass, ProgramKVManager, Tier)
from serving.core.program_orchestrator import (          # noqa: E402
    Attribution, ProgramOrchestrator)


def build(capacity=1000, attribution=Attribution.SPLIT, block_size=16):
    orch = ProgramOrchestrator(attribution)
    kv = ProgramKVManager(instance=0, capacity_tokens=capacity,
                          orchestrator=orch, block_size=block_size)
    return orch, kv


def resident(orch, *names, now=0.0):
    for n in names:
        orch.on_turn_arrival(n, 0, now=now)
        orch.place(n, 0)


def insert(kv, program, tokens, lock=False):
    """Insert a token key for `program`, stamping ownership the way the engine
    does (`_current_owner` along the insert walk)."""
    kv.radix._current_owner = program
    try:
        kv.radix.insert(list(tokens))
    finally:
        kv.radix._current_owner = None
    if lock:
        for node in kv._nodes():
            if set(node.key) <= set(tokens):
                kv.radix.inc_lock_ref(node)
    return tokens


def owned(kv, program):
    return [n for n in kv._nodes() if program in n.owners]


# ------------------------------------------------------ the tree is the store

def test_the_plane_keeps_no_block_records_of_its_own():
    """Two records of one fact is the failure this design removes. The only
    per-node state here is tier, which the tree does not model."""
    _, kv = build()
    state = {k for k in vars(kv) if not k.startswith("__")}
    assert "blocks" not in state
    assert state & {"radix"}


def test_inserting_stamps_ownership():
    orch, kv = build()
    resident(orch, "a")
    insert(kv, "a", range(100))
    assert owned(kv, "a")
    assert kv.context_tokens("a") == 100


# --------------------------------------------------------- classification

def test_an_unreferenced_unpinned_node_is_cached():
    orch, kv = build()
    resident(orch, "a")
    insert(kv, "a", range(100))
    assert kv.occupancy()["cached"] == 100


def test_a_pinned_program_makes_its_nodes_pinned_not_cached():
    orch, kv = build()
    resident(orch, "a")
    insert(kv, "a", range(100))
    kv.pin("a", "r0", now=1.0)
    occ = kv.occupancy()
    assert occ["pinned"] == 100 and occ["cached"] == 0


def test_a_live_reference_is_locked_and_beats_a_pin():
    """A pinned program's running turn cannot be reclaimed even by breaking
    the pin, so it must classify as LOCKED."""
    orch, kv = build()
    resident(orch, "a")
    insert(kv, "a", range(100), lock=True)
    kv.pin("a", "r0", now=1.0)
    occ = kv.occupancy()
    assert occ["locked"] == 100 and occ["pinned"] == 0


# ------------------------------------------------------------- pressure

def test_pressure_excludes_cached_blocks():
    """The parity claim: vLLM's kv_cache_usage counts cached blocks as free,
    so a cache-heavy pool must not read as pressured here either."""
    orch, kv = build(capacity=1000)
    resident(orch, "a", "b")
    insert(kv, "a", range(0, 200), lock=True)
    insert(kv, "b", range(1000, 1700))
    assert kv.occupancy()["cached"] == 700
    assert kv.pressure() == pytest.approx(0.2)


def test_pinning_raises_pressure_because_it_removes_reclaimability():
    orch, kv = build(capacity=1000)
    resident(orch, "a", "b")
    insert(kv, "a", range(0, 200), lock=True)
    insert(kv, "b", range(1000, 1300))
    assert kv.pressure() == pytest.approx(0.2)
    kv.pin("b", "r0", now=1.0)
    assert kv.pressure() == pytest.approx(0.5)


def test_offloaded_nodes_stop_occupying_the_npu():
    orch, kv = build(capacity=1000)
    resident(orch, "a")
    insert(kv, "a", range(400))
    kv.pin("a", "r0", now=1.0)
    assert kv.pressure() == pytest.approx(0.4)
    assert kv.offload("a", Tier.CPU) == 400
    assert kv.pressure() == pytest.approx(0.0)


# ---------------------------------------------------------- attribution

def test_split_charges_a_shared_prefix_fractionally():
    orch, kv = build(attribution=Attribution.SPLIT)
    resident(orch, "a", "b")
    insert(kv, "a", range(400))
    insert(kv, "b", range(400))          # identical prefix -> shared nodes
    assert kv.charged("a") == 200 and kv.charged("b") == 200


def test_split_sums_to_what_is_occupied():
    """The property FULL lacks."""
    orch, kv = build(attribution=Attribution.SPLIT)
    resident(orch, "a", "b", "c")
    for p in ("a", "b", "c"):
        insert(kv, p, range(300))
    assert kv.charged("a") + kv.charged("b") + kv.charged("c") == 300


def test_full_double_counts_and_that_is_why_it_is_not_the_default():
    """The rule that produced an apparent 103% pool over-subscription."""
    orch, kv = build(attribution=Attribution.FULL)
    resident(orch, "a", "b")
    insert(kv, "a", range(400))
    insert(kv, "b", range(400))
    assert kv.charged("a") + kv.charged("b") == 800 > 400


def test_holder_charges_the_pin_holder_only():
    orch, kv = build(attribution=Attribution.HOLDER)
    resident(orch, "a", "b")
    insert(kv, "a", range(400))
    insert(kv, "b", range(400))
    kv.pin("a", "r0", now=1.0)
    assert kv.charged("a") == 400 and kv.charged("b") == 0


def test_context_tokens_ignores_sharing():
    """Recompute cost is the full context whoever else shares it."""
    orch, kv = build(attribution=Attribution.SPLIT)
    resident(orch, "a", "b")
    insert(kv, "a", range(400))
    insert(kv, "b", range(400))
    assert kv.context_tokens("a") == 400
    assert kv.charged("a") == 200


def test_sync_writes_both_quantities_to_the_orchestrator():
    orch, kv = build()
    resident(orch, "a", "b")
    insert(kv, "a", range(400))
    insert(kv, "b", range(400))
    kv.sync_footprints()
    p = orch.get("a")
    assert p.context_tokens == 400 and p.charged_tokens == 200


# ---------------------------------------------------------------- evict

def test_evict_frees_only_what_nothing_else_owns():
    """Taking a shared node would charge one program's decision to another."""
    orch, kv = build()
    resident(orch, "a", "b")
    insert(kv, "a", range(0, 100))        # a's own
    insert(kv, "a", range(500, 900))      # shared prefix
    insert(kv, "b", range(500, 900))
    kv.evict("a")
    assert kv.context_tokens("b") == 400
    assert kv.context_tokens("a") == 0


def test_evict_never_takes_a_locked_node():
    orch, kv = build()
    resident(orch, "a")
    insert(kv, "a", range(100), lock=True)
    assert kv.evict("a") == 0
    assert kv.context_tokens("a") == 100


# ----------------------------------------------------------- on_pressure

def test_policy_choice_is_honoured_and_pins_are_broken():
    orch, kv = build(capacity=1000)
    resident(orch, "a", "b")
    insert(kv, "a", range(0, 300))
    insert(kv, "b", range(1000, 1300))
    kv.pin("a", "r0", now=1.0)
    r = kv.on_pressure(200, now=2.0, policy=lambda progs, need, now: ["a"])
    assert r.declined is False
    assert ("a", "r0") in r.broke_pins
    assert kv.counters["pins_broken"] == 1


def test_declining_is_safe_and_the_valve_still_reclaims():
    orch, kv = build(capacity=1000)
    resident(orch, "a")
    insert(kv, "a", range(300))
    r = kv.on_pressure(200, now=2.0, policy=lambda *a: None)
    assert r.declined is True and r.tokens >= 200


def test_a_raising_policy_does_not_kill_the_engine():
    """Under policy search, candidates misbehave. That must cost a fallback,
    not a crash."""
    def boom(*a):
        raise ValueError("bad candidate")

    orch, kv = build(capacity=1000)
    resident(orch, "a")
    insert(kv, "a", range(300))
    r = kv.on_pressure(200, now=2.0, policy=boom)
    assert r.declined is True and r.tokens >= 200


def test_valve_takes_expired_pins_first():
    orch, kv = build(capacity=1000)
    resident(orch, "a", "b")
    insert(kv, "a", range(0, 300))
    insert(kv, "b", range(1000, 1300))
    kv.pin("a", "r0", now=0.0, deadline_ts=1.0)
    kv.on_pressure(200, now=5.0, policy=None)
    assert kv.counters["pins_expired"] == 1


def test_pressure_never_reclaims_locked_nodes():
    orch, kv = build(capacity=1000)
    resident(orch, "a")
    insert(kv, "a", range(900), lock=True)
    r = kv.on_pressure(500, now=2.0, policy=lambda progs, need, now: ["a"])
    assert r.tokens == 0
    assert kv.context_tokens("a") == 900


def test_counters_use_the_arena_vocabulary():
    from_arena = {"pins_created", "pins_released", "pins_expired",
                  "pins_broken", "retention_discards"}
    _, kv = build()
    assert from_arena <= set(kv.counters)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------------- reserve / commit split

def test_reserved_tokens_are_not_matchable_until_committed():
    """Publishing at decision time would let another program hit on a prefix
    that has not been computed yet, inflating cache hits."""
    orch, kv = build(capacity=10_000)
    resident(orch, "a")
    key = list(range(320))                   # a whole number of blocks
    assert kv.allocate("a", len(key)) == 320
    assert kv.probe(key) == 0                # reserved, invisible
    kv.commit("a", key)
    assert kv.probe(key) == 320              # published


def test_a_reservation_occupies_the_pool():
    """Otherwise the pool looks emptier than it is for exactly as long as a
    step is in flight -- which is when admission decisions are made."""
    orch, kv = build(capacity=1000)
    resident(orch, "a")
    kv.allocate("a", 400)
    assert kv.occupancy()["free"] == 600
    assert kv.inflight_tokens() == 400


def test_a_reservation_that_does_not_fit_is_refused():
    orch, kv = build(capacity=500)
    resident(orch, "a", "b")
    assert kv.allocate("a", 400) == 400
    assert kv.allocate("b", 400) is None


def test_commit_clears_the_reservation_so_it_is_not_counted_twice():
    orch, kv = build(capacity=1000)
    resident(orch, "a")
    key = list(range(400))
    kv.allocate("a", len(key))
    kv.commit("a", key)
    assert kv.inflight_tokens() == 0
    assert kv.occupancy()["locked"] == 400
    assert kv.occupancy()["free"] == 600     # not 200


def test_dropping_a_reservation_returns_the_space():
    orch, kv = build(capacity=1000)
    resident(orch, "a")
    kv.allocate("a", 400)
    assert kv.drop_reservation("a") == 400
    assert kv.occupancy()["free"] == 1000


# ------------------------------------------------------ block granularity

def test_a_partial_block_is_not_a_hit():
    """vLLM caches in blocks. A prefix matching eleven tokens of a sixteen-token
    block is not a hit and none of that block is reusable -- matching at token
    granularity hands out hits the engine would never give, about half a block
    per turn."""
    orch, kv = build(capacity=10_000)
    resident(orch, "a")
    key = list(range(300))                   # 18 whole blocks + 12 tokens
    kv.commit("a", key)
    assert kv.probe(key) == 288              # 18 * 16


def test_the_block_size_is_the_configured_one():
    orch, kv = build(capacity=10_000, block_size=32)
    resident(orch, "a")
    key = list(range(100))
    kv.commit("a", key)
    assert kv.probe(key) == 96               # 3 * 32


# ------------------------------------------------------- lock balance

def test_a_turn_holds_exactly_one_lock_however_long_it_decodes():
    """`commit` runs once per decode step. Taking a fresh lock each time
    without dropping the previous one leaves a twenty-token turn holding
    nineteen locks forever, and a locked node cannot be reclaimed by pressure
    or by any policy -- the pool fills with the residue of finished work."""
    orch, kv = build(capacity=10_000)
    resident(orch, "a")
    for n in range(320, 480, 16):                 # a turn extending its key
        kv.commit("a", list(range(n)), owner="a:0")
    assert kv.occupancy()["locked"] > 0
    kv.release(list(range(464)), owner="a:0")
    assert kv.occupancy()["locked"] == 0, "locks leaked across decode steps"


def test_two_turns_of_one_program_are_two_holders():
    """A fan-out puts two turns of the same program in flight at once. Sharing
    one lock slot would make the second commit release the first turn's blocks
    while it is still computing on them."""
    orch, kv = build(capacity=10_000)
    resident(orch, "a")
    kv.commit("a", list(range(100, 200)), owner="a:0")
    kv.commit("a", list(range(900, 1000)), owner="a:1")
    kv.release(list(range(100, 200)), owner="a:0")
    assert kv.occupancy()["locked"] > 0, "releasing one turn freed the other's"
    kv.release(list(range(900, 1000)), owner="a:1")
    assert kv.occupancy()["locked"] == 0


def test_release_without_an_owner_still_works():
    """The older call shape, kept: callers that never took a tracked lock."""
    orch, kv = build(capacity=10_000)
    resident(orch, "a")
    kv.commit("a", list(range(320)))
    kv.release(list(range(320)))
    assert kv.occupancy()["locked"] == 0


# ------------------------------------------------ the pin must be real

def test_pressure_takes_the_unpinned_blocks_not_the_pinned_ones():
    """The tree's own LRU skips lock_ref>0, which is a LIVE reference. A policy
    pin is not one -- it lives in the orchestrator and the tree has never heard
    of it. Handing eviction to the tree meant pressure took the PINNED
    program's blocks and left the unpinned one's, because the pinned program
    had committed first and was the older entry."""
    orch, kv = build(capacity=1000)
    for pid, rng in (("keepme", range(0, 320)), ("dropme", range(1000, 1320))):
        orch.on_turn_arrival(pid, 0, now=0.0)
        orch.place(pid, 0)
        kv.commit(pid, list(rng), owner=pid + ":0")
        kv.release(list(rng), owner=pid + ":0")
    kv.pin("keepme", "keepme:0", now=0.0, deadline_ts=1e9)

    kv.on_pressure(320, now=1.0)
    assert kv.probe(list(range(0, 320))) == 320, "the pin was ignored"
    assert kv.probe(list(range(1000, 1320))) == 0, "the unpinned copy survived"


def test_an_expired_pin_is_reclaimed():
    """Protection with a deadline is protection until the deadline."""
    orch, kv = build(capacity=1000)
    for pid, rng in (("old", range(0, 320)), ("new", range(1000, 1320))):
        orch.on_turn_arrival(pid, 0, now=0.0)
        orch.place(pid, 0)
        kv.commit(pid, list(rng), owner=pid + ":0")
        kv.release(list(rng), owner=pid + ":0")
    kv.pin("old", "old:0", now=0.0, deadline_ts=5.0)
    kv.on_pressure(600, now=99.0)                 # long past the deadline
    assert kv.counters["pins_expired"] >= 1


def test_a_pin_with_nothing_else_to_take_is_not_silently_broken():
    """Breaking a pin is a decision someone makes and records, never the
    oldest thing to hand."""
    orch, kv = build(capacity=1000)
    orch.on_turn_arrival("p", 0, now=0.0)
    orch.place("p", 0)
    kv.commit("p", list(range(320)), owner="p:0")
    kv.release(list(range(320)), owner="p:0")
    kv.pin("p", "p:0", now=0.0, deadline_ts=1e9)
    r = kv.on_pressure(320, now=1.0)
    assert r.tokens == 0 and r.broke_pins == ()
    assert kv.probe(list(range(320))) == 320


# ------------------------------------------------------- occupancy cost

def test_the_fast_path_agrees_with_the_walk():
    """The fast path reads the tree's incremental counters instead of walking
    every node. If the two ever disagree, every pressure decision is made from
    a different number depending on whether a pin happens to exist."""
    orch, kv = build(capacity=100_000)
    for pid, rng in (("a", range(0, 3200)), ("b", range(9000, 12200)),
                     ("c", range(50000, 50320))):
        orch.on_turn_arrival(pid, 0, now=0.0)
        orch.place(pid, 0)
        kv.commit(pid, list(rng), owner=pid + ":0")
    kv.release(list(range(0, 3200)), owner="a:0")

    fast = kv.occupancy()
    # force the walk by pinning something, then unpin and compare the parts
    kv.pin("a", "a:0", now=0.0, deadline_ts=1e9)
    kv._invalidate()
    walked = kv.occupancy()
    assert walked["locked"] == fast["locked"]
    assert walked["pinned"] + walked["cached"] == fast["pinned"] + fast["cached"]


def test_occupancy_does_not_walk_the_tree_when_nothing_is_pinned():
    """The walk is invalidated on every allocate, commit and release, so at a
    board-cell pool it dominates the run."""
    orch, kv = build(capacity=100_000)
    orch.on_turn_arrival("a", 0, now=0.0)
    orch.place("a", 0)
    kv.commit("a", list(range(0, 6400)), owner="a:0")

    calls = {"n": 0}
    real = kv._nodes
    def counting():
        calls["n"] += 1
        return real()
    kv._nodes = counting
    kv._invalidate()
    kv.occupancy()
    assert calls["n"] == 0, "occupancy walked the tree with no pins present"


def test_the_walk_still_runs_once_a_pin_exists():
    """A pin is the one thing the tree cannot see, so its share has to be
    counted the slow way -- correctness first, speed only where it is free."""
    orch, kv = build(capacity=100_000)
    orch.on_turn_arrival("a", 0, now=0.0)
    orch.place("a", 0)
    kv.commit("a", list(range(0, 6400)), owner="a:0")
    kv.release(list(range(0, 6400)), owner="a:0")
    kv.pin("a", "a:0", now=0.0, deadline_ts=1e9)

    calls = {"n": 0}
    real = kv._nodes
    def counting():
        calls["n"] += 1
        return real()
    kv._nodes = counting
    kv._invalidate()
    occ = kv.occupancy()
    assert calls["n"] == 1
    assert occ["pinned"] > 0 and occ["cached"] == 0


def test_pressure_does_not_take_a_prefix_that_is_held():
    """A request credited a prefix hit skips recomputing those tokens, so the
    blocks must survive until it commits. They are CACHED, and a request that
    was just preempted owns the least recently used chain in the tree -- so the
    reclaim pass run on its own behalf is the one most likely to take them.
    `commit` would then republish the whole context on a reservation sized for
    one chunk, and the pool goes over capacity by the difference.

    vLLM holds them the same way: `allocate_slots` takes the computed blocks
    before allocating new ones. So does the old plane (scheduler.py:575)."""
    orch, kv = build(capacity=10_000)
    resident(orch, "a")
    kv.commit("a", list(range(640)), owner="a:0")
    kv.release(list(range(640)), owner="a:0")
    assert kv.occupancy()["cached"] == 640

    kv.hold_prefix("a:1", list(range(640)))          # credited to a's next turn
    kv.on_pressure(10_000, now=1.0)
    assert kv.probe(list(range(640))) == 640, \
        "pressure took a prefix that was held"

    # and the hold is releasable, so this is not a leak
    kv.drop_hold("a:1")
    kv.on_pressure(640, now=2.0)
    assert kv.probe(list(range(640))) == 0


def test_pressure_still_takes_blocks_with_no_live_reservation():
    """The guard must not make everything unreclaimable."""
    orch, kv = build(capacity=10_000)
    resident(orch, "a")
    kv.commit("a", list(range(640)), owner="a:0")
    kv.release(list(range(640)), owner="a:0")
    kv.on_pressure(640, now=1.0)
    assert kv.probe(list(range(640))) == 0


def test_the_one_pass_footprint_walk_agrees_with_the_per_program_definitions():
    """`sync_footprints` computes every program's context, charge and tier
    split in a SINGLE pass over the tree, because the readable spelling -- the
    three per-program methods, per program -- is three full walks per resident
    program on a function that runs every step, which is most of a cell's
    runtime rather than a part of it.

    The per-program methods remain the definition. This pins the fast path to
    them, under every attribution rule and with a prefix genuinely shared, so
    the two cannot drift.
    """
    shared = list(range(320))
    for attribution in Attribution:
        orch, kv = build(capacity=10_000, attribution=attribution)
        resident(orch, "a", "b", "c")
        # a and b share a 320-token prefix; c is disjoint. Each then diverges,
        # so no program's footprint is a prefix of another's.
        kv.commit("a", shared + list(range(1000, 1160)), owner="a:0")
        kv.commit("b", shared + list(range(2000, 2080)), owner="b:0")
        kv.commit("c", list(range(3000, 3480)), owner="c:0")

        kv.sync_footprints()
        for pid in ("a", "b", "c"):
            state = orch.get(pid)
            assert state.context_tokens == kv.context_tokens(pid), (
                f"{attribution.name}: context for {pid}")
            assert state.charged_tokens == kv.charged(pid), (
                f"{attribution.name}: charge for {pid}")
            assert state.tier_tokens == kv._tier_tokens(pid), (
                f"{attribution.name}: tiers for {pid}")

        # and the sharing is real, or this proves nothing
        assert kv.context_tokens("a") > kv.charged("a") or (
            attribution is Attribution.FULL), (
            f"{attribution.name}: the shared prefix was not actually shared")
