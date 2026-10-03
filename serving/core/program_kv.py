
from __future__ import annotations

import heapq

import enum
from dataclasses import dataclass
from typing import (Callable, Dict, Iterable, List, Optional, Sequence,
                    Set, Tuple)

from .program_orchestrator import Attribution, ProgramOrchestrator
from .radix_tree import RadixCache, TreeNode


class Tier(enum.IntEnum):
    """Where a block physically sits. Values match the engine's Device enum."""

    NPU = 1
    CPU = 2
    CXL = 3


class BlockClass(enum.Enum):
    LOCKED = "locked"
    PINNED = "pinned"
    CACHED = "cached"


@dataclass(frozen=True)
class Reclaim:
    """What a pressure round took back, and from whom."""

    tokens: int
    broke_pins: Tuple[Tuple[str, str], ...] = ()    # (program_id, request_id)
    declined: bool = False                          # policy chose nothing

    @property
    def enough(self) -> bool:
        return self.tokens > 0


class ProgramKVManager:
    """The KV plane for ONE instance.

    Per-instance because pressure is per-instance. A program is cluster-scoped,
    which is why the orchestrator holds the invariant that a program's live
    context sits on exactly one instance -- without it a footprint is a vector
    and every per-instance pressure decision about a program is ill-posed.
    """

    def __init__(self, instance: int, capacity_tokens: int,
                 orchestrator: ProgramOrchestrator,
                 radix: Optional[RadixCache] = None,
                 attribution: Optional[Attribution] = None,
                 block_size: int = 16) -> None:
        self.instance = instance
        self.capacity_tokens = capacity_tokens
        self.block_size = max(1, int(block_size))
        self.orch = orchestrator
        # `page_size = block_size`, because a KV block is the unit vLLM caches
        # in: a prefix that matches eleven tokens of a sixteen-token block is
        # not a hit, and no part of that block is reusable. At page_size 1 the
        # tree matches and publishes partial blocks, which costs nothing in
        # memory but hands out hits the engine would not -- 1,096 tokens of
        # them on bfcl20_kv22_j0.06, about half a block per turn, which is
        # exactly the shape of a rounding that is not being done.
        self.radix = radix if radix is not None else RadixCache(
            node_id=0, device="NPU", page_size=self.block_size,
            capacity=capacity_tokens, kv_size=1, instance_id=instance)
        self.attribution = attribution or orchestrator.attribution
        self.counters: Dict[str, int] = {
            "pins_created": 0, "pins_released": 0, "pins_expired": 0,
            "pins_broken": 0, "retention_discards": 0, "offloads": 0,
            "overflow_reclaims": 0,
        }
        #: Tier placement is not something the tree models, so it is held here
        #: keyed by node id -- a property OF a node, not a copy of one. Absent
        #: means NPU.
        self._tier: Dict[int, Tier] = {}
        #: TURN -> tokens scheduled but not yet committed to the tree.
        #:
        #: Keyed by the turn, not the program. A reservation belongs to the
        #: step that made it: keyed by program, one turn's outstanding
        #: reservation was netted out of the next turn's `need` (correct, to
        #: avoid double-reserving the same tokens) and then released whole by
        #: whichever turn committed first -- so a commit grew the tree by
        #: 36,359 tokens having been charged for 19,980. See `allocate`.
        self._inflight: Dict[str, int] = {}
        #: turn -> the tree NODE that turn holds one lock on.
        #:
        #: The node, not the key. A key does not stably name a node: the tree
        #: grows along that path, so `match_prefix(key)` resolves deeper than it
        #: did when the lock was taken, and the decrement lands on a different
        #: node from the increment -- leaking a lock on one and driving another
        #: negative, once per decode step.
        #:
        #: A node reference survives the thing keys were meant to protect
        #: against: `_split_node` keeps the child object and copies `lock_ref`
        #: onto the new parent, so a held node stays held and
        #: `dec_lock_ref` still walks the right chain.
        self._lock_holder: Dict[str, object] = {}
        # `occupancy` walks the whole tree, and the scheduler asks for it several times
        # per step (can_fit, pressure, admit). The tree only changes on
        # commit/evict/offload/pressure, so the walk is cached and invalidated on those
        # rather than repeated.
        self._occ_cache = None

    # ------------------------------------------------------------- walking

    def _nodes(self) -> Iterable[TreeNode]:
        """Every node carrying tokens. The root holds none."""
        stack = [self.radix.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is not self.radix.root_node and node.key:
                yield node

    def _pinned_programs(self) -> Set[str]:
        """Read from the orchestrator, never cached. It owns pins."""
        return {p.program_id for p in self.orch.pinned()}

    @staticmethod
    def classify(node: TreeNode, pinned_programs: Set[str]) -> BlockClass:
        if node.lock_ref > 0:
            return BlockClass.LOCKED
        if node.owners & pinned_programs:
            return BlockClass.PINNED
        return BlockClass.CACHED

    def tier_of(self, node: TreeNode) -> Tier:
        return self._tier.get(node.id, Tier.NPU)

    # ---------------------------------------------------------- accounting

    def _invalidate(self) -> None:
        self._occ_cache = None

    def occupancy(self) -> Dict[str, int]:
        """Tokens by class on the NPU tier, plus what is left.

        LOCKED comes from the tree's own counter. Only the pinned share of the
        evictable half is walked, because the tree cannot distinguish a policy
        pin from ordinary cache -- that distinction is this plane's whole job.
        """
        if self._occ_cache is not None:
            return self._occ_cache
        pinned_programs = self._pinned_programs()

        if not pinned_programs and not self._tier:
            # Nothing is pinned and nothing is offloaded, so the three classes
            # collapse to the two the TREE already counts incrementally
            # (inc_lock_ref/dec_lock_ref and insert/delete keep
            # protected_size_/evictable_size_ exact -- asserted in
            # tests/test_program_kv.py). Walking every node to re-derive them
            # is the whole cost of this call, it is invalidated on every
            # allocate, commit and release, and at a 114k-token pool that is
            # thousands of nodes several times per scheduling step. It is why a
            # board cell took 64 minutes to reach a deadlock the old planes
            # cleared in 40.
            locked = self.radix.protected_size()
            cached = self.radix.evictable_size()
            inflight = self.inflight_tokens()
            used = locked + cached + inflight
            self._occ_cache = {"locked": locked + inflight, "pinned": 0,
                               "cached": cached,
                               "free": max(0, self.capacity_tokens - used)}
            return self._occ_cache

        locked = pinned = cached = 0
        for node in self._nodes():
            if self.tier_of(node) is not Tier.NPU:
                continue
            n = len(node.key)
            klass = self.classify(node, pinned_programs)
            if klass is BlockClass.LOCKED:
                locked += n
            elif klass is BlockClass.PINNED:
                pinned += n
            else:
                cached += n
        # Reserved-but-uncommitted tokens occupy the pool even though no tree
        # node carries them yet. Omitting them makes the pool look emptier than
        # it is for exactly as long as a step is in flight, which is when
        # admission decisions are being made.
        inflight = self.inflight_tokens()
        used = locked + pinned + cached + inflight
        self._occ_cache = {"locked": locked + inflight, "pinned": pinned,
                           "cached": cached,
                           "free": max(0, self.capacity_tokens - used)}
        return self._occ_cache

    def pressure(self) -> float:
        """Fraction of the pool that cannot be taken without breaking a pin.

        This -- not total occupancy -- is comparable with vLLM's
        `kv_cache_usage`, which treats cached blocks as free. A policy keyed on
        utilization must be handed this on both hosts or it is not one policy.
        """
        occ = self.occupancy()
        return (occ["locked"] + occ["pinned"]) / max(1, self.capacity_tokens)

    def charged(self, program_id: str) -> int:
        """This program's footprint under the active attribution rule."""
        pinned_programs = self._pinned_programs()
        total = 0
        for node in self._nodes():
            if program_id not in node.owners:
                continue
            n = len(node.key)
            if self.attribution is Attribution.FULL:
                total += n
            elif self.attribution is Attribution.SPLIT:
                total += n // max(1, len(node.owners))
            else:                                          # HOLDER
                holders = node.owners & pinned_programs
                if program_id in holders or (not holders
                                             and node.owners == {program_id}):
                    total += n
        return total

    def context_tokens(self, program_id: str) -> int:
        """Full context regardless of sharing -- the recompute cost if dropped.
        Distinct from `charged`, which is a pressure quantity."""
        return sum(len(node.key) for node in self._nodes()
                   if program_id in node.owners)

    def _tier_tokens(self, program_id: str) -> Tuple[int, ...]:
        per = {t: 0 for t in Tier}
        for node in self._nodes():
            if program_id in node.owners:
                per[self.tier_of(node)] += len(node.key)
        return tuple(per[t] for t in sorted(Tier))

    def sync_footprints(self) -> None:
        """Write every resident program's footprint into the orchestrator."""
        programs = list(self.orch.on_instance(self.instance))
        if not programs:
            return
        ids = {p.program_id for p in programs}
        pinned = self._pinned_programs()
        tiers = sorted(Tier)
        ctx = {i: 0 for i in ids}
        charged = {i: 0 for i in ids}
        per_tier = {i: {t: 0 for t in Tier} for i in ids}
        for node in self._nodes():
            owners = node.owners & ids
            if not owners:
                continue
            n = len(node.key)
            tier = self.tier_of(node)
            split = n // max(1, len(node.owners))
            holders = node.owners & pinned
            for pid in owners:
                ctx[pid] += n
                per_tier[pid][tier] += n
                if self.attribution is Attribution.FULL:
                    charged[pid] += n
                elif self.attribution is Attribution.SPLIT:
                    charged[pid] += split
                elif pid in holders or (not holders
                                        and node.owners == {pid}):
                    charged[pid] += n
        for p in programs:
            pid = p.program_id
            self.orch.set_footprint(
                pid,
                context_tokens=ctx[pid],
                charged_tokens=charged[pid],
                tier_tokens=tuple(per_tier[pid][t] for t in tiers))

    # ------------------------------------------------------------- acquire

    def probe(self, token_ids: Sequence[int]) -> int:
        """How much of this prefix is already cached. The prefix-cache hit.

        Read before allocating, because cached tokens are neither recomputed nor
        re-charged -- that is the entire value of the cache, and a scheduler
        that ignored it would make every cache-rescuing policy look worthless.
        """
        if not token_ids:
            return 0
        return self.radix.match_prefix(list(token_ids)).hit_length

    def headroom(self) -> int:
        """Tokens free, SIGNED. Negative means the pool is over-subscribed.

        `occupancy()["free"]` clamps at zero, which is right for reporting and
        wrong for deciding how much to reclaim: clamped, "exactly full" and
        "five thousand over" are the same number, and a reclaim sized against
        it frees a block, changes nothing, and is asked again forever.
        """
        occ = self.occupancy()
        used = occ["locked"] + occ["pinned"] + occ["cached"]
        return self.capacity_tokens - used

    def reclaim_target(self, n_tokens: int) -> int:
        """How much must actually come back for `n_tokens` to fit.

        The shortfall of the REQUEST plus the deficit of the POOL. Asking only
        for the former is what stalled a board cell with 97,925 reclaimable
        tokens in front of it: the request needed 10, the pool was 5,371 over,
        so freeing 10 left `free` clamped at 0 and the allocation failed again.
        """
        return max(0, n_tokens - self.headroom())

    def can_fit(self, n_tokens: int) -> bool:
        """Is there room without taking anything from anyone?

        Free means genuinely unused. Cached blocks are NOT counted here even
        though vLLM would evict them freely -- reclaiming them is `on_pressure`'s
        job and it is a decision with a cost, so the caller asks for it rather
        than having it happen silently inside an allocation.
        """
        return self.occupancy()["free"] >= max(0, n_tokens)

    def allocate(self, program_id: str, n_tokens: int, now: float = 0.0,
                 owner: Optional[str] = None) -> Optional[int]:
        """RESERVE room for the `n_tokens` this step will compute. Phase one.

        Takes a SIZE, not a token key. It used to take the key and charge
        `len(key) - probe(key)`, which cost a full key build and a tree walk on
        every call -- 384us at the mean turn, 1.6ms at the largest, four to six
        times per request per step -- and made the reservation depend on a
        prefix nothing was holding, so pressure could invalidate it between
        reserving and committing.

        `memory_model.is_avail` compares two integers. This is the same
        quantity: the caller already knows how many tokens the step computes,
        because that is what it asked the scheduler for.

        Returns the tokens newly reserved, or None if they do not fit --
        failure is a return value, not an exception, because the caller's
        response is to reclaim and retry, not to unwind.
        """
        if n_tokens <= 0:
            return 0
        if not self.can_fit(n_tokens):
            return None
        who = owner if owner is not None else program_id
        self._inflight[who] = self._inflight.get(who, 0) + n_tokens
        self._invalidate()
        return n_tokens

    def commit(self, program_id: str, token_ids: Sequence[int],
               owner: Optional[str] = None) -> int:
        """Publish computed tokens to the prefix tree and lock them. Phase two."""
        full = list(token_ids)
        if not full:
            return 0
        # Only WHOLE blocks are publishable. vLLM hashes a block when it fills
        # (block_pool.cache_full_blocks); a partially-filled block is owned by
        # its request and matchable by nobody. Inserting the unaligned key
        # instead published a new node on every decode step -- a 17-token
        # suffix, then an 18-token one -- so a 32-token context stored 152
        # tokens and a 48-token one 288. The request plane has always floored
        # here (radix_tree.cache_unfinished_req); this did not.
        aligned = (len(full) // self.block_size * self.block_size
                   if self.block_size > 1 else len(full))
        residual = len(full) - aligned
        key = full[:aligned]
        if not key:
            # Nothing publishable yet, but the tokens are still resident: keep
            # them reserved or the pool under-counts a whole block per turn.
            who = owner if owner is not None else program_id
            self._inflight[who] = residual
            self._invalidate()
            return 0
        self.radix._current_owner = program_id
        try:
            self.radix.insert(key)
        finally:
            self.radix._current_owner = None
        node = self.radix.match_prefix(key).last_device_node
        who = owner if owner is not None else program_id
        prev = self._lock_holder.pop(who, None)
        if node is not None and prev is not node:
            # Order matters: take the new lock BEFORE dropping the old. The new
            # node is a descendant of the old, so releasing first can make the
            # whole path reclaimable for the instant in between.
            #
            # Only when it MOVES. Block alignment made consecutive decode steps
            # resolve to the same node -- commits 33..47 all floor to the
            # 32-token one -- and incrementing each time while `prev is node`
            # skipped the matching unlock took fifteen references the single
            # release at completion could not give back. A finished request then
            # left 32 tokens locked with no policy involved, and nothing
            # reclaims a locked node.
            self.radix.inc_lock_ref(node)
        if prev is not None and prev is not node:
            self._unlock(prev)
        if node is not None:
            self._lock_holder[who] = node
        # The tail that does not fill a block stays reserved: it occupies KV
        # exactly as vLLM's partially-filled block does, and the tree cannot
        # account for it because it was not published.
        if residual:
            self._inflight[who] = residual
        else:
            self._inflight.pop(who, None)
        self._invalidate()
        # The reservation was sized against a prefix that may since have been
        # evicted, so the insert can have added more than `allocate` charged.
        # Give the difference back now: over-subscription is invisible while it
        # happens (`free` clamps at zero) and surfaces later as a stall with a
        # pool full of memory nothing will reclaim.
        over = -self.headroom()
        if over > 0:
            self._evict_lru(over)
            self._invalidate()
            self.counters["overflow_reclaims"] += 1
        return len(key)

    def hold_prefix(self, owner: str, token_ids: Sequence[int]) -> int:
        """Lock the cached prefix a request was just credited a hit against."""
        key = list(token_ids)
        if not key:
            return 0
        node = self.radix.match_prefix(key).last_device_node
        if node is None or node is self.radix.root_node:
            return 0
        prev = self._lock_holder.get(owner)
        if prev is node:
            return 0
        self.radix.inc_lock_ref(node)      # new first, as in `commit`
        if prev is not None:
            self._unlock(prev)
        self._lock_holder[owner] = node
        self._invalidate()
        return len(key)

    def _unlock(self, node) -> int:
        """Drop the one lock held on `node`'s chain."""
        if node is None or node is self.radix.root_node:
            return 0
        if node.lock_ref <= 0:
            # A lock dropped twice. Silently this leaves lock_ref negative, and
            # `_evict_lru` requires exactly 0, so the blocks become permanently
            # unreclaimable -- a stall with the pool full of memory nothing can
            # take. Fail where it happens, not 300 preemptions later.
            raise AssertionError(
                f"unlocking a node at lock_ref={node.lock_ref}: the hold was "
                f"already dropped (node len={len(node.key)}, "
                f"owners={sorted(node.owners)})")
        return self.radix.dec_lock_ref(node)

    def drop_reservation(self, program_id: str,
                         owner: Optional[str] = None) -> int:
        """Give back a reservation whose step will never complete -- the turn was
        preempted. Without this the pool leaks the difference."""
        self._invalidate()
        return self._inflight.pop(owner or program_id, 0)

    def drop_hold(self, owner: str) -> int:
        """Release the prefix hold from an `allocate` whose step never ran."""
        held = self._lock_holder.pop(owner, None)
        if held is None:
            return 0
        freed = self._unlock(held)
        self._invalidate()
        return freed

    def inflight_tokens(self) -> int:
        """Scheduled-but-uncommitted tokens. Occupied, unmatchable."""
        return sum(self._inflight.values())

    def release(self, token_ids: Sequence[int],
                owner: Optional[str] = None) -> int:
        """Unlock a turn's blocks. They become ordinary cache; they are NOT freed."""
        if owner is not None and owner in self._lock_holder:
            freed = self._unlock(self._lock_holder.pop(owner))
            # The turn is over: its unfilled tail is not resident any more
            # either. commit() keeps that block reserved while the turn runs
            # (it is owned, like vLLM's partly-filled block, and unpublishable);
            # leaving it charged here billed the pool for every finished turn's
            # last partial block forever.
            self._inflight.pop(owner, None)
            self._invalidate()
            return freed
        key = list(token_ids)
        if not key:
            return 0
        node = self.radix.match_prefix(key).last_device_node
        self._invalidate()
        return self.radix.dec_lock_ref(node) if node is not None else 0

    # ----------------------------------------------------------- operations

    def pin(self, program_id: str, request_id: str, now: float,
            deadline_ts: Optional[float] = None) -> None:
        self.orch.grant_pin(program_id, request_id, now, deadline_ts)
        self._invalidate()
        self.counters["pins_created"] += 1

    def unpin(self, program_id: str, request_id: Optional[str] = None) -> None:
        state = self.orch.get(program_id)
        if state is not None and state.pins:
            self.orch.release_pin(program_id, request_id)
            self.counters["pins_released"] += 1
            self._invalidate()

    def keep(self, program_id: str) -> None:
        """Retain without a pin: the blocks stay ordinary cache, subject to LRU.

        Distinct from `pin`, which makes them unreclaimable without breaking it,
        and distinct from saying nothing -- a mechanism check needs to tell a
        decision to do nothing from the absence of a decision.
        """
        return None

    def evict(self, program_id: str) -> int:
        """Drop this program's blocks that nothing else needs.

        Shared nodes survive with the program removed from their owner set:
        another program still depends on them, and taking them would charge one
        program's decision to a program that made none. Locked nodes are never
        touched.
        """
        freed = 0
        for node in list(self._nodes()):
            if program_id not in node.owners or node.lock_ref > 0:
                continue
            node.owners.discard(program_id)
            if not node.owners and not node.children:
                freed += len(node.key)
                self._tier.pop(node.id, None)
                self.radix._delete_leaf(node)
        if freed:
            self.counters["retention_discards"] += 1
        self._invalidate()
        return freed

    def offload(self, program_id: str, tier: Tier = Tier.CPU) -> int:
        """Move this program's unreferenced blocks to a slower tier.

        Frees NPU tokens without losing the prefix: the program pays a transfer
        on its next turn instead of a full reprefill.
        """
        moved = 0
        for node in self._nodes():
            if (program_id in node.owners and node.lock_ref == 0
                    and self.tier_of(node) is Tier.NPU):
                self._tier[node.id] = tier
                moved += len(node.key)
        if moved:
            self.counters["offloads"] += 1
        self._invalidate()
        return moved

    # ------------------------------------------------------------- pressure

    def on_pressure(self, need_tokens: int, now: float,
                    policy: Optional[Callable] = None) -> Reclaim:
        """Reclaim `need_tokens`, giving the policy first refusal.

        The policy may decline -- return None or nothing -- and then the valve
        runs: expired pins first, then the tree's own LRU. Declining must be
        safe, because a pressure hook that can deadlock the engine when a policy
        misbehaves is worse than no hook, and under policy search policies do
        misbehave.

        A policy is never allowed to take LOCKED blocks. A live turn's KV is not
        a policy decision; preempting the turn is, and that is the scheduler's.
        """
        declined = True
        broke: List[Tuple[str, str]] = []
        freed = 0

        if policy is not None:
            chosen = None
            try:
                chosen = policy(list(self.orch.on_instance(self.instance)),
                                need_tokens, now)
            except Exception:          # a bad candidate must not kill the run
                chosen = None
            if chosen:
                declined = False
                for program_id in chosen:
                    state = self.orch.get(program_id)
                    if state is None:
                        continue
                    if state.pins:
                        broke.extend((program_id, p.request_id)
                                     for p in state.pins)
                        self.counters["pins_broken"] += len(state.pins)
                        self.orch.release_pin(program_id)
                    freed += self.evict(program_id)
                    if freed >= need_tokens:
                        break

        if freed < need_tokens:
            freed += self._valve(need_tokens - freed, now)

        return Reclaim(tokens=freed, broke_pins=tuple(broke), declined=declined)

    def _valve(self, need_tokens: int, now: float) -> int:
        """The engine's own reclaim, when the policy declines or falls short.

        Expired pins first -- protection the policy has already given up on --
        then the tree's LRU eviction, which skips locked nodes by construction.
        """
        freed = 0
        for program_id, pin in list(self.orch.expired_pins(now)):
            self.orch.release_pin(program_id, pin.request_id)
            self.counters["pins_expired"] += 1
            freed += self.evict(program_id)
            if freed >= need_tokens:
                return freed

        if freed < need_tokens:
            freed += self._evict_lru(need_tokens - freed)
        return freed

    def _evict_lru(self, need_tokens: int) -> int:
        """LRU over CACHED blocks only -- pins are skipped.

        The tree's own `evict` cannot do this. It skips `lock_ref > 0`, which
        is a LIVE reference, and a policy pin is not one: it lives in the
        orchestrator and the tree has never heard of it. Calling it here meant
        pressure took the pinned program's blocks and left the unpinned one's
        alone, because the pinned program had committed first and was therefore
        the older entry -- the pin was not merely ignored, it was actively
        counterproductive. A retention policy in that world decides, increments
        its counter, produces a plausible JCT, and protects nothing.

        Pins are broken only where that is a decision someone made and
        recorded: `on_pressure`'s expired pass, or a policy naming a victim.
        Never here, silently, as the oldest thing to hand.
        """
        # Nothing unlocked anywhere: the walk cannot find a candidate, so do
        # not take it. The tree keeps this count incrementally, and under a
        # tight pool this is the common case -- `on_pressure` fired on 38% of
        # all acquires in one profile, and each call was walking ~3,600 nodes
        # to discover there was nothing to take. 75.7 million node visits,
        # 57% of total runtime.
        if self.radix.evictable_size() <= 0:
            return 0

        pinned = self._pinned_programs()
        def takeable(node) -> bool:
            return (node is not self.radix.root_node
                    and not node.children
                    and node.lock_ref == 0
                    and self.tier_of(node) is Tier.NPU
                    and self.classify(node, pinned) is BlockClass.CACHED)

        # Oldest first, and the parent is pushed back after a child goes --
        # the tree's own algorithm (radix_tree.evict). A single pass over
        # today's leaves would stop at the tail of each chain and free a
        # fraction of what it reported being able to free.
        # Collected fresh. A maintained heap was tried and reverted: nodes
        # created by a split, or made leaves when a sibling is deleted, are
        # never pushed, so the heap is an incomplete view of what is
        # reclaimable and eviction quietly stops finding candidates that exist.
        # That is the same five-site invariant as the lock protocol, on a tree
        # that rewrites its own shape underneath it, and it cost correctness
        # for about half the runtime. The walk stays until the leaf set can be
        # maintained by the tree itself rather than guessed at from outside.
        leaves = self._collect_evictable_leaves()
        heapq.heapify(leaves)
        freed = 0
        while freed < need_tokens and leaves:
            node = heapq.heappop(leaves)
            if not takeable(node):
                continue
            parent = node.parent
            freed += len(node.key)
            self._tier.pop(node.id, None)
            self.radix._delete_leaf(node)
            self.radix._record_remove_event(node)
            if parent is not None and takeable(parent):
                heapq.heappush(leaves, parent)
        self._invalidate()
        return freed

    def _collect_evictable_leaves(self) -> List[TreeNode]:
        """Leaves, via the tree's own collector.

        `_nodes()` is a generator that yields every node with a key; building
        a list comprehension over it cost 110 s of a 343 s profile. The tree
        walks its own structure without the per-node generator frame.
        """
        return [n for n in self.radix._collect_leaves()
                if n is not self.radix.root_node]
