"""Causal, bounded host copies for the radix-cache simulator.

This is deliberately separate from native InferCept residency. The radix
index keeps a source chain until its copy completes; it cannot exchange
individual occupied CPU/GPU pages using the native engine's scratch buffer.
Transfers are serialized in the trace, not claimed to overlap computation.
"""
from dataclasses import dataclass
import math

from policies.infercept_budget import measured_swap_limit, plan_swap_budget
from policies.utils.waste_model import discard_waste, preserve_waste, t_fwd_s

from .memory_model import Device


class MinWasteSwapPolicy:
    """InferCept's rule: rank by min(preserve, discard), drop a source once
    holding it costs more than re-prefilling it (Eq. 2 vs Eq. 4)."""

    name = "min-waste"

    def __init__(self, profile):
        self.profile = profile

    def score(self, copy, now_s, requests):
        tokens = len(copy.ids) - copy.copied
        # The radix source remains fully resident until the last chunk is
        # copied. A partial host copy has not reduced GPU occupancy yet.
        preserve = preserve_waste(len(copy.ids), now_s)
        discard = discard_waste(tokens, sum(not r.is_prefill() for r in requests),
                                sum(r.num_computed_tokens for r in requests),
                                self.profile)
        return min(preserve, discard), preserve, discard

    def drop(self, copy, now_s, requests):
        _, preserve, discard = self.score(copy, now_s, requests)
        return preserve > discard

    def next_reconsider_ns(self, copy, now_ns, requests):
        # Model the engine's 1 ms sleep between empty iterations, anchored
        # at this failed scheduling check. Jump to the first poll strictly
        # after the cost crossing; do not execute all intervening polls.
        # Empty-iteration processing overhead is unmeasured and excluded.
        _, _, discard = self.score(copy, 0, requests)
        threshold_ns = copy.since_ns + math.floor(discard / len(copy.ids) * 1e9)
        poll_ns = 1_000_000
        polls = max(1, (threshold_ns - now_ns) // poll_ns + 1)
        return now_ns + polls * poll_ns


class PreemptSwapPolicy:
    """Autellix's rule: the scheduler already decided to evict this call.

    Its KV must reach the host or the call re-prefills, which is the outcome
    swapping exists to avoid, so nothing here may drop a source. Candidates
    go oldest first: the call that has been waiting to leave the GPU longest
    is the one blocking the pool.
    """

    name = "preempt"

    def score(self, copy, now_s, requests):
        return (now_s, None, None)

    def drop(self, copy, now_s, requests):
        return False

    def next_reconsider_ns(self, copy, now_ns, requests):
        return None  # This policy never discards a queued source.


@dataclass
class Copy:
    request_id: str
    ids: list
    node: object
    since_ns: int
    copied: int = 0
    cancelled: bool = False


class HostSwap:
    def __init__(self, memory, profile, stats, policy=None):
        self.memory = memory
        self.profile = profile
        self.stats = stats
        # Who ranks candidates and who may be dropped. The forward-time model
        # stays on the controller because the budget T_swap(N) = T_fwd(B) is a
        # property of the hardware, not of the policy that wants the copy.
        self.policy = policy if policy is not None else MinWasteSwapPolicy(profile)
        self.queued = {}
        self.pending = {}

    def enqueue(self, request_id, request, now_ns):
        memory = self.memory
        cache = memory.npu_prefix_cache
        ids = (request.input_hash_ids + request.output_hash_ids)[:request.num_computed_tokens]
        # Never copy generated KV for which the source has no resident pages.
        matched = cache.match_prefix(ids)
        count = matched.hit_length // memory.block_size * memory.block_size
        if not count:
            return 0
        ids = ids[:count]
        node = cache.match_prefix(ids).last_device_node
        self.cancel(request_id)
        cache.inc_lock_ref(node)
        self.queued[request_id] = Copy(request_id, ids, node, now_ns)
        # A queued copy is not a completed swap. Counters advance at commit.
        return 0

    def cancel(self, request_id):
        copy = self.queued.pop(request_id, None)
        if copy is None:
            return
        copy.cancelled = True
        if not any(item[0] is copy for items in self.pending.values() for item in items):
            self.memory.npu_prefix_cache.dec_lock_ref(copy.node)

    def close(self):
        if self.pending:
            raise RuntimeError('cannot close host swap with unfinished transfers')
        for request_id in list(self.queued):
            self.cancel(request_id)

    def _score(self, copy, now_ns, requests):
        elapsed = max(0, now_ns - copy.since_ns) / 1e9
        return self.policy.score(copy, elapsed, requests)

    def reconsider(self, now_ns, requests):
        """Release stale queued sources even when no forward batch will run."""
        removed = 0
        active = {id(item[0]) for items in self.pending.values() for item in items}
        for request_id, copy in list(self.queued.items()):
            if id(copy) in active:
                continue
            elapsed = max(0, now_ns - copy.since_ns) / 1e9
            if self.policy.drop(copy, elapsed, requests):
                self.cancel(request_id)
                self.memory.npu_prefix_cache.evict_chain(copy.node)
                removed += 1
        self.memory.apply_kv_cache_events()
        return removed

    def next_reconsider_ns(self, now_ns, requests):
        """Next policy event when queued sources cannot get a forward batch.

        In-flight DMA owns its source until completion and must never be
        discarded by an idle timer.
        """
        active = {id(item[0]) for items in self.pending.values() for item in items}
        times = [self.policy.next_reconsider_ns(copy, now_ns, requests)
                 for copy in self.queued.values()
                 if id(copy) not in active and not copy.cancelled]
        return min((time for time in times if time is not None), default=None)

    def plan(self, batch):
        """Use this upcoming batch's budget; retain sources until completion."""
        if batch.batch_id in self.pending:
            raise RuntimeError('host copy batch planned twice')
        memory = self.memory
        block = memory.block_size
        per_rank_bytes = int(memory.get_kv(block))
        rate = memory.cpu_mem_bw_gbs * 1e9
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('host swap requires measured positive per-rank bandwidth')
        batch.host_link_bytes_s = rate
        if batch.total_len <= 0:
            return
        limit = measured_swap_limit(t_fwd_s(self.profile, batch.total_len),
                                    per_rank_bytes, rate)
        # Restores already admitted by the request scheduler consume the same
        # link. If they exceed the hiding window, no store is granted.
        loads = math.ceil(batch.load / per_rank_bytes)
        cpu = memory.second_tier_prefix_cache
        free_cpu = int(memory.avail_size(Device.CPU) // (per_rank_bytes * memory.num_npus))
        active = {id(item[0]) for items in self.pending.values() for item in items}
        candidates = [copy for copy in self.queued.values() if id(copy) not in active]
        for copy in candidates:
            # If an external cache user reclaimed an earlier host chunk,
            # copying only its tail would manufacture a missing prefix.
            hit = cpu.peek_prefix_length(copy.ids[:copy.copied])
            copy.copied = min(copy.copied, hit // block * block)
        candidates.sort(key=lambda copy: (-self._score(copy, batch.batch_time, batch.requests)[0],
                                          copy.since_ns, copy.request_id))
        demand = sum((len(copy.ids) - copy.copied) // block for copy in candidates)
        # The scheduled batch has already reserved its GPU allocation. Recover
        # the pre-allocation capacity for the shared budget calculation, using
        # reclaimable cache as free physical pages, as the native pool does.
        new_demand = math.ceil(min(batch.total_len, self.profile.S) / block)
        free_gpu = max(0, int((memory.avail_size(Device.NPU)
                              + memory.evictable_size(Device.NPU)) // per_rank_bytes))
        free_gpu += math.ceil(batch.kv_size / per_rank_bytes)
        if self.policy.name == "min-waste":
            # Unlike the native pre-admission planner, this batch's restores
            # are committed. Deduct them first so optimization cannot replace
            # a mandatory load with an outgoing copy to increase new capacity.
            plan = plan_swap_budget(max(0, limit - loads),
                                    max(0, free_gpu - loads), free_cpu, 0,
                                    demand if loads <= limit else 0, new_demand)
            remaining = plan.store_blocks
        else:
            # Autellix has already chosen to evict these sources. InferCept's
            # optional, proactive store budget must not override that decision.
            remaining = min(demand, max(0, limit - loads), free_cpu)
        items = []
        for copy in candidates:
            take = min(remaining, (len(copy.ids) - copy.copied) // block)
            if not take:
                continue
            end = copy.copied + take * block
            # Reserve the destination BEFORE scheduling DMA. A previously
            # copied prefix may have been evicted; account for that too.
            hit = cpu.peek_prefix_length(copy.ids[:end])
            reserve = int(memory.get_kv(end - hit) * memory.num_npus)
            if reserve > memory.avail_size(Device.CPU):
                continue
            memory.cpu_swap_reserved += reserve
            # Protect an existing destination prefix against unrelated LRU
            # activity while this copy is in flight.
            target = cpu.match_prefix(copy.ids[:end]).last_device_node
            cpu.inc_lock_ref(target)
            items.append((copy, end, reserve, target))
            remaining -= take
            batch.host_store_bytes += take * per_rank_bytes
        if items:
            self.pending[batch.batch_id] = items
        # Budgeted copies have first claim. Compare preserve/discard for
        # sources that could not be transferred in this iteration.
        self.reconsider(batch.batch_time, batch.requests)

    def complete(self, batch):
        memory = self.memory
        cpu = memory.second_tier_prefix_cache
        for copy, end, reserve, target in self.pending.pop(batch.batch_id, ()):
            memory.cpu_swap_reserved -= reserve
            try:
                cpu.insert(copy.ids[:end])
                memory.apply_kv_cache_events()
            finally:
                cpu.dec_lock_ref(target)
            moved = end - copy.copied
            copy.copied = end
            self.stats['swapped_tokens'] += moved
            self.stats['swapped_out'] += moved // memory.block_size
            if copy.cancelled or end == len(copy.ids):
                memory.npu_prefix_cache.dec_lock_ref(copy.node)
                if not copy.cancelled:
                    memory.npu_prefix_cache.evict_chain(copy.node)
                    self.queued.pop(copy.request_id, None)
        memory.apply_kv_cache_events()


class RestoreGate:
    """FCFS whole-context restore admission for serialized host copies.

    Without this the simulator restores whatever the scheduler happened to
    admit, and charges the load afterwards: the budget bounds the total but
    nothing decides *whose* restore goes first. The paper restores in arrival
    order. This substrate restores whole prefixes, not native physical-page
    chunks. An oversized head must run alone rather than wait forever for a
    window that cannot grow. The trace charges its full transfer duration;
    this is not a claim of native overlap or chunk-restore fidelity.

    The budget is computed against the batch being assembled rather than the
    last one that ran: walking the queue in order, the accumulated token count
    *is* the upcoming forward pass, so a larger batch hides more restore. Using
    the previous batch's size would grant an idle engine a window it has not
    earned, which is the failure the store side already had to fix.
    """

    def __init__(self, memory, profile, stats):
        self.memory = memory
        self.profile = profile
        self.stats = stats
        stats.setdefault("restore_held", 0)
        stats.setdefault("restore_admitted", 0)
        stats.setdefault("restore_oversized", 0)

    def _restore_tokens(self, req):
        """Tokens this request would have to pull back from the host.

        Probed read-only against both tiers rather than read off the request:
        the gate runs before the scheduler's prefix_match, so the request's
        own storage_cache_hit is still zero here.
        """
        if not req.is_prefill() or req.storage_restored:
            return 0
        host = self.memory.peek_storage_hit(req)
        npu = self.memory.peek_prefix_hit(req)
        return max(0, int(host) - int(npu))

    def admit(self, waiting, max_num_batched_tokens):
        """Return the prefix of `waiting` whose restores fit, in queue order."""
        memory = self.memory
        block = max(1, memory.block_size)
        per_rank_bytes = int(memory.get_kv(block))
        rate = (memory.cpu_mem_bw_gbs or 0) * 1e9
        if rate <= 0 or not math.isfinite(rate):
            raise ValueError('FCFS restore needs a measured per-rank link rate')
        kept, tokens, restore_blocks = [], 0, 0
        for req in waiting:
            want = self._restore_tokens(req)
            step = max(1, int(req.original_input) - int(req.num_computed_tokens))
            ahead = min(tokens + step, int(max_num_batched_tokens))
            limit = measured_swap_limit(t_fwd_s(self.profile, max(1, ahead)),
                                        per_rank_bytes, rate)
            need = restore_blocks + -(-want // block)
            if want and need > limit:
                if not kept:
                    # A retry with identical inputs cannot enlarge the window.
                    # Let ordinary memory admission reserve the entire restore;
                    # HostSwap.plan grants no outgoing budget past this window.
                    kept.append(req)
                    self.stats["restore_admitted"] += 1
                    self.stats["restore_oversized"] += 1
                    break
                # Its turn has not come: hold it, and hold everything behind
                # it too, or the order stops being first-come.
                self.stats["restore_held"] += 1
                break
            kept.append(req)
            tokens = ahead
            if want:
                restore_blocks = need
                self.stats["restore_admitted"] += 1
        return kept
