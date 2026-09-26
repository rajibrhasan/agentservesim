"""Synchronous SAGA fairness preemption inside one native vLLM engine."""
import math
import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import RequestStatus

from .policy_scheduler import PolicyObservations


class SagaScheduler(PolicyObservations, Scheduler):
    def __init__(self, *args, **kwargs):
        cfg = kwargs['vllm_config']
        if cfg.scheduler_config.async_scheduling or cfg.speculative_config is not None:
            raise ValueError('SAGA fairness requires synchronous, nonspeculative scheduling')
        super().__init__(*args, **kwargs)
        self._policy_last_scheduled = {}
        self.afs_shares = {}
        self.afs_preemptions = 0
        self.afs_wait_since = {}
        self._saga_cache_hashes = {}

    def _free_blocks(self, request):
        # Keep content identities, never physical block IDs: a freed block can
        # be reused for an unrelated request before the next observation.
        extra = request.sampling_params.extra_args or {}
        program = extra.get('program_id')
        if program is not None:
            hashes = self._saga_cache_hashes.setdefault(str(program), set())
            for group in self.kv_cache_manager.get_blocks(request.request_id).blocks:
                hashes.update(b.block_hash for b in group
                              if not b.is_null and b.block_hash is not None)
        return super()._free_blocks(request)

    def kv_protection_stats(self):
        stats = super().kv_protection_stats()
        index = self.kv_cache_manager.block_pool.cached_block_hash_to_block
        resident = {}
        for program, hashes in list(self._saga_cache_hashes.items()):
            live = {h for h in hashes if index.get_one_block(h) is not None}
            if live:
                self._saga_cache_hashes[program] = live
                resident[program] = len(live)
            else:
                del self._saga_cache_hashes[program]
        stats['policy_observation']['cached_blocks_by_program'] = resident
        return stats

    def set_afs_shares(self, shares):
        shares = {str(k): float(v) for k, v in shares.items()}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in shares.values()):
            raise ValueError('AFS shares must be finite fractions')
        self.afs_shares = shares

    def _share(self, request):
        params = request.sampling_params
        extra = params.extra_args or {} if params is not None else {}
        return self.afs_shares.get(extra.get('saga_tenant', 'default'), 0.0)

    def _admission_blocked(self, urgent):
        """Check native dense-model admission without allocating or breaking pins.

        Waiting time alone is not contention: an aged call can arrive from
        the gateway while this engine has room to serve it alongside runners.
        Account for the next running pass, then its first prefill chunk.
        """
        if len(self.running) >= self.max_num_running_reqs:
            return True
        manager = self.kv_cache_manager
        budget = self.max_num_scheduled_tokens
        blocks_needed = 0
        threshold = self.scheduler_config.long_prefill_token_threshold
        for request in self.running:
            tokens = request.num_tokens - request.num_computed_tokens
            if threshold > 0:
                tokens = min(tokens, threshold)
            tokens = min(tokens, budget,
                         self.max_model_len - 1 - request.num_computed_tokens)
            if tokens <= 0:
                continue
            budget -= tokens
            end = request.num_computed_tokens + tokens
            blocks_needed += manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id, num_tokens=end,
                new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
                num_encoder_tokens=0,
                total_computed_tokens=request.num_computed_tokens,
                num_tokens_main_model=end)
        if budget <= 0:
            return True
        cached = manager.empty_kv_cache_blocks.blocks
        computed = urgent.num_computed_tokens
        if computed == 0 and manager.enable_caching and not urgent.skip_reading_prefix_cache:
            # Unlike get_computed_blocks, this does not record a second cache
            # query in the benchmark's hit statistics.
            cached, computed = manager.coordinator.find_longest_cache_hit(
                urgent.block_hashes, urgent.num_tokens - 1)
        tokens = urgent.num_tokens - computed
        if threshold > 0:
            tokens = min(tokens, threshold)
        if not self.scheduler_config.enable_chunked_prefill and tokens > budget:
            return True
        end = computed + min(tokens, budget)
        if self.scheduler_reserve_full_isl:
            end = urgent.num_tokens
        blocks_needed += manager.coordinator.get_num_blocks_to_allocate(
            request_id=urgent.request_id, num_tokens=end,
            new_computed_blocks=cached, num_encoder_tokens=0,
            total_computed_tokens=computed, num_tokens_main_model=end)
        return blocks_needed > manager.block_pool.get_num_free_blocks()

    def schedule(self):
        now = time.monotonic()
        ready = [r for r in self.waiting if r.status in
                 (RequestStatus.WAITING, RequestStatus.PREEMPTED)]
        self.afs_wait_since = {r.request_id: self.afs_wait_since.get(r.request_id, now)
                               for r in ready}
        victims = []
        if self.afs_shares and ready and self.running:
            urgent = max(ready, key=self._share)
            extra = urgent.sampling_params.extra_args or {}
            since = min(self.afs_wait_since[urgent.request_id],
                        float(extra.get('saga_ready_s', now)))
            victim = min(self.running, key=self._share)
            if (now - since > 0.5 and self._share(urgent) > self._share(victim)
                    and self._admission_blocked(urgent)):
                self.running.remove(victim)
                self._preempt_request(victim, now)
                self.waiting.remove_request(victim)
                victims.append(victim)
                self.afs_preemptions += 1
        if self.afs_shares:
            key = lambda r: (-self._share(r), r.arrival_time, r.request_id)
            self.waiting = FCFSRequestQueue(sorted(self.waiting, key=key))
            self.running.sort(key=key)
        try:
            output = super().schedule()
            output.preempted_req_ids.update(r.request_id for r in victims)
            return output
        finally:
            # Keep a preempted request out for one worker update, then let
            # native recomputation restore it with its sampled tokens intact.
            for request in victims:
                self.waiting.add_request(request)
