"""Synchronous Autellix integration with vLLM v0.19's real scheduler.

The small EngineCore patch calls schedule()/complete() at actual execution
boundaries. Native vLLM still builds batches, updates token IDs, and reports
outputs; this adapter controls selection and physical CPU swap ownership.
"""
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

from policies.autellix_runtime import AutellixRuntime, QueueConfig


@dataclass
class SwappedRequest:
    computed_tokens: int
    cpu_blocks: tuple


class PolicyEngine:
    def __init__(self, core, config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        from vllm.v1.request import RequestStatus
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue

        self.core = core
        self.scheduler = s = core.scheduler
        self.status = RequestStatus
        self.queue_type = FCFSRequestQueue
        v = core.vllm_config
        if config['name'] != 'autellix':
            raise ValueError('this execution adapter implements autellix')
        if s._admission is not None or s.kv_cache_manager._pin_at_free_ttl is not None:
            raise ValueError('Autellix engine cannot combine with an admission gate '
                             'or completion-time retention pins')
        if (v.scheduler_config.async_scheduling or core.batch_queue is not None
                or v.parallel_config.pipeline_parallel_size != 1
                or v.parallel_config.data_parallel_size != 1
                or v.speculative_config is not None or s.connector is not None
                or s.is_encoder_decoder or v.lora_config is not None):
            raise ValueError('policy engine requires synchronous TP-only text decoding '
                             'without speculation, LoRA or an external KV connector')
        groups = s.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec):
            raise ValueError('policy swap requires one full-attention KV group')
        self.block_size = groups[0].kv_cache_spec.block_size
        self.world_size = v.parallel_config.world_size
        self.runtime = AutellixRuntime(QueueConfig(
            tuple(config['service_boundaries_s']), tuple(config['quanta_s']),
            config['starvation_ratio']))
        # Keep thresholds explicit: no workload-specific tuning hidden in code.
        self.replan_steps = int(config.get('replan_steps', 1))
        if self.replan_steps <= 0:
            raise ValueError('replan_steps must be positive')
        # Calls queued on the engine beyond the fitting set, admitted natively
        # as soon as a selected call finishes before the next replan.
        self.overprovision = int(config.get('overprovision_calls', 0))
        if self.overprovision < 0:
            raise ValueError('overprovision_calls must be nonnegative')
        self.swap_transport = str(config.get('swap_transport', 'batched'))
        if self.swap_transport not in ('batched', 'contiguous'):
            raise ValueError("swap_transport must be 'batched' or 'contiguous'")
        replies = self._rpc('policy_kv_init', int(config['cpu_bytes_per_rank']),
                            self.swap_transport)
        self._worker_init = sorted(replies, key=lambda r: r['rank'])
        self._migration = None
        self.cpu_capacity = min(r['cpu_blocks'] for r in replies)
        if self.cpu_capacity <= 0:
            raise ValueError('CPU swap pool has no blocks')
        self.free_cpu = deque(range(self.cpu_capacity))
        self.swapped = {}
        self._active = ()
        self._selection = ()
        self._overprovisioned = set()
        self._steps = 0
        self._tags = {}
        self._completed = {}
        self.stats = {'swap_out_blocks': 0, 'swap_in_blocks': 0,
                      'cpu_capacity_holds': 0, 'execution_s': 0.0,
                      'swap_seconds': 0.0, 'swap_transport': self.swap_transport,
                      'overprovisioned_admissions': 0}

    def _rpc(self, method, *args):
        replies = self.core.model_executor.collective_rpc(method, args=args)
        ranks = [r['rank'] for r in replies]
        if len(ranks) != self.world_size or set(ranks) != set(range(self.world_size)):
            raise RuntimeError(f'{method}: missing or duplicated worker acknowledgements')
        return replies

    def _copy(self, gpu, cpu, to_cpu):
        replies = self._rpc('policy_kv_copy', list(gpu), list(cpu), to_cpu)
        if any(r['blocks'] != len(gpu) for r in replies):
            raise RuntimeError('workers did not acknowledge the complete KV transfer')
        # The slowest rank bounds the transfer; workers report device time.
        self.stats['swap_seconds'] += max(float(r.get('seconds', 0.0)) for r in replies)

    def _swap_out(self, request):
        s = self.scheduler
        rid = request.request_id
        blocks = tuple(s.kv_cache_manager.get_block_ids(rid)[0])
        if not blocks:
            return
        if len(blocks) > len(self.free_cpu):
            raise MemoryError('CPU swap pool cannot hold the preemption victim')
        cpu = tuple(self.free_cpu.popleft() for _ in blocks)
        # A failed worker RPC is fatal; do not free/reuse any potentially active
        # DMA buffer. Original request ownership remains intact on that path.
        self._copy(blocks, cpu, True)
        self.swapped[rid] = SwappedRequest(request.num_computed_tokens, cpu)
        s.kv_cache_manager.free(request)
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        request.status = self.status.PREEMPTED
        if s.log_stats:
            from vllm.v1.engine import EngineCoreEventType
            request.record_event(EngineCoreEventType.PREEMPTED, time.monotonic())
        self.stats['swap_out_blocks'] += len(blocks)

    def _swap_in(self, request):
        s = self.scheduler
        record = self.swapped[request.request_id]
        blocks = s.kv_cache_manager.allocate_slots(
            request, 0, num_external_computed_tokens=record.computed_tokens,
            delay_cache_blocks=True)
        if blocks is None:
            return False
        gpu = s.kv_cache_manager.get_block_ids(request.request_id)[0]
        if len(gpu) != len(record.cpu_blocks):
            raise RuntimeError('restored KV shape differs from the suspended request')
        self._copy(gpu, record.cpu_blocks, False)
        request.num_computed_tokens = record.computed_tokens
        s.kv_cache_manager.cache_blocks(request, record.computed_tokens)
        self.free_cpu.extend(record.cpu_blocks)
        del self.swapped[request.request_id]
        self.stats['swap_in_blocks'] += len(gpu)
        return True

    def _reconcile(self, now):
        s = self.scheduler
        for rid in tuple(self.runtime.calls):
            if rid not in s.requests:
                self.runtime.cancel(rid, now)
                self._tags.pop(rid, None)
                old = self.swapped.pop(rid, None)
                if old is not None:
                    self.free_cpu.extend(old.cpu_blocks)
        for rid, request in s.requests.items():
            if rid in self.runtime.calls:
                continue
            if request.has_encoder_inputs or request.resumable:
                raise ValueError('policy engine requires non-streaming text requests')
            extra = request.sampling_params.extra_args or {}
            if 'program_id' not in extra:
                raise ValueError('Autellix requests require sampling extra_args.program_id')
            self.runtime.arrive(rid, str(extra['program_id']), now,
                                extra.get('program_service_s'),
                                extra.get('program_wait_s'))
            self._tags[rid] = str(extra.get('kv_tag', rid))

    def abort_completed(self):
        """Reclaim suspended calls even when cancellation leaves the engine idle."""
        if self._active:
            # EngineCore drains aborts after GPU execution but before completing
            # its scheduled output. complete() accounts that batch first.
            return
        self._reconcile(time.monotonic())
        if self._migration is not None:
            self._migration.release_consumed()

    def before_abort(self, request_ids):
        if self._migration is not None:
            self._migration.before_abort(request_ids)

    def migration(self, operation, *args):
        from bench.core.policy_migration import PrefixMigration

        if self._active:
            raise RuntimeError('migration must run at an engine execution boundary')
        if operation == 'block_size':
            return self.block_size
        allowed = {'begin_export', 'begin_import', 'read', 'write', 'publish',
                   'release_export', 'release_import'}
        if operation not in allowed:
            raise ValueError('unsupported migration operation')
        if self._migration is None:
            self._migration = PrefixMigration(self, [r['layout'] for r in self._worker_init])
        return getattr(self._migration, operation)(*args)

    def _plan(self, now):
        s = self.scheduler
        pool = s.kv_cache_manager.block_pool
        owned = {r.request_id: s.kv_cache_manager.get_blocks(r.request_id).blocks[0]
                 for r in s.running}
        # All residents can be swapped if selected against. Count distinct GPU
        # blocks once; shared prefixes must not manufacture physical capacity.
        references = Counter(b.block_id for blocks in owned.values() for b in blocks)
        releasable = {bid for bid, count in references.items()
                      if pool.blocks[bid].ref_cnt == count}
        capacity = pool.get_num_free_blocks() + len(releasable)
        budget = s.max_num_scheduled_tokens
        needed_tokens, needed_blocks, reused_blocks = {}, {}, {}

        def fits(rid, selected):
            request = s.requests[rid]
            if len(selected) >= s.max_num_running_reqs:
                return False
            remaining = budget - sum(needed_tokens[i] for i in selected)
            if remaining <= 0:
                return False
            computed = (self.swapped[rid].computed_tokens if rid in self.swapped
                        else request.num_computed_tokens)
            existing = owned.get(rid, ())
            if not computed and rid not in self.swapped:
                existing, computed = self._prefix(request)
            count = request.num_tokens - computed
            threshold = s.scheduler_config.long_prefill_token_threshold
            if threshold > 0:
                count = min(count, threshold)
            count = min(count, remaining)
            if count <= 0:
                return False
            needed_tokens[rid] = count
            reserve_to = (request.num_tokens if s.scheduler_reserve_full_isl
                          else computed + count)
            needed_blocks[rid] = max(0, math.ceil(reserve_to / self.block_size) - len(existing))
            # Blocks held by migration receipts or another external owner are
            # already outside free capacity. Shared selected prefixes consume
            # capacity once, and only uncached tokens consume compute budget.
            reused_blocks[rid] = {b.block_id for b in existing
                                  if b.ref_cnt == 0 or b.block_id in releasable}
            ids = (*selected, rid)
            reused = set().union(*(reused_blocks[i] for i in ids))
            return sum(needed_blocks[i] for i in ids) + len(reused) <= capacity

        return self.runtime.plan(now, tuple(owned), fits, self.overprovision)

    def _prefix(self, request):
        manager = self.scheduler.kv_cache_manager
        if not manager.enable_caching or request.skip_reading_prefix_cache:
            return (), 0
        # This read-only lookup does not count an extra cache query in metrics.
        groups, count = manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1)
        return groups[0], count

    @contextmanager
    def _hold_prefixes(self, selected):
        pool = self.scheduler.kv_cache_manager.block_pool
        blocks = {}
        for rid in selected:
            request = self.scheduler.requests[rid]
            if request.num_computed_tokens == 0 and rid not in self.swapped:
                prefix, _ = self._prefix(request)
                blocks.update((b.block_id, b) for b in prefix)
        held = tuple(blocks.values())
        pool.touch(held)
        try:
            yield
        finally:
            pool.free_blocks(reversed(held))

    def schedule(self):
        s = self.scheduler
        now = time.monotonic()
        self._reconcile(now)
        if self._steps % self.replan_steps == 0 or not self._selection:
            self._selection = self._planned(now)
        selected = tuple(r for r in self._selection if r in s.requests)
        if not selected and s.requests:
            # A multi-step window may have lost its last call to completion.
            self._selection = self._planned(now)
            selected = self._selection
            if not selected:
                raise MemoryError('no queued request fits the policy engine KV/token limits')
        victims = [r for r in s.running if r.request_id not in selected]
        cpu_needed = sum(len(s.kv_cache_manager.get_block_ids(r.request_id)[0])
                         for r in victims)
        if cpu_needed > len(self.free_cpu):
            # Preserve residents when host memory cannot hold the full change.
            # Never substitute unrequested recomputation for a failed swap.
            self.stats['cpu_capacity_holds'] += 1
            selected = tuple(r.request_id for r in s.running)
            victims = []
        # Reserve planned prefix hits before restoring another call or letting
        # earlier queue entries allocate blocks that might evict those hits.
        with self._hold_prefixes(selected):
            output = self._schedule_selected(selected, victims)
        if self._migration is not None:
            self._migration.release_consumed()
        self._active = tuple(output.num_scheduled_tokens)
        if self._active:
            admitted = [rid for rid in self._active if rid in self._overprovisioned]
            self.stats['overprovisioned_admissions'] += len(admitted)
            self._overprovisioned.difference_update(admitted)
            self.runtime.start_batch(self._active, time.monotonic())
        self._steps += 1
        return output

    def _planned(self, now):
        plan = self._plan(now)
        self._overprovisioned = set(plan.overprovisioned)
        return plan.selected + plan.overprovisioned

    def _schedule_selected(self, selected, victims):
        s = self.scheduler
        for request in victims:
            self._swap_out(request)
        for rid in selected:
            if rid not in self.swapped:
                continue
            if self._swap_in(s.requests[rid]):
                continue
            if rid in self._overprovisioned:
                continue  # queued on the engine; restored when a slot frees
            raise RuntimeError('planned request could not reserve restoration blocks')

        # Native vLLM visits RUNNING before WAITING. Present the selected set as
        # one FCFS queue so a newly arrived high-priority call really precedes a
        # lower-priority resident. Only actual CPU resumes replace worker block
        # IDs; resident requeues retain worker state, allocations and progress.
        pending = {r.request_id: r for r in (*s.running, *s.waiting, *s.skipped_waiting)}
        s._policy_resident_requeues = {
            r.request_id for r in s.running if r not in victims}
        s._policy_resident_block_counts = {
            rid: tuple(len(ids) for ids in s.kv_cache_manager.get_block_ids(rid))
            for rid in s._policy_resident_requeues}
        for request in s.running:
            request.status = self.status.PREEMPTED
        s.running = []
        s.waiting = self.queue_type([pending[r] for r in selected])
        s.skipped_waiting = self.queue_type()
        held = [r for rid, r in pending.items() if rid not in selected]
        try:
            output = s.schedule()
        finally:
            for request in held:
                s.waiting.add_request(request)
        return output

    def complete(self):
        if not self._active:
            return
        replies = self._rpc('policy_model_elapsed')
        service = max(r['seconds'] for r in replies)
        if not math.isfinite(service) or service < 0:
            raise RuntimeError('worker returned invalid execution time')
        done = [rid for rid in self._active if rid not in self.scheduler.requests]
        for rid in done:
            call = self.runtime.calls[rid]
            self._completed[self._tags.pop(rid)] = {
                'execution_s': call.execution_s + service,
                'wait_s': call.wait_s,
                'program_id': call.program_id}
        self.runtime.finish_batch(time.monotonic(), done, execution_s=service)
        self.stats['execution_s'] += service
        self._active = ()
        self._reconcile(time.monotonic())
        if self._migration is not None:
            self._migration.release_consumed()

    def snapshot(self, tag=None):
        if tag is not None:
            return self._completed[tag]
        return {**self.stats, 'cpu_blocks_used': self.cpu_capacity - len(self.free_cpu),
                'swapped_requests': len(self.swapped),
                'active_requests': len(self.runtime.calls),
                'migration_exports': len(self._migration.exports) if self._migration else 0,
                'migration_imports': len(self._migration.imports) if self._migration else 0,
                'program_service_s': {pid: p.service_s
                                      for pid, p in self.runtime.processes.items()}}
