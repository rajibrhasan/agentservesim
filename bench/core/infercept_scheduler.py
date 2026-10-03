
import math
import logging
import os
import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import RequestStatus
from vllm.v1.engine import EngineCoreEventType

from policies.infercept_budget import measured_swap_limit, plan_swap_budget
from policies.utils.waste_model import (
    WasteProfile,
    discard_waste,
    preserve_waste,
    t_fwd_s,
)

from .infercept_residency import InferceptResidency
from .kv_measurement import TurnKV


class InferceptSessionScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        config = kwargs['vllm_config']
        if config.scheduler_config.async_scheduling or config.speculative_config is not None:
            raise ValueError('InferCept session lifecycle requires synchronous decoding without speculation')
        super().__init__(*args, **kwargs)
        if self.policy != SchedulingPolicy.FCFS:
            raise ValueError('InferCept requires original-arrival FCFS scheduling')
        self.intercepted_since = {}
        self.interception_count = 0
        self.observed_interception_s = {}

    def schedule(self):
        # Native FCFS always chooses skipped_waiting before waiting. That can
        # put a younger tool-resumed session ahead of an older preempted call.
        # Merge at the iteration boundary; native scheduling still skips blocked
        # tool/grammar inputs without preventing ready requests from running.
        key = lambda request: (request.arrival_time, request.request_id)
        self.waiting = FCFSRequestQueue(sorted(
            (*self.waiting, *self.skipped_waiting), key=key))
        self.skipped_waiting = FCFSRequestQueue()
        self.running.sort(key=key)
        return super().schedule()

    @staticmethod
    def _is_intercepted_session(request):
        params = request.sampling_params
        return bool(params is not None and (params.extra_args or {}).get('infercept_session'))

    def _update_request_as_session(self, session, update):
        if not self._is_intercepted_session(session):
            return super()._update_request_as_session(session, update)
        if update.mm_features:
            raise ValueError('InferCept continuation currently accepts text tool results only')
        if update.sampling_params is None or not (
                update.sampling_params.extra_args or {}).get('infercept_session'):
            raise ValueError('every InferCept continuation must retain its session tag')
        # Generic streaming truncates token history at the computed KV length.
        # That is not a token-history boundary after discard/partial offload.
        # Fold ALL known outputs into the prompt, including the sampled token,
        # then restore the actual computed length for scheduling/recomputation.
        computed = session.num_computed_tokens
        arrival = session.arrival_time
        if (update.sampling_params.extra_args or {}).get('infercept_full_prompt'):
            # The policy scheduler already reconciled physical ownership.
            # Replace the request sequence; never append a full prompt.
            session._all_token_ids.clear()
            session._output_token_ids.clear()
            session.prompt_token_ids.clear()
            session.block_hashes.clear()
            session.num_prompt_tokens = 0
            session.num_computed_tokens = 0
        else:
            session.num_computed_tokens = session.num_tokens
        try:
            super()._update_request_as_session(session, update)
        finally:
            session.num_computed_tokens = computed
        session.arrival_time = arrival
        session.max_tokens = update.max_tokens
        start = self.intercepted_since.pop(session.request_id, None)
        if start is not None:
            elapsed = time.monotonic() - start
            self.observed_interception_s[session.request_id] = elapsed

    def _handle_stopped_request(self, request):
        finished = super()._handle_stopped_request(request)
        if (not finished and self._is_intercepted_session(request)
                and request.status == RequestStatus.WAITING_FOR_STREAMING_REQ):
            self.intercepted_since[request.request_id] = time.monotonic()
            self.interception_count += 1
        return finished

    def _free_request(self, request, delay_free_blocks=False):
        self.intercepted_since.pop(request.request_id, None)
        return super()._free_request(request, delay_free_blocks=delay_free_blocks)


class InferceptPolicyScheduler(InferceptSessionScheduler):
    """InferCept's iteration-level preserve/discard/swap controller.

    Transfer capacity comes from the measured forward profile and measured
    per-rank host-link bandwidth supplied in ``additional_config``. No device
    constants or workload lookahead are embedded here. Unknown interception
    duration is estimated as elapsed time since interception, as in section
    4.4 of the paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = kwargs['vllm_config'].additional_config.get('infercept_policy')
        if not isinstance(cfg, dict):
            raise ValueError('InferCept needs additional_config.infercept_policy')
        if self.connector is None:
            raise ValueError('InferCept policy requires InferceptConnector')
        self.profile = WasteProfile.from_json(str(cfg['profile']))
        if self.profile.S <= 0:
            raise ValueError('InferCept saturation point must be positive')
        self.paper_scheduling = os.environ.get('INFERCEPT_PAPER_SCHEDULING') == '1'
        self.bandwidth_bytes_s = float(cfg['bandwidth_bytes_s'])
        if not math.isfinite(self.bandwidth_bytes_s) or self.bandwidth_bytes_s <= 0:
            raise ValueError('InferCept measured bandwidth must be positive')
        cache = kwargs['kv_cache_config']
        self.bytes_per_block = sum(t.size for t in cache.kv_cache_tensors) // cache.num_blocks
        if self.bytes_per_block <= 0:
            raise ValueError('InferCept could not determine physical block bytes')
        cpu_blocks = self.connector.host_bytes // self.bytes_per_block - self.connector.scratch_blocks
        self.residency = InferceptResidency(
            self.kv_cache_manager, self.connector, cpu_blocks)
        self._last_scheduled_tokens = 0
        self._progress_log_at = time.monotonic()
        self._resumed = set()
        self._pending_full_prompts = {}
        self._turn_kv = {}
        self._turn_kv_history = []
        self.infercept_stats = {
            'iterations': 0,
            'preserve_decisions': 0,
            'discard_decisions': 0,
            'swap_plans': 0,
            'planned_load_blocks': 0,
            'planned_store_blocks': 0,
            'unhidden_transfer_blocks': 0,
            'waiting_ownership_preemptions': 0,
        }

    def _handle_stopped_request(self, request):
        if request.request_id in self._turn_kv:
            self._turn_kv[request.request_id].complete = True
        finished = super()._handle_stopped_request(request)
        if (not finished and self._is_intercepted_session(request)
                and request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
                and request.request_id not in self.residency.states):
            self.residency.pause(request)
        return finished

    def _update_request_as_session(self, session, update):
        history = max(0, session.num_tokens - 1)
        if (update.sampling_params.extra_args or {}).get('infercept_full_prompt'):
            if self.residency.pending is not None:
                self._pending_full_prompts[session.request_id] = update
                return
            self.residency.reconcile_full_prompt(session, update.prompt_token_ids)
        super()._update_request_as_session(session, update)
        self._begin_kv_turn(session, history)
        state = self.residency.states.get(session.request_id)
        if state is None:
            return
        pending = self.residency.pending
        if state.cpu or (pending is not None and pending.contains(state)):
            session.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            self._resumed.add(session.request_id)
        elif state.discarded:
            self._resumed.add(session.request_id)
        else:
            # Preserve: the original pages are still resident and runnable.
            self.residency.resume(session.request_id)

    def _free_request(self, request, delay_free_blocks=False):
        self._pending_full_prompts.pop(request.request_id, None)
        self._resumed.discard(request.request_id)
        self.residency.cancel(request.request_id)
        return super()._free_request(request, delay_free_blocks=delay_free_blocks)

    def _resident_blocks(self, state):
        return len(self.residency.single.req_to_blocks.get(
            state.request.request_id, ()))

    def _waste(self, state, now):
        blocks = self._resident_blocks(state)
        tokens = min(state.computed_tokens, blocks * self.residency.block_size)
        started = self.intercepted_since.get(state.request.request_id, now)
        gap_s = max(0.0, now - started)
        running_context = sum(request.num_computed_tokens for request in self.running)
        running_queries = sum(
            request.num_computed_tokens >= request.num_tokens
            for request in self.running)
        preserve = preserve_waste(tokens, gap_s)
        discard = discard_waste(
            tokens, running_queries, running_context, self.profile)
        return min(preserve, discard), preserve, discard

    @staticmethod
    def _take_chunks(records, budget, count):
        actions = []
        remaining = budget
        for record in records:
            available = count(record)
            if remaining <= 0:
                break
            take = min(available, remaining)
            if take:
                actions.append((record.request.request_id, take))
                remaining -= take
        return tuple(actions)

    def _anticipated_forward_tokens(self):
        """Conservative current-iteration work available to hide DMA."""
        if not self.running and self.scheduler_reserve_full_isl:
            ready = sorted(
                (r for r in (*self.waiting, *self.skipped_waiting)
                 if r.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)),
                key=lambda r: (r.arrival_time, r.request_id))
            # Native FCFS stops at an inadmissible ready head. Queued tokens
            # behind it provide neither executable work nor a DMA overlap
            # window. Reserving blocks for them can starve remote-KV loads.
            if ready and not self._full_prompt_fits(ready[0]):
                return 0
        seen = set()
        total = 0
        for request in (*self.running, *self.waiting, *self.skipped_waiting):
            if request.request_id in seen or request.status in (
                    RequestStatus.WAITING_FOR_REMOTE_KVS,
                    RequestStatus.WAITING_FOR_STREAMING_REQ):
                continue
            seen.add(request.request_id)
            remaining = max(0, request.num_tokens - request.num_computed_tokens)
            if request.infercept_compute_limit is not None:
                remaining = min(remaining, max(
                    0, request.infercept_compute_limit - request.num_computed_tokens))
            total += remaining
            if total >= self.profile.S:
                return self.profile.S
        return min(total, self.profile.S, self.max_num_scheduled_tokens)

    def _park_for_load(self, request):
        """Hold a request off the batch until its host chunks are restored.

        A running request leaves the batch the way vLLM pauses a streaming
        request: it is removed from running and queued as blocked. It later
        re-enters through the new-request path, which sends the worker the
        full block table and trusts the request's computed-token count.
        """
        if request.status == RequestStatus.RUNNING:
            self.running.remove(request)
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        if request not in self.waiting and request not in self.skipped_waiting:
            self._enqueue_waiting_request(request)

    def _prepare_resumed(self):
        """Advance chunked recomputation and expose requests ready to restore.

        The status of a running request is never rewritten here: vLLM moves a
        stopping request out of the batch only when it was RUNNING, so a
        stale status would leave a paused request in the batch.
        """
        running_group = sum(
            (not request.is_prefill_chunk
             and request.num_tokens - request.num_computed_tokens == 1)
            if self.paper_scheduling else request.num_computed_tokens >= request.num_tokens
            for request in self.running)
        recompute_chunk = max(self.profile.S - running_group, 1)
        for rid in tuple(self._resumed):
            state = self.residency.states.get(rid)
            if state is None:
                self._resumed.discard(rid)
                continue
            request = state.request
            pending = self.residency.pending
            if pending is not None and pending.contains(state):
                self._park_for_load(request)
                continue
            boundary = min(state.cpu) * self.residency.block_size if state.cpu else None
            if state.discarded and request.num_computed_tokens < state.computed_tokens:
                if boundary is not None and request.num_computed_tokens >= boundary:
                    request.infercept_compute_limit = boundary
                    self._park_for_load(request)
                else:
                    limit = min(state.computed_tokens,
                                request.num_computed_tokens + recompute_chunk)
                    if boundary is not None:
                        limit = min(limit, boundary)
                    elif (request.num_computed_tokens == 0
                          and self.kv_cache_manager.enable_caching
                          and not request.skip_reading_prefix_cache):
                        # A new prefix hit can already cover the recomputation
                        # limit. Leave room for one uncached token so native
                        # admission can adopt that prefix and make progress.
                        # Probe without recording a second cache query; native
                        # scheduling performs the authoritative lookup/charge.
                        _, cached = self.kv_cache_manager.coordinator.find_longest_cache_hit(
                            request.block_hashes, request.num_tokens - 1)
                        limit = max(limit, cached + 1)
                    request.infercept_compute_limit = limit
                    if request.status != RequestStatus.RUNNING:
                        request.status = RequestStatus.WAITING
                continue
            if state.cpu:
                request.infercept_compute_limit = boundary
                self._park_for_load(request)
                continue
            self.residency.resume(rid)
            if request.status != RequestStatus.RUNNING:
                request.status = RequestStatus.WAITING
            self._resumed.discard(rid)

    def _plan_transfers(self, now):
        paused = [state for state in self.residency.states.values()
                  if state.request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
                  and self._resident_blocks(state)]
        scored = sorted(
            ((self._waste(state, now), state) for state in paused),
            key=lambda item: (-item[0][0], item[1].request.arrival_time,
                              item[1].request.request_id))
        resumable = sorted(
            (self.residency.states[rid] for rid in self._resumed
             if rid in self.residency.states
             and self.residency.states[rid].cpu
             and self.residency.states[rid].request.status
             == RequestStatus.WAITING_FOR_REMOTE_KVS),
            key=lambda state: (state.request.arrival_time,
                               state.request.request_id))

        def loadable(state):
            start = self._resident_blocks(state)
            count = 0
            while start + count in state.cpu:
                count += 1
            return count

        store_demand = min(
            sum(self._resident_blocks(state) for _, state in scored),
            self.connector.scratch_blocks)
        load_demand = sum(loadable(state) for state in resumable)
        if not store_demand and not load_demand:
            # With no swap plan pending, apply the dynamic preserve/discard
            # decision immediately to every intercepted resident context.
            for (_, preserve, discard), state in scored:
                if preserve <= discard:
                    self.infercept_stats['preserve_decisions'] += 1
                else:
                    self.residency.discard_prefix(state.request.request_id)
                    self.infercept_stats['discard_decisions'] += 1
            return

        batch_tokens = self._anticipated_forward_tokens()
        if batch_tokens:
            limit = measured_swap_limit(
                t_fwd_s(self.profile, batch_tokens), self.bytes_per_block,
                self.bandwidth_bytes_s)
        elif load_demand:
            # Only paused and returning requests remain: no forward pass can hide
            # a transfer, and the paper's budget is zero. A returning request
            # would then never get its host chunks back, so move one explicit
            # unhidden block per iteration and report it separately. An idle
            # system with nothing to restore plans no transfer at all.
            limit = 1
        else:
            limit = 0
        # Split the measured link budget before applying staging/demand caps.
        # store_demand already bounds outgoing copies by scratch capacity;
        # shrinking the link budget first would distort the free-GPU balance.
        limit = max(0, limit)
        pool = self.kv_cache_manager.block_pool
        # Do not reserve capacity for nonexistent forward work: that can
        # starve the sole returning request even with free GPU blocks.
        new_demand = math.ceil(min(
            batch_tokens, self.profile.S, self.max_num_scheduled_tokens)
            / self.residency.block_size)
        budget = plan_swap_budget(
            limit, pool.get_num_free_blocks(), len(self.residency.free_cpu),
            load_demand, store_demand, new_demand)
        loads = self._take_chunks(resumable, budget.load_blocks, loadable)
        stores = self._take_chunks(
            (state for _, state in scored), budget.store_blocks,
            self._resident_blocks)
        storing = {rid for rid, _ in stores}
        discard_remainders = set()
        for (_, preserve, discard), state in scored:
            rid = state.request.request_id
            selected = next((count for selected_rid, count in stores
                             if selected_rid == rid), 0)
            remaining = self._resident_blocks(state) - selected
            if preserve <= discard:
                self.infercept_stats['preserve_decisions'] += 1
            elif rid in storing and remaining:
                discard_remainders.add(rid)
                self.infercept_stats['discard_decisions'] += 1
            elif rid not in storing:
                self.residency.discard_prefix(rid)
                self.infercept_stats['discard_decisions'] += 1
        if loads or stores:
            self.residency.transfer_for_iteration(
                stores=stores, loads=loads,
                discard_remainders=discard_remainders)
            self.infercept_stats['swap_plans'] += 1
            self.infercept_stats['planned_load_blocks'] += sum(n for _, n in loads)
            self.infercept_stats['planned_store_blocks'] += sum(n for _, n in stores)
            if not batch_tokens:
                self.infercept_stats['unhidden_transfer_blocks'] += sum(n for _, n in loads)

    def _requeue_partial_prefills(self):
        """Keep resident recovery work in the FCFS waiting cohort (paper 4.3).

        PREEMPTED is only the native worker protocol's cached-request envelope:
        no pages are freed, progress reset, or preemption counters incremented.
        The existing resident-requeue extension sends only newly added blocks.
        """
        resident = {rid for rid in self._policy_resident_requeues
                    if rid in self.requests
                    and self.requests[rid].num_computed_tokens > 0
                    and self.requests[rid].status in (
                        RequestStatus.WAITING, RequestStatus.PREEMPTED,
                        RequestStatus.RUNNING)}
        partial = [r for r in self.running
                   if r.is_prefill_chunk or r.num_tokens - r.num_computed_tokens > 1]
        for request in partial:
            self.running.remove(request)
            self.waiting.add_request(request)
            resident.add(request.request_id)
        queued = {r.request_id for r in (*self.waiting, *self.skipped_waiting)}
        resident.intersection_update(queued)
        for rid in resident:
            # _prepare_resumed may have made a resident request WAITING.
            # Never overwrite a DMA/tool wait retained from an earlier
            # partial prefill. The restore planner requires REMOTE_KVS;
            # native streaming accounting requires STREAMING_REQ.
            self.requests[rid].status = RequestStatus.PREEMPTED
        self._policy_resident_requeues = resident
        self._policy_resident_block_counts = {
            rid: tuple(len(ids) for ids in self.kv_cache_manager.get_block_ids(rid))
            for rid in resident}

    def schedule(self):
        for request in self.requests.values():
            if (self._is_intercepted_session(request)
                    and request.request_id not in self._turn_kv):
                self._begin_kv_turn(request, 0)
        pending = self.residency.pending
        if not self.residency.finish():
            raise RuntimeError('previous InferCept transfer lacks an all-rank ACK')
        if pending is not None:
            for chunk in pending.chunks:
                metric = self._turn_kv.get(chunk.state.request.request_id)
                if not chunk.storing and not chunk.state.cancelled and metric is not None and not metric.complete:
                    for index in chunk.indices:
                        metric.restored.append((index * self.residency.block_size,
                            min((index + 1) * self.residency.block_size, chunk.state.computed_tokens)))
        for rid, update in tuple(self._pending_full_prompts.items()):
            del self._pending_full_prompts[rid]
            if rid in self.requests:
                self._update_request_as_session(self.requests[rid], update)
        self._prepare_resumed()
        if self.paper_scheduling:
            self._requeue_partial_prefills()
        self._relieve_waiting_ownership()
        self._plan_transfers(time.monotonic())
        before = {rid: r.num_computed_tokens for rid, r in self.requests.items()}
        configured_budget = self.max_num_scheduled_tokens
        if self.paper_scheduling:
            self.max_num_scheduled_tokens = min(configured_budget, self.profile.S)
        try:
            output = super().schedule()
        finally:
            self.max_num_scheduled_tokens = configured_budget
        for rid, count in output.num_scheduled_tokens.items():
            metric = self._turn_kv.get(rid)
            if metric is not None and not metric.complete:
                end = self.requests[rid].num_computed_tokens
                start = end - count
                if start > before[rid]:
                    metric.prefix.append((before[rid], start))
                metric.compute(start, end)
        self._last_scheduled_tokens = output.total_num_scheduled_tokens
        self.infercept_stats['iterations'] += 1
        now = time.monotonic()
        if now - self._progress_log_at >= 60:
            self._progress_log_at = now
            logging.getLogger(__name__).warning(
                'InferCept progress: iterations=%d completed_turns=%d running=%d '
                'waiting=%d free_blocks=%d scheduled_tokens=%d loaded_blocks=%d '
                'stored_blocks=%d transfer_pending=%s',
                self.infercept_stats['iterations'],
                sum(metric.complete for metric in self._turn_kv_history),
                len(self.running), len(self.waiting) + len(self.skipped_waiting),
                self.kv_cache_manager.block_pool.get_num_free_blocks(),
                output.total_num_scheduled_tokens, self.residency.stats['loaded_blocks'],
                self.residency.stats['stored_blocks'], self.residency.pending is not None)
            if not output.total_num_scheduled_tokens:
                for request in (*self.waiting, *self.skipped_waiting):
                    state = self.residency.states.get(request.request_id)
                    logging.getLogger(__name__).warning(
                        'InferCept waiting: request=%s status=%s computed=%d tokens=%d '
                        'compute_limit=%s gpu_blocks=%s cpu_chunks=%s resumed=%s '
                        'resident_requeue=%s full_prompt_fits=%s',
                        request.request_id, request.status.name,
                        request.num_computed_tokens, request.num_tokens,
                        request.infercept_compute_limit,
                        tuple(len(ids) for ids in self.kv_cache_manager.get_block_ids(
                            request.request_id)),
                        sorted(state.cpu) if state is not None else [],
                        request.request_id in self._resumed,
                        request.request_id in self._policy_resident_requeues,
                        self._full_prompt_fits(request))
        return output

    def _relieve_waiting_ownership(self):
        """Break idle FCFS admission deadlock from younger resumed owners."""
        if self.running or self.residency.pending is not None or not self.scheduler_reserve_full_isl:
            return
        ready = sorted((r for r in (*self.waiting, *self.skipped_waiting)
                        if r.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED,
                                        RequestStatus.WAITING_FOR_REMOTE_KVS)),
                       key=lambda r: (r.arrival_time, r.request_id))
        if not ready:
            return
        head = ready[0]
        manager = self.kv_cache_manager

        for victim in reversed(ready[1:]):
            if self._full_prompt_fits(head):
                break
            if not self.residency.single.req_to_blocks.get(victim.request_id):
                continue
            if victim.request_id in self.residency.states:
                # Discard only GPU ownership. Keep host chunks and the known
                # context boundary so recomputation stops before stored KV.
                self.residency.discard_prefix(victim.request_id)
            else:
                manager.free(victim)
            self.encoder_cache_manager.free(victim)
            victim.status = RequestStatus.PREEMPTED
            victim.num_computed_tokens = 0
            victim.spec_token_ids = []
            victim.num_preemptions += 1
            logging.getLogger(__name__).warning(
                'InferCept idle admission recovery: preempted waiting %s for %s; free GPU blocks=%d',
                victim.request_id, head.request_id, manager.block_pool.get_num_free_blocks())
            if self.log_stats:
                victim.record_event(EngineCoreEventType.PREEMPTED, time.monotonic())
            self.infercept_stats['waiting_ownership_preemptions'] += 1

    def _full_prompt_fits(self, request):
        """Read-only native allocator feasibility, including reusable blocks."""
        manager = self.kv_cache_manager
        cached = manager.empty_kv_cache_blocks.blocks
        computed = request.num_computed_tokens
        if computed == 0 and manager.enable_caching and not request.skip_reading_prefix_cache:
            cached, computed = manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1)
        end = min(request.num_tokens, self.max_model_len)
        needed = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=end,
            new_computed_blocks=cached, num_encoder_tokens=0,
            total_computed_tokens=computed, num_tokens_main_model=end)
        return needed <= manager.block_pool.get_num_free_blocks()

    def _begin_kv_turn(self, request, history):
        previous = self._turn_kv.get(request.request_id)
        turn = previous.turn + 1 if previous is not None else 0
        program = (request.sampling_params.extra_args or {}).get('program_id', request.request_id)
        metric = TurnKV(program, turn, request.num_prompt_tokens, history)
        self._turn_kv[request.request_id] = metric
        self._turn_kv_history.append(metric)

    def policy_metrics(self):
        return {
            **self.infercept_stats,
            **self.residency.stats,
            'paused_requests': len(self.residency.states),
            'resumed_requests': len(self._resumed),
            'cpu_blocks_used': self.residency.cpu_blocks - len(self.residency.free_cpu),
            'last_scheduled_tokens': self._last_scheduled_tokens,
            'turn_kv_measurements': [m.result() for m in self._turn_kv_history if m.complete],
        }
