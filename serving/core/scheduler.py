import bisect
from time import time
import csv
import os
import json
MB_TO_BYTE = 1024 * 1024

#: SIM_ADMIT_VALVE=0 disables breaking pins on the ordinary allocation path,
#: leaving only the stall-recovery valves. An ablation handle, not a tuning
#: knob: the path mirrors vLLM's get_new_blocks, which reclaims protected
#: blocks for any allocation the free queue cannot satisfy, so OFF is the less
#: faithful setting. It exists because the path is new (2026-09-15) and
#: accounts for a third of all forced reclaims, and seed's error moved
#: -4.5% -> -15.3% when it landed; turning it off says whether it is the cause.
_ADMIT_VALVE = os.environ.get("SIM_ADMIT_VALVE", "1") != "0"

from .request import *
from .utils import *
from .controller import *
from .memory_model import *
from .graph_generator import *
from .trace_generator import *
from .logger import print_markup, print_rule
from .pim_model import *
import numpy as np

# class that shedules request of astra-sim
class Scheduler:
    def __init__(self, model, node_id, instance_id, max_num_seqs, max_num_batched_tokens,
                 num_npus, tp_size, pp_size, npu_mem, cpu_mem,
                 start_npu, pd_type, fp, block_size, req_num,
                 prioritize_prefill, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, enable_chunked_prefill=False,
                 long_prefill_token_threshold=0, cxl_mem=0, ep_size=1, kv_cache_dtype='auto'):
        self.model = model
        self.max_model_len = None
        self.config = get_config(model)
        self.node_id = node_id
        self.instance_id = instance_id
        self._schedule_trace = None
        if os.environ.get('SIM_SCHEDULE_TRACE_DIR'):
            from runtime.schedule_trace import ScheduleTrace
            self._schedule_trace = ScheduleTrace(os.environ['SIM_SCHEDULE_TRACE_DIR'],
                                                 f'sim-{node_id}-{instance_id}')
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = min(max_num_batched_tokens, self.config['max_position_embeddings'])
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.req_num = req_num
        self.start_npu = start_npu
        self.pd_type = pd_type
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefix_storage = prefix_storage
        self.prioritize_prefill = prioritize_prefill
        # Scheduling knob (unified_policy.py): "fcfs" (stock) or
        # "priority" (waiting queue ordered by harness-stamped priority).
        self.scheduling_policy = "fcfs"
        # Wider scheduling hooks (unified_policy.UnifiedPolicyAdapter):
        # victim rule and admission gate; consulted only when the adapter
        # reports custom_hooks, so stock and published values are unchanged.
        self.policy_hooks = None
        # Set by serving/__main__.py from --cleanup-inputs: delete a batch's
        # Chakra workload files as soon as the batch is done on all NPUs.
        self.cleanup_et = False
        # Recompute preemptions performed (reported at end of run; the real
        # bench records the engine's count in timeseries.csv for comparison).
        self.num_preemptions = 0
        self.preemption_log = []  # see __main__ for the column list
        self._admit_counter = 0
        self._none_reason = None  # why the last schedule() returned no batch (stuck diagnostics)
        self._preempt_counter = 0
        self._preempted_this_step = False
        # Preemption-storm guard (search hardening): with
        # SIM_PREEMPT_STORM_LIMIT=N set, a request preempted more than N
        # times raises instead of livelocking until the cell timeout. Off
        # (0) by default so published-policy runs are untouched.
        self._storm_limit = int(os.environ.get("SIM_PREEMPT_STORM_LIMIT", "0") or 0)
        self._preempt_count_by_req = {}
        # lists are sorted in arrival time manner
        self.request = []
        self.inflight = []
        self.done = []
        self.batch_ids = -1

        # memory model
        self.memory = MemoryModel(model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem, ep_size=ep_size, pp_size=pp_size, kv_cache_dtype=kv_cache_dtype)

        # logger
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)
    
 

    def _running_order_key(self, req):
        """Match the native running list, including InferCept session FCFS.

        InferceptSessionScheduler re-sorts RUNNING by original arrival each
        iteration. Re-admission after a tool call or preemption must not make
        an older program the newest running victim. Stock keeps admission order.
        """
        if req.infercept_session and self.scheduling_policy != "priority":
            return (req.queue_arrival, req.id)
        return (req.admit_seq, req.id)

    def _hook_victim(self, candidates, current):
        """Policy-chosen preemption victim (wider scheduling hooks), or
        None for the engine's own rule."""
        hooks = self.policy_hooks
        if hooks is None or not getattr(hooks, "custom_hooks", False):
            return None
        return hooks.select_victim(candidates, current)

    def schedule(self, current, sys, batch_id=-1):
        trace = self._schedule_trace if sys == self.start_npu else None
        if trace is not None:
            from runtime.schedule_trace import simulator_state
            before = simulator_state(self)
        self.memory.sim_now = current      # stamp for the D1 evict trace
        if self.enable_prefix_caching:
            batch = self.schedule_with_prefix(current, sys, batch_id)
            if (batch is None and self.memory.host_swap is not None
                    and sys == self.start_npu and not self.inflight):
                # There is no forward work to hide a store. Re-evaluate
                # paused contexts and retry if discarding one freed capacity.
                released = self.memory.host_swap.reconsider(
                    current, [r for r in self.request if r.arrival <= current])
                if released:
                    batch = self.schedule_with_prefix(current, sys, batch_id)
        else:
            batch = self.schedule_base(current, sys, batch_id)
        if trace is not None:
            trace.write(plane='sim', time_ns=current, before=before,
                        after=simulator_state(self),
                        scheduled_tokens=batch.scheduled_tokens if batch is not None else {},
                        batch_id=batch.batch_id if batch is not None else None,
                        no_batch_reason=self._none_reason if batch is None else None,
                        scheduled_requests=[] if batch is None else [dict(
                            id=r.id, program=r.session_id, turn=r.sub_request_index,
                            computed=r.num_computed_tokens, prompt_tokens=r.submitted_input,
                            recompute_end=r.original_input, first_token_counted=r.first_token_counted)
                            for r in batch.requests],
                        batch_work=None if batch is None else dict(total_tokens=batch.total_len,
                            prefill_queries=batch.prefill_q_list, prefill_contexts=batch.prefill_k_list,
                            decode_contexts=batch.decode_k_list, transfer_bytes=batch.load)
                        )
        return batch

    def _get_reload_size(self, batch_req, batch_len):
        load_size = 0
        for req in batch_req[:batch_len]:
            if req.evict:
                load_size += self.memory.get_evict_kv(req)
        return load_size

    # batch the request scheduling method
    def schedule_base(self, current, sys, batch_id=-1):
        # first NPU to process new batch
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.pp_size:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]

            # max_num_seqs limits total running requests (vLLM behavior)
            running_reqs = sum(len(b.requests) for b in self.inflight)
            available_slots = max(0, int(self.max_num_seqs) - running_reqs)
            batch_len = min(len(batch_req), available_slots)

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            kv_size = 0
            evict_size = 0

            # Get decode requests for preemption decisions
            gen_req = [req for req in batch_req if not req.is_prefill()]
            
            if self.prioritize_prefill and not self.enable_chunked_prefill:
                prefill_req = [req for req in batch_req if req.is_prefill()]

                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = min(len(batch_req), available_slots)
                    batch_req = batch_req[:batch_len]
            
            # Chunked prefill: process decode requests first, then prefill requests
            if self.enable_chunked_prefill:
                prefills = [req for req in batch_req if req.is_prefill()]
                decodes = [req for req in batch_req if not req.is_prefill()]
                batch_req = decodes + prefills
                batch_len = len(batch_req)
            
            # ============ STEP 1: Token budget allocation (FIRST) ============
            # Build scheduled_tokens dict: req.id -> tokens to process this step
            scheduled_tokens = {}
            
            if self.enable_chunked_prefill:
                # vLLM-style chunked prefill: schedule running (decode + ongoing prefill)
                # first, then waiting (new prefill) requests. Token budget is the main
                # constraint; long_prefill_token_threshold caps per-request tokens per step.
                token_budget = self.max_num_batched_tokens
                new_batch_req = []
                threshold = self.long_prefill_token_threshold
                # Decode requests first (each decode request = 1 token)
                for req in batch_req:
                    if not req.is_prefill():
                        if token_budget <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                # Then prefill requests (chunked)
                for req in batch_req:
                    if req.is_prefill():
                        if token_budget <= 0:
                            break
                        remaining = req.original_input - req.num_computed_tokens
                        # Per-request cap: long_prefill_token_threshold
                        if 0 < threshold < remaining:
                            remaining = threshold
                        chunk = min(remaining, token_budget)
                        if chunk <= 0:
                            break
                        req.chunk_len = chunk
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = chunk
                        token_budget -= chunk
                batch_req = new_batch_req
                batch_len = len(batch_req)

            else:
                # Non-chunked: compute scheduled tokens for each request
                total_len = 0
                for req in batch_req:
                    if req.is_prefill():
                        scheduled_tokens[req.id] = req.input
                        total_len += req.input
                    else:
                        scheduled_tokens[req.id] = 1
                        total_len += 1

                while total_len > self.max_num_batched_tokens:
                    # print(f"[NON_CHUNKED] total_len({total_len} = sum([req 0 ~ {batch_len - 1}])) exceed 'max_num_batched_tokens'")
                    last_req = batch_req[-1]
                    total_len -= scheduled_tokens[last_req.id]
                    del scheduled_tokens[last_req.id]
                    batch_req = batch_req[:-1]
                    batch_len -= 1
                
                # DEBUG: Check if total_len reached max
                # if total_len >= self.max_num_batched_tokens * 0.9:
                #     print(f"[NON-CHUNKED] Near max tokens! total_len: {total_len}/{self.max_num_batched_tokens}")
                #     print(f"              Batch: {batch_len} reqs, scheduled_tokens: {scheduled_tokens}")
            
                # Early return due to max_num_batched_tokens limitation (It occurs only when No chunked-prefill)
                if batch_len == 0:
                    print("     [WARNNING] Cannot load the request to batch due to max_num_batched_tokens limitation")
                    return None
            # ============ STEP 2: KV size calculation (with scheduled_tokens) ============
            temp_len = batch_len
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                load_size = self._get_reload_size(batch_req, i)
                if self.memory.is_avail(kv_size + load_size, Device.NPU):
                    temp_len = i
                    break
            
            # ============ STEP 3: Eviction if needed ============
            while temp_len == 0:
                # print("Evict Request to CPU due to memory limitation")
                # preempt request one by one until there is enough space
                if len(gen_req) == 0:
                    return None
                
                # check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue

                # else
                req_to_evict = gen_req[-1]
                evicted_kv_size = self.memory.get_evict_kv(req_to_evict)
                evict_size += evicted_kv_size
                req_to_evict.evict = True
                self.logger.info("Eviction of the request #%d", req_to_evict.id)
                gen_req = gen_req[:-1]
                # spill to cpu (host) memory. get_evict_kv returns per-rank
                # bytes; cpu_used is tracked in full-cluster bytes (matches
                # MemoryModel.apply_kv_cache_events convention), so scale by
                # num_npus when crossing the NPU->CPU boundary.
                self.memory.free(evicted_kv_size, Device.NPU)
                self.memory.allocate(evicted_kv_size * self.num_npus, Device.CPU)

                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)

                # check if can batch
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                    load_size = self._get_reload_size(batch_req, i)
                    if self.memory.is_avail(kv_size + load_size, Device.NPU):
                        temp_len = i
                        break

            batch_len = temp_len
            batch_req = batch_req[:batch_len]

            # Recompute kv_size for final batch
            kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            load_size = self._get_reload_size(batch_req, batch_len)

            # delete from request queue
            for req in batch_req:
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                if req.evict:
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            # ============ STEP 4: Allocate memory ============
            if kv_size > 0:
                self.memory.allocate(kv_size, Device.NPU)

            # Reload evicted KV to NPU and remove the spilled copy from CPU.
            # load_size is per-rank, cpu_used is full-cluster.
            if load_size > 0:
                self.memory.allocate(load_size, Device.NPU)
                self.memory.free(load_size * self.num_npus, Device.CPU)
            
            # ============ STEP 5: Build batch with lists ============
            total_len = 0
            kv_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            for req in batch_req:
                if req.is_prefill():
                    # Use scheduled_tokens for chunk size
                    chunk_size = scheduled_tokens.get(req.id, req.original_input - req.num_computed_tokens)

                    total_len += chunk_size
                    if req.is_init:  # Only set queuing delay on first chunk
                        req.set_que_delay(current)
                        if req.first_sched_ts < 0:
                            req.first_sched_ts = current
                        if req.first_cache_hit < 0:
                            req.first_cache_hit = req.npu_cache_hit
                    q_list.append(chunk_size)
                    prefill_q_list.append(chunk_size)
                    # prefill_k_list: already computed tokens (k_cache from previous chunks)
                    prefill_k_list.append(req.num_computed_tokens)
                    # k_list: total kv cache after this step (computed + new)
                    # k_list.append(req.num_computed_tokens + chunk_size)
                    num_prefill += 1

                else:
                    # Decode
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.num_computed_tokens
                    decode_k_list.append(req.num_computed_tokens)
                    # k_list.append(req.num_computed_tokens)

            # make batch, output doesn't matter here!! always one iteration
            # batch is also 1
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, load_size)
            # add already fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            # print(f"[BATCH DEBUG] Batch: {len(new_batch_req)} reqs, scheduled_tokens: {scheduled_tokens}")
            # batch.log()
            # add scheduled_tokens to batch for debugging
            batch.scheduled_tokens = scheduled_tokens
            return batch
        
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch == None:
                    return None
                # check if this has been runned in the system
                if sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
    
    def _running_step_kv(self, running):
        """KV the already-running set will take this step, in bytes.

        vLLM allocates a running request's blocks for the step inside
        schedule(), BEFORE the admission gate is consulted for any waiting
        request (vllm/v1/core/sched/scheduler.py: the running loop calls
        allocate_slots, then the waiting loop asks the gate), so the gate
        reads a free queue that already excludes them. This gate runs before
        the step's own reserve_kv, so without this it counts that space as
        free and admits turns the engine would have held.

        Sized the way the token-budget loop below will size it -- 1 token per
        decode, the chunk cap for an ongoing prefill -- because that is what
        reserve_kv will charge a few lines later.
        """
        if not running:
            return 0
        toks = {}
        threshold = self.long_prefill_token_threshold
        for r in running:
            if r.is_prefill():
                remaining = max(0, r.original_input - r.num_computed_tokens)
                if 0 < threshold < remaining:
                    remaining = threshold
                toks[r.id] = max(1, min(remaining, self.max_num_batched_tokens))
            else:
                toks[r.id] = 1
        return self.memory.get_block_kv(running, len(running), toks)

    def schedule_with_prefix(self, current, sys, batch_id=-1):
        self._current_ns = current
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and min(r.arrival for r in self.request) > current:
                self._none_reason = f"t={current} site=future_arrival min_arrival={min(r.arrival for r in self.request)}"
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.pp_size:
                self._none_reason = f"t={current} site=pp_inflight inflight={len(self.inflight)}"
                return None

            # scheduling start
            arrived = [req for req in self.request if req.arrival <= current]
            paper_budget = getattr(self.policy_hooks, 'infercept_paper_token_budget', None)
            if paper_budget is not None and not self.enable_chunked_prefill:
                raise ValueError('InferCept paper scheduling requires chunked prefill')

            # vLLM v1 order: RUNNING requests first, in admission order; then
            # WAITING requests follow the configured queue. FCFS prepends
            # preempted work; priority queues reinsert by priority and arrival
            # even after preemption. Priority never reorders running requests.
            # InferCept's native session scheduler instead re-sorts running
            # calls by their original program arrival on every iteration.
            running = sorted((r for r in arrived if r.admit_seq is not None),
                             key=self._running_order_key)
            waiting = [r for r in arrived if r.admit_seq is None]
            if paper_budget is not None:
                # Retain physical/token ownership, but partial prefills compete
                # with waiting work in original-session FCFS order each step.
                waiting.extend(r for r in running if r.is_prefill())
                running = [r for r in running if not r.is_prefill()]
            if self.scheduling_policy == "priority":
                wkey = lambda r: (r.priority, r.arrival, r.id)
            else:
                wkey = lambda r: ((1, r.queue_arrival, r.id) if r.infercept_session
                                  else (0, -r.preempt_seq, 0) if r.preempt_seq is not None
                                  else (1, r.arrival, r.id))
            waiting.sort(key=wkey)
            if (self.policy_hooks is not None
                    and getattr(self.policy_hooks, "custom_hooks", False)):
                # Admission gate (gateway-side hold): held turns stay in
                # the waiting queue with their place; see filter_waiting.
                waiting = self.policy_hooks.filter_waiting(
                    waiting, running, self.memory, len(self.inflight), current,
                    pending_reserve=self._running_step_kv(running), sched=self)
                apply_plan = getattr(self.policy_hooks, "apply_scheduling_plan", None)
                if apply_plan is not None:
                    running = apply_plan(self, running, current)

            # max_num_seqs limits total running requests (vLLM behavior)
            inflight_reqs = sum(len(b.requests) for b in self.inflight)
            available_slots = max(0, int(self.max_num_seqs) - inflight_reqs - len(running))
            batch_req = running + waiting[:available_slots]
            batch_len = len(batch_req)

            # nothing to batch
            if batch_len == 0:
                self._none_reason = (f"t={current} site=no_arrived pending={len(self.request)} "
                                     f"slots={available_slots}")
                return None

            # Prioritize prefill (without chunked prefill) or reorder for chunked prefill
            if self.prioritize_prefill and not self.enable_chunked_prefill:
                prefill_req = [req for req in batch_req if req.is_prefill()]
                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = min(len(batch_req), available_slots)
                    batch_req = batch_req[:batch_len]
            
            # Get decode requests for preemption decisions
            gen_req = [req for req in batch_req if not req.is_prefill()]
            # gen_req = [req for req in batch_req if not (req.num_computed_tokens >= req.original_input)]
            
            # ============ STEP 0: Prefix Matching ============
            # Only match prefix for NEW prefill requests (first chunk)
            # Ongoing chunked prefills already have their prefix cache info
            # for req in batch_req:
            #     if req.is_prefill():
            #         self.memory.prefix_match(req)
            
            # ============ STEP 1: Token budget allocation ============
            scheduled_tokens = {}
            
            if self.enable_chunked_prefill:
                # Chunked prefill: assign token budget to requests
                token_budget = self.max_num_batched_tokens
                if paper_budget is not None:
                    token_budget = min(token_budget, paper_budget)
                new_batch_req = []
                
                # vLLM traverses running requests in admission order, then
                # waiting requests. A running prefill consumes budget before
                # a later decode; do not regroup by phase.
                threshold = self.long_prefill_token_threshold
                for req in batch_req:
                    if token_budget <= 0:
                        break
                    if not req.is_prefill():
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                    else:
                        # Calculate remaining tokens without considering prefix cache
                        # because it is already considered in "self.memory.prefix_match(req)" -> req.num_computed_tokens
                        # Re-match at every scheduling attempt until the prefix is
                        # locked (vLLM looks computed blocks up at schedule time):
                        # a hit taken on an earlier attempt can be evicted while
                        # the request waits, and a stale hit makes the later
                        # insert re-create unreserved blocks (pool over-subscription).
                        if req.num_computed_tokens == 0 or (
                                not req._prefix_locked
                                and req.num_computed_tokens <= req.prefix_cache_hit):
                            req.num_computed_tokens = 0
                            self.memory.prefix_match(req)
                        remaining = req.original_input - req.num_computed_tokens
                        if req.infercept_recompute_end:
                            if req.num_computed_tokens >= req.infercept_recompute_end:
                                req.infercept_recompute_end = 0
                            else:
                                # Native _prepare_resumed bounds recovery of
                                # discarded history before processing new input.
                                # In synchronous vLLM the known sequence includes
                                # the sampled token, so its running_group test is
                                # zero and this limit is the measured profile S.
                                remaining = min(
                                    remaining, req.infercept_recompute_chunk,
                                    req.infercept_recompute_end - req.num_computed_tokens)
                        # Per-request cap: long_prefill_token_threshold
                        if 0 < threshold < remaining:
                            remaining = threshold
                        chunk = min(remaining, token_budget)
                        if chunk <= 0:
                            break

                        req.chunk_len = chunk
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = chunk
                        token_budget -= chunk

                batch_req = new_batch_req
                batch_len = len(batch_req)
            else:
                # Non-chunked: compute scheduled tokens for each request
                total_len = 0
                for req in batch_req:
                    if req.is_prefill():
                        if req.num_computed_tokens == 0 or (
                                not req._prefix_locked
                                and req.num_computed_tokens <= req.prefix_cache_hit):
                            req.num_computed_tokens = 0
                            self.memory.prefix_match(req)
                        # Consider prefix cache hit for non-chunked prefill
                        prefix_hit = req.prefix_cache_hit
                        tokens_to_compute = max(req.original_input - prefix_hit, 1)
                        scheduled_tokens[req.id] = tokens_to_compute
                        req.chunk_len = tokens_to_compute  # Set chunk_len for add_done()
                        total_len += tokens_to_compute
                    else:
                        scheduled_tokens[req.id] = 1
                        total_len += 1

                while total_len > self.max_num_batched_tokens:
                    last_req = batch_req[-1]
                    total_len -= scheduled_tokens[last_req.id]
                    del scheduled_tokens[last_req.id]
                    batch_req = batch_req[:-1]
                    batch_len -= 1
            
            # ============ STEP 1.5 + 2: Lock prefix and fit test, one request at a time ============
            # vLLM v1 walks the waiting queue in order: a request's cached
            # blocks are pinned only when IT is admitted, and the blocks it
            # needs may evict any unreferenced block, including the cached
            # prefix of a request further back in the queue. Locking the
            # prefix of every candidate first and then testing the fit
            # against what was left evictable starved admission under a deep
            # waiting queue: with dozens of waiting requests whose (mostly
            # cached) prefixes covered the whole pool, the union of their
            # locks left nothing evictable, the head request could not fit
            # its few uncached tokens, and the instance idled with a
            # 100%-evictable pool (flat SWE-bench open-loop trace, job
            # 40584882_0). Lock incrementally and test the cumulative fit
            # after each lock; per-request KV demand is non-negative, so the
            # first failure is the longest fitting prefix of batch_req.
            kv_size = 0
            evict_size = 0
            temp_len = 0
            # D1 diagnostic (inert unless SIM_FITCHECK_TRACE is set): snapshot the
            # usable terms BEFORE this loop locks any prefill's prefix hit, so a
            # preemption can be checked against what vLLM's allocate_slots would
            # have seen at the same instant.
            _fc_pre = None
            if os.environ.get("SIM_FITCHECK_TRACE"):
                _fc_pre = (self.memory.avail_size(Device.NPU),
                           self.memory.evictable_size(Device.NPU),
                           self.memory.parked_size(Device.NPU))
            for i, req in enumerate(batch_req):
                if req.is_prefill() and req.npu_last_node is not None and not req._prefix_locked:
                    self.memory.lock_prefix(req, Device.NPU)
                    req._prefix_locked = True
                # Parked entries may overlap running locks. A nominal pin
                # count is not allocatable space; reclaim, then measure it.
                total_useable_size = (self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU))
                kv_size = self.memory.get_block_kv(batch_req, i + 1, scheduled_tokens)
                if req.admit_seq is None and req.is_prefill():
                    # Mirror scheduler_reserve_full_isl: feasibility of the
                    # complete prompt, including protected blocks, is checked
                    # before allocating just this step's chunk. Earlier
                    # candidates still consume only their scheduled chunks.
                    full_tokens = dict(scheduled_tokens)
                    full_tokens[req.id] = max(
                        1, req.original_input - req.num_computed_tokens)
                    full_need = self.memory.get_block_kv(
                        batch_req, i + 1, full_tokens)
                    if full_need > (total_useable_size
                                    + self.memory.reclaimable_parked_size(Device.NPU)):
                        break

                if total_useable_size < kv_size:
                    # Mirror of vLLM's allocation, INCLUDING its refusal.
                    #
                    #   kv_cache_manager.allocate_slots:
                    #       if num_blocks_to_allocate > get_num_free_blocks():
                    #           return None            # no valve, no pins broken
                    #   block_pool.get_new_blocks:      # only reached otherwise
                    #       ret.extend(self._reclaim_protected(shortfall))
                    #
                    # and get_num_free_blocks() is `free_queue + len(_protected)`.
                    # So the engine breaks pins ONLY for an allocation that will
                    # then succeed; one that does not fit even counting protected
                    # blocks is declined untouched.
                    #
                    # This fired on the shortfall alone, without asking whether
                    # the request could fit at all -- exactly the case the engine
                    # declines -- so pins were destroyed for allocations that
                    # broke out two lines later regardless. Every symptom follows:
                    # forced reclaims ran +19..32% over hardware, expired
                    # collapsed 77% (pins killed before their TTL could elapse),
                    # and gate's error doubled while its hold count was unchanged
                    # -- the policy's decisions intact, the pins under them burned.
                    # Continuum was the pure case: 1,154,267 forced breaks, 100%
                    # of them here, against a real engine's 948,049.
                    # Reclaimable, not merely parked: a node a running
                    # request also holds has lock_ref >= 2, so dropping the pin
                    # frees nothing. Counting those admitted work the pool could
                    # not take and destroyed the pins on the way to refusing it.
                    parked = self.memory.reclaimable_parked_size(Device.NPU)
                    if (self.memory.kv_protection is not None and _ADMIT_VALVE
                            and total_useable_size + parked >= kv_size):
                        self.memory.kv_protection.ensure_evictable_tokens(
                            self.memory,
                            (kv_size - self.memory.avail_size(Device.NPU))
                            // max(1, self.memory._bytes_per_token),
                            site="admit")
                        total_useable_size = (
                            self.memory.avail_size(Device.NPU)
                            + self.memory.evictable_size(Device.NPU))
                    if total_useable_size < kv_size:
                        break
                temp_len = i + 1
            # Candidates behind the first non-fitting request were matched but
            # never locked: forget their (unpinned) hit so the next attempt
            # re-matches instead of trusting a prefix the LRU may evict.
            for req in batch_req[temp_len + 1:]:
                if req.is_prefill() and req.admit_seq is None and not req._prefix_locked:
                    self.memory.erase_prefix_info(req)
            
            # ============ STEP 3: Eviction if needed ============
            evicted_req = []
            self._preempted_this_step = False
            # vLLM v1 (scheduler.py schedule()): RUNNING requests are served
            # first; when one cannot allocate the blocks for its next chunk
            # or decode token, the scheduler preempts running[-1] (the most
            # last request in the native running order, or the lowest-priority one
            # under priority scheduling) and retries, until it fits or the
            # request preempts itself. WAITING requests never trigger
            # preemption: they are admitted only if they fit as-is. The
            # running set is the leading prefix of batch_req (native-ordered running
            # requests, then waiting requests).
            def _n_running():
                return sum(1 for r in batch_req if r.admit_seq is not None)
            while self.enable_prefix_caching and temp_len < _n_running():
                # Token-budget trimming only decides who gets work this step;
                # unscheduled RUNNING requests still own KV and remain native
                # preemption candidates. Using batch_req here could preempt
                # the requesting prefill while shielding later running owners.
                running_in = [r for r in running if r.admit_seq is not None]
                victim = self._hook_victim(running_in, current)
                if victim is not None:
                    pass
                elif self.scheduling_policy == "priority":
                    victim = max(running_in, key=lambda r: (r.priority, r.arrival, r.id))
                else:
                    victim = max(running_in, key=self._running_order_key)
                self.preemption_log.append((
                    current, victim.id, victim.num_computed_tokens,
                    max(0, victim.num_computed_tokens - victim.original_input),
                    kv_size, self.memory.avail_size(Device.NPU), self.memory.evictable_size(Device.NPU),
                    # D1: how fresh is the victim? admit_age 0 means it was
                    # admitted in this very step (admit-then-preempt churn).
                    victim.admit_seq if victim.admit_seq is not None else -1,
                    self._admit_counter, len(batch_req), temp_len,
                    victim.original_input, victim.npu_cache_hit))
                if _fc_pre is not None:
                    try:
                        _bpt = max(1, self.memory._bytes_per_token)
                        _post = (self.memory.avail_size(Device.NPU),
                                 self.memory.evictable_size(Device.NPU),
                                 self.memory.parked_size(Device.NPU))
                        with open(os.environ["SIM_FITCHECK_TRACE"], "a") as _f:
                            _f.write(json.dumps({
                                "t": current / 1e9, "victim": victim.id,
                                "pre_avail": _fc_pre[0] // _bpt, "pre_evict": _fc_pre[1] // _bpt,
                                "pre_parked": _fc_pre[2] // _bpt,
                                "post_avail": _post[0] // _bpt, "post_evict": _post[1] // _bpt,
                                "post_parked": _post[2] // _bpt,
                                "demand": kv_size // _bpt,
                                "n_running": _n_running(), "temp_len": temp_len,
                                "batch_len": len(batch_req)}) + "\n")
                    except Exception:
                        pass
                self._preempt_recompute(victim)
                self.num_preemptions += 1
                self._preempted_this_step = True
                self.logger.info("Preemption (recompute) of the request #%d", victim.id)
                gen_req = [r for r in gen_req if r is not victim]
                batch_req = [r for r in batch_req if r is not victim]
                batch_len = len(batch_req)
                current_usable_size = (self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU))
                temp_len = 0
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                    if current_usable_size >= kv_size:
                        temp_len = i
                        break
            if self.enable_prefix_caching and temp_len == 0:
                # Allocation-time reclamation above is the only valve.
                # A failed feasibility check must leave retention pins intact.
                valve_note = "valve=allocation_only"
                if temp_len == 0:
                    # Nothing to schedule this step (vLLM stops at the
                    # waiting queue head). Dropping an un-admitted prefill
                    # is not a preemption.
                    head = batch_req[0] if batch_req else None
                    head_note = "head=None"
                    if head is not None:
                        head_note = (
                            f"head=req{head.id} chunk={scheduled_tokens.get(head.id)} "
                            f"computed={head.num_computed_tokens} hit={head.npu_cache_hit} "
                            f"locked={head._prefix_locked} admit_seq={head.admit_seq} "
                            f"kv1={self.memory.get_block_kv(batch_req, 1, scheduled_tokens)}")
                    self._none_reason = (
                        f"t={current} site=valve_fail {head_note} {valve_note} "
                        f"avail={self.memory.avail_size(Device.NPU)} "
                        f"evictable={self.memory.evictable_size(Device.NPU)} "
                        f"parked_adm={self.memory.parked_size(Device.NPU)} "
                        f"batch_req={len(batch_req)} n_running={_n_running()} "
                        f"preempted={self._preempted_this_step} inflight={len(self.inflight)}")
                    for req in batch_req:
                        if req.is_prefill() and req._prefix_locked:
                            self._drop_prefill(req)
                    return None
            while temp_len == 0:
                # print("eviction occurs!!")
                if len(gen_req) == 0:
                    # print("gen_req length == 0 (No decode) => return None (No Batch)")
                    # No request to evict but no memory - rollback prefix cache lock
                    for req in batch_req:
                        if req.is_prefill() and req._prefix_locked:
                            self._drop_prefill(req)
                    return None
                
                # Check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue
                
                # Preempt like vLLM v1: under priority scheduling the running
                # request with the largest (priority, arrival) value, i.e. the
                # lowest priority (under PLAS: the program with the most
                # attained service); under FCFS the most recently started.
                victim = self._hook_victim(gen_req, current)
                if victim is not None:
                    pass
                elif self.scheduling_policy == "priority":
                    victim = max(gen_req, key=lambda r: (r.priority, r.arrival, r.id))
                else:
                    victim = gen_req[-1]
                if self.enable_prefix_caching:
                    self.preemption_log.append((
                        current, victim.id, victim.num_computed_tokens,
                        max(0, victim.num_computed_tokens - victim.original_input),
                        kv_size, self.memory.avail_size(Device.NPU), self.memory.evictable_size(Device.NPU),
                        victim.admit_seq if victim.admit_seq is not None else -1,
                        self._admit_counter, len(batch_req), -1,
                        victim.original_input, victim.npu_cache_hit))
                    # vLLM v1 preempts by RECOMPUTE: every block of the victim
                    # is released and it re-enters as a prefill of prompt +
                    # tokens generated so far (its resident prefix is picked
                    # up again by prefix_match if it survives in the LRU).
                    # Marking it evicted without unlocking its cached nodes
                    # (the swap path below) frees nothing under prefix
                    # caching, so the loop preempted every decode and the
                    # instance passed forever with a full pool.
                    self._preempt_recompute(victim)
                    self.num_preemptions += 1
                    self._preempted_this_step = True
                    self.logger.info("Preemption (recompute) of the request #%d", victim.id)
                else:
                    if victim.is_prefill() and getattr(victim, '_prefix_locked', False):
                        self.memory.unlock_prefix(victim, Device.NPU)
                        victim._prefix_locked = False
                    victim.evict = True
                    evicted_req.append(victim)
                    self.logger.info("Eviction of the request #%d", victim.id)
                gen_req = [r for r in gen_req if r is not victim]
                # batch_req is ordered decodes-then-prefills and is trimmed by
                # batch_len below, so the victim must leave it too (a
                # mid-list victim would otherwise stay while a later decode
                # is trimmed).
                batch_req = [r for r in batch_req if r is not victim]

                current_usable_size = (self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU))
                
                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)
                
                # Check if can batch now
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                    if current_usable_size >= kv_size:
                        temp_len = i
                        break

            # vLLM admits no WAITING request in a step that preempted.
            if self._preempted_this_step and temp_len > 0:
                keep = [r for r in batch_req[:temp_len] if r.admit_seq is not None]
                batch_req = keep + [r for r in batch_req if r not in keep]
                temp_len = len(keep)
                if temp_len == 0:
                    # Preempted everything and admitted nothing. vLLM would
                    # idle this step and retry next tick, but the simulator
                    # raises on an idle instance with no future arrival, so
                    # this must make progress in-call: break pins for the
                    # head's full remaining prefill (Continuum's running==0
                    # valve) and retry admission, deviating from the
                    # no-waiting-admission-after-preempt rule only in this
                    # otherwise-fatal state.
                    if (self.memory.kv_protection is not None
                            and not self.inflight and batch_req):
                        _head = batch_req[0]
                        _full_need = self.memory.get_kv(
                            max(1, _head.original_input - _head.num_computed_tokens))
                        _avail = self.memory.avail_size(Device.NPU)
                        if _full_need > _avail + self.memory.evictable_size(Device.NPU):
                            self.memory.kv_protection.ensure_evictable_tokens(
                                self.memory,
                                int(_full_need - _avail)
                                // max(1, self.memory._bytes_per_token) + self.memory.block_size,
                                site="head_full")
                        usable_now = (self.memory.avail_size(Device.NPU)
                                      + self.memory.evictable_size(Device.NPU))
                        for i in range(len(batch_req), 0, -1):
                            if usable_now >= self.memory.get_block_kv(
                                    batch_req, i, scheduled_tokens):
                                temp_len = i
                                break
                    if temp_len == 0:
                        # An empty batch would reach ASTRA-Sim as a
                        # zero-request trace. Unlock candidates and skip.
                        self._none_reason = (
                            f"t={current} site=preempt_guard_zero batch_req={len(batch_req)} "
                            f"avail={self.memory.avail_size(Device.NPU)} "
                            f"evictable={self.memory.evictable_size(Device.NPU)}")
                        for req in batch_req:
                            if req.is_prefill() and req._prefix_locked:
                                self._drop_prefill(req)
                        return None
            # Unlock prefix for requests that didn't make it into the batch
            for req in batch_req[temp_len:]:
                if req.is_prefill() and req._prefix_locked:
                    self._drop_prefill(req)

            batch_len = temp_len
            batch_req = batch_req[:batch_len]
            
            # Recompute kv_size for final batch
            kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            evict_size = (kv_size - self.memory.avail_size(Device.NPU)) if kv_size > self.memory.avail_size(Device.NPU) else 0
            
            if evict_size > 0:
                self.memory.evict_prefix_cache(evict_size, Device.NPU)
            # Hold the bytes this step computes until they are inserted into
            # the prefix cache (multi-chunk prefills insert only at the end).
            self.memory.reserve_kv(batch_req, scheduled_tokens)

            for req in batch_req:
                if req.admit_seq is None:
                    self._admit_counter += 1
                    req.admit_seq = self._admit_counter
                    req.infercept_retained_prefix = False
                    req.preempt_seq = None
                    # Queue-persistent retention (Continuum released-code
                    # semantics): the program's parked protection is
                    # released at batch admission, not at arrival.
                    if self.memory.kv_protection is not None:
                        self.memory.kv_protection.on_turn_admitted(req, current)

            # ============ STEP 4: Allocate memory & handle evicted requests ============
            evict_load_size = 0
            prefix_load_size = 0
            
            for req in batch_req:
                # Remove from request queue
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                # Load prefix cache from storage if needed
                if (req.is_prefill() and req.storage_cache_hit > req.npu_cache_hit
                        and not req.storage_restored):
                    prefix_load_size += (req.storage_cache_hit - req.npu_cache_hit) * self.memory.get_kv(1)
                    req.storage_restored = True

                # Handle evicted requests
                if req.evict:
                    self.memory.prefix_match(req)
                    self.memory.lock_prefix(req, Device.NPU)
                    if self.prefix_storage is not None:
                        self.memory.unlock_prefix(req, Device.CPU)
                    evict_load_size += self.memory.get_evict_kv(req)
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            # ============ STEP 5: Build batch with lists ============
            total_len = 0
            kv_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            
            # Evict storage prefix cache if needed
            total_size = 0
            for req in batch_req:
                total_size += self.memory.get_total_kv(req) * self.num_npus
            for req in evicted_req:
                total_size += self.memory.get_total_kv(req) * self.num_npus
            
            if self.prefix_storage is not None:
                storage_evict_size = (total_size - self.memory.avail_size(self.prefix_storage)) if total_size > self.memory.avail_size(self.prefix_storage) else 0
                if storage_evict_size > 0:
                    self.memory.evict_prefix_cache(storage_evict_size, self.prefix_storage)

            for req in batch_req:
                # Update the prefix cache for incoming batch
                # NOTE: Moved to add_done() to ensure prefix cache is updated after chunk computation
                # self.memory.cache_unfinished_req(req, Device.NPU)
                # if self.prefix_storage is not None:
                #     self.memory.cache_unfinished_req(req, self.prefix_storage)
                
                if req.is_prefill():
                    # Use scheduled_tokens for chunk size. num_computed_tokens
                    # already includes any prefix-cache hit (memory_model.py
                    # bumps it on first prefix_match), so chunk_size is already
                    # the count of tokens actually computed this iteration —
                    # no further prefix-hit subtraction is needed downstream.
                    chunk_size = scheduled_tokens.get(req.id, req.original_input - req.num_computed_tokens)
                    if chunk_size > self.max_num_batched_tokens:
                        raise Exception("Chunk length exceeds max num batched tokens")

                    total_len += chunk_size
                    if req.is_init:  # Only set queuing delay on first chunk
                        req.set_que_delay(current)
                        if req.first_sched_ts < 0:
                            req.first_sched_ts = current
                        if req.first_cache_hit < 0:
                            req.first_cache_hit = req.npu_cache_hit

                    q_list.append(chunk_size)
                    num_prefill += 1
                    prefill_q_list.append(chunk_size)
                    # prefill_k_list: already computed tokens (k_cache from previous chunks)
                    prefill_k_list.append(req.num_computed_tokens)
                else:
                    # Decode: use num_computed_tokens (inevitable modification)
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.num_computed_tokens  # inevitable modification: was req.input
                    decode_k_list.append(req.num_computed_tokens)  # inevitable modification: was req.input
                
                k_list.append(req.num_computed_tokens)  # inevitable modification: was req.input
            
            # Storage needs to hold evicted cache
            if self.prefix_storage is not None:
                for req in evicted_req:
                    self.memory.storage_cache_evicted_req(req)

            
            # For debugging
            # self.memory.npu_prefix_cache.pretty_print()
            # self.memory.npu_prefix_cache.print_prefix_info()
            self.memory.last_batch_tokens = total_len
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, evict_load_size + prefix_load_size)
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            if self.memory.host_swap is not None:
                self.memory.host_swap.plan(batch)
            _started = getattr(self.policy_hooks, "on_batch_started", None)
            if _started is not None:
                # Actual batch membership, for schedulers that own a queue
                # model (Autellix MLFQ). The plan is advisory; this is truth.
                _started(self.memory, batch_req, current)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            # print(f"[BATCH DEBUG] Batch: {len(new_batch_req)} reqs, scheduled_tokens: {scheduled_tokens}")
            batch.scheduled_tokens = scheduled_tokens
            # batch.log()
            return batch
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch is None or sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
        
    # pop inflight, add to done
    def _preempt_recompute(self, req):
        """Release everything a running request holds and restart it as a
        prefill whose prompt now includes the tokens it had generated
        (vLLM RECOMPUTE preemption). Its total length (req.output) and
        arrival are unchanged, so completion and JCT accounting are
        unaffected; the generated tokens are re-prefilled, not re-decoded."""
        # vLLM counts the token it just sampled: request.num_tokens is
        # prompt + output_token_ids, and the sampled token is appended before
        # the preemption, so after computing N tokens the sequence is N+1 long
        # and max_cache_hit_length is N. Without the +1 the recompute prompt
        # was one token short, the cap (original_input - 1) landed one below a
        # page boundary, and a 32-token cached context was floored to a
        # 16-token hit -- half the context re-prefilled for nothing.
        # Only in decode: a request preempted mid-prefill has sampled nothing.
        if req.num_computed_tokens >= req.original_input:
            generated = req.num_computed_tokens - req.original_input + 1
        else:
            generated = 0
        if self._storm_limit > 0:
            n = self._preempt_count_by_req.get(req.id, 0) + 1
            self._preempt_count_by_req[req.id] = n
            if n > self._storm_limit:
                raise RuntimeError(
                    f"preemption storm: request #{req.id} preempted {n} times "
                    f"(limit {self._storm_limit}); computed={req.num_computed_tokens} "
                    f"prompt={req.original_input} generated={generated} "
                    f"policy={self.scheduling_policy} hooks="
                    f"{getattr(self.policy_hooks, 'custom_hooks', None)}")
        if os.environ.get("AGS_DEBUG_PREEMPT"):
            print(f"[PREEMPT] t={getattr(self, '_current_ns', 0)/1e9:.3f} req={req.id} computed={req.num_computed_tokens} "
                  f"original_input={req.original_input} input={req.input} output={req.output} generated={generated} "
                  f"chunk_len={req.chunk_len} hit={req.prefix_cache_hit} admit_seq={req.admit_seq} n_pre={req.n_preempted}", flush=True)
        req.n_preempted += 1
        req.infercept_retained_prefix = False
        if req._prefix_locked and req.npu_last_node is not None:
            self.memory.unlock_prefix(req, Device.NPU)
        # vLLM RECOMPUTE frees the victim's blocks into the free queue tail-first
        # (kv_cache_manager.free: "so that the tail blocks are evicted first"),
        # where they stay hashed and re-hittable until other allocations consume
        # them. Leaving the chain cached-but-unlocked in the radix LRU (leaf-first
        # eviction) is the same thing; evicting it eagerly was tried (chain
        # eviction) and livelocked long prompts (every re-admission restarted
        # from zero), which the engine does not do.
        self.memory.erase_prefix_info(req)
        self.memory.release_kv_reservation(req)
        req.original_input += generated
        req.num_computed_tokens = 0
        req.chunk_len = 0
        req._prefix_locked = False
        req.evict = False
        req.admit_seq = None
        self._preempt_counter += 1
        req.preempt_seq = self._preempt_counter

    def _drop_prefill(self, req):
        """A prefill request that had a slot but does not fit in memory is
        taken out of the batch. vLLM's equivalent is preemption by
        recompute: the request gives back every block and restarts its
        prefill from scratch (its prefix hit is re-matched when it is next
        scheduled). Leaving computed-but-unlocked tokens behind would let
        LRU evict them while the request still counts them as computed,
        and its completion insert would re-create them unreserved; keeping
        them locked instead starves the running set under a small pool.
        Reservations for tokens that will not be inserted are dropped."""
        if req.infercept_retained_prefix and req.admit_seq is None:
            # A failed admission does not revoke session-owned KV. Explicit
            # pressure recovery/preemption is responsible for reclamation.
            return
        _computed_before = req.num_computed_tokens
        req.infercept_retained_prefix = False
        self.memory.unlock_prefix(req, Device.NPU)
        self.memory.erase_prefix_info(req)
        self.memory.release_kv_reservation(req)
        req.num_computed_tokens = 0
        req._prefix_locked = False
        if req.admit_seq is not None:
            # A started request losing its slot is a preemption: it re-enters
            # the waiting queue at the front (vLLM prepends preempted requests).
            req.n_preempted += 1
            self.preemption_log.append((
                getattr(self, "_current_ns", 0), req.id, _computed_before, 0, -1,
                self.memory.avail_size(Device.NPU), self.memory.evictable_size(Device.NPU),
                req.admit_seq if req.admit_seq is not None else -1,
                self._admit_counter, -1, -1,
                req.original_input, req.npu_cache_hit))
            req.admit_seq = None
            self._preempt_counter += 1
            req.preempt_seq = self._preempt_counter
            self.num_preemptions += 1
            self._preempted_this_step = True

    def _cache_unfinished_checked(self, req, batch):
        """Insert a completed prefill's context, preempting to make room.

        vLLM never reaches an insert that does not fit: it takes blocks
        incrementally inside schedule() and preempts the moment allocate_slots
        cannot satisfy one. This simulator charges the context at prefill
        completion instead, so a running set that each fit at admission can
        collectively outgrow the pool and leave nothing evictable -- every
        node locked by its own request. Measured on rtx70b_swe50_j0.02 with
        continuum: 18,447 MB held against an 18,099 MB pool, evictable zero,
        nine requests locked.

        When that happens, prefer another running request as the victim:
        lowest priority under priority scheduling, last in native running
        order under FCFS. Preempting releases its lock and reservation, allowing the
        pending cache events to be reconciled on retry. The completing
        request is the last resort. This completion-time recovery remains
        distinct from vLLM's allocation-time preemption.

        Publication timing remains a real structural difference -- this makes
        its consequence recoverable rather than fatal, it does not remove it.
        """
        for _ in range(len(batch.requests)):
            try:
                self.memory.cache_unfinished_req(req, Device.NPU)
                return
            except KVCapacityError:
                victims = [r for r in batch.requests
                           if r.admit_seq is not None and r is not req]
                if not victims:
                    victims = [r for r in batch.requests if r.admit_seq is not None]
                if not victims:
                    break
                if self.scheduling_policy == "priority":
                    victim = max(victims, key=lambda r: (r.priority, r.arrival, r.id))
                else:
                    victim = max(victims, key=self._running_order_key)
                self.logger.info(
                    "Insert for request #%d does not fit; preempting #%d",
                    req.id, victim.id)
                self._preempt_recompute(victim)
                self.num_preemptions += 1
                self._preempted_this_step = True
                if victim is req:
                    return
        try:
            self.memory.cache_unfinished_req(req, Device.NPU)
        except KVCapacityError as e:
            rows = [f"req{r.id}: input={r.original_input} computed={r.num_computed_tokens} "
                    f"hit={r.npu_cache_hit} out={r.output} reserved={r.kv_reserved / MB_TO_BYTE:.1f}MB "
                    f"locked={r._prefix_locked} evict={r.evict}" for r in batch.requests]
            raise RuntimeError(f"{e}\nfailing req={req.id}\nRUNNING SET ({len(batch.requests)}):\n  "
                               + "\n  ".join(rows)) from e

    def add_done(self, id, sys, finish):
        if self._schedule_trace is not None:
            self._schedule_trace.write(plane='sim', event='batch_completion',
                                       time_ns=finish, completion_id=id, system=sys)
        self.memory.sim_now = finish       # stamp for the D1 evict trace
        prompt_t = 0
        gen_t = 0
        end_reqs = []
        if len(self.inflight) == 0:
            return prompt_t, gen_t, end_reqs
        batch = None
        # find batch
        id -= 1
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        # no batch return
        if batch == None:
            return prompt_t, gen_t, end_reqs
        # already done
        if sys in batch.end:
            return prompt_t, gen_t, end_reqs
        else:
            # add to done system
            batch.end.append(sys)
            # check all npus are done
            if self.pd_type != "prefill":
                if self.start_npu not in batch.end or (self.start_npu + self.num_npus - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
            else:
                if self.start_npu not in batch.end or (self.start_npu + self.num_npus * 2 - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
        self.logger.info(
            "Batch #%d is done",
            batch.batch_id,
        )
        if self.memory.host_swap is not None:
            self.memory.host_swap.complete(batch)
        if self.cleanup_et:
            # Every NPU of the instance has finished this batch: its Chakra
            # workload files are dead. Deleting them here (rather than at
            # end of run) keeps the inputs root flat on tmpfs/-dev-shm.
            for _f in getattr(batch, "et_files", ()):
                try:
                    os.remove(_f)
                except FileNotFoundError:
                    pass
            batch.et_files = ()
                
        pool = []
        for req in batch.requests:
            # For chunked prefill, use computed tokens to determine prefill vs decode
            # Use is_prefill() method which checks num_computed_tokens < original_input
            is_prefill_req = req.is_prefill()
            
            # change phase
            if is_prefill_req:
                # Get chunk_len from scheduling step
                chunk_len = req.chunk_len if req.chunk_len > 0 else (req.original_input - req.num_computed_tokens)
                if chunk_len > self.max_num_batched_tokens:
                    raise Exception("Chunk length exceeds max num batched tokens")

                # Update num_computed_tokens
                if os.environ.get("AGS_DEBUG_PREEMPT") and req.n_preempted and (
                        req.num_computed_tokens + chunk_len > req.original_input):
                    print(f"[CHUNK_OVERSHOOT] t={finish/1e9:.3f} req={req.id} computed={req.num_computed_tokens} "
                          f"chunk_len={chunk_len} original_input={req.original_input} hit={req.prefix_cache_hit} "
                          f"batch={batch.batch_id} reqs={[r.id for r in batch.requests]}", flush=True)
                req.num_computed_tokens += chunk_len
                req.chunk_len = 0  # Reset for next step
                
                # Check if prefill is complete
                if req.num_computed_tokens >= req.original_input:
                    # Update prefix cache before clearing is_init (for stats tracking)
                    if self.enable_prefix_caching:
                        self._cache_unfinished_checked(req, batch)
                        if self.prefix_storage is not None:
                            self.memory.cache_unfinished_req(req, self.prefix_storage)
                    req.is_init = False
                    # Include prefix cache hit tokens in prompt throughput
                    prompt_t += chunk_len + req.prefix_cache_hit
                    req.set_ttft(finish)
                    
                    if self.pd_type == "prefill":
                        # Prefill instance: send to decode instance
                        self.logger.info("Request #%d is prefill done", req.id)
                        self.logger.info("Request #%d is sent to decode instance", req.id)
                        # req.num_computed_tokens += 1  # First decode token was generated
                        
                        # remove kv cache here
                        if self.enable_prefix_caching:
                            self.memory.unlock_prefix(req, Device.NPU)
                        else:
                            kv_size = self.memory.get_evict_kv(req)
                            self.memory.free(kv_size, Device.NPU)

                        end_reqs.append(req)
                        continue
                    else:
                        # Non-PD: prefill complete -- the last prefill token through
                        # lm_head produces an output token, and it is a real one on a
                        # RESUMED prefill too: _preempt_recompute grows the recompute
                        # prompt to include the token sampled before the preemption
                        # (vLLM's request.num_tokens), so the resumed pass ends one
                        # position further on and emits the next token, not a repeat.
                        # So it always counts. What differs is only whether it is the
                        # FIRST token: the first has no preceding token to measure
                        # from, every later one does -- and the interval of a resumed
                        # one is where the preemption stall shows up.
                        gen_t += 1
                        if req.first_token_counted:
                            req.add_itl(finish)
                        else:
                            req.first_token_counted = True
                        # req.num_computed_tokens += 1  # Count the first generated token
                        # req.set_ttft(finish)
                        # pool.append(req)
                        # continue
                else:
                    # Prefill not complete, return to pool for next chunk
                    prompt_t += chunk_len
                    # pool.append(req)
                    # continue
            else:
                # Decode phase
                if req.is_init:
                    # Full prefix cache hit: all input tokens were cached, so the
                    # request never entered the prefill-complete path where is_init
                    # is cleared. Lock the prefix node (was skipped because
                    # is_prefill() returned False during scheduling), count prefix
                    # stats once, then clear is_init.
                    if self.enable_prefix_caching:
                        if req.npu_last_node is not None and not req._prefix_locked:
                            self.memory.lock_prefix(req, Device.NPU)
                            req._prefix_locked = True
                        self.memory.cache_unfinished_req(req, Device.NPU)
                        if self.prefix_storage is not None:
                            self.memory.cache_unfinished_req(req, self.prefix_storage)
                    req.is_init = False
                    req.set_ttft(finish)
                    # Full prefix hit: count all cached tokens as prompt throughput
                    prompt_t += req.prefix_cache_hit
                gen_t += 1
                req.add_itl(finish)
                req.num_computed_tokens += 1

            # Update computed tokens for decode
            # req.num_computed_tokens += 1

            # check done
            if req.output <= req.num_computed_tokens + 1:
                # print("Request #{} is done".format(req.id))
                self.logger.info("Request #%d is done", req.id)
                # remove kv cache here
                if self.enable_prefix_caching:
                    self.memory.cache_finished_req(req, Device.NPU) # insert happens here
                    if self.prefix_storage is not None:
                        self.memory.cache_finished_req(req, Device.CPU)
                else:
                    kv_size = self.memory.get_evict_kv(req)
                    self.memory.free(kv_size, Device.NPU)
                req.add_latency(finish)
                self.done.append(req)
                end_reqs.append(req)

            # return to pool
            else:
                # print("Request #{} is not finished => go to pool".format(req.id))
                # Update prefix cache after chunk completion (moved from schedule_with_prefix())
                if self.enable_prefix_caching:
                    self.memory.cache_unfinished_req(req, Device.NPU)
                    if self.prefix_storage is not None:
                        self.memory.cache_unfinished_req(req, self.prefix_storage)
                pool.append(req)
        # return to request pool, both are already sorted with arrival_time
        if self.prioritize_prefill:
            self.request = self._merge_by_arrival_id(pool, self.request)
        else:
            self.request = pool + self.request
        _done = getattr(self.policy_hooks, "on_batch_done", None)
        if _done is not None:
            # The measured execution interval: what exhausts an Autellix
            # quantum and demotes the call. Duration, not wall position,
            # because the runtime accrues service per batch.
            _done(self.memory, [r.id for r in end_reqs], finish,
                  max(0, finish - batch.batch_time))
        del self.inflight[idx]
        del batch

        return prompt_t, gen_t, end_reqs
    

    ##### Helper Functions ######
    # get new batch id
    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    # ---------------------------------------------------------- reporting

    def return_prefix_info(self):
        """Prefix-cache hit accounting, asked of the scheduler rather than of
        its memory model, so the caller need not know which plane it has."""
        return self.memory.return_prefix_info()

    def mem_report(self):
        """Memory telemetry in host-neutral terms.

        Exists so the main loop can ask a scheduler how full it is without
        reaching through it into a particular memory implementation. The
        program-aware planes answer the same questions from a different
        structure; anything main reads through `.memory` directly cannot be
        answered by them.

        Additive: nothing here changes what this scheduler does.
        """
        mm = self.memory
        c = mm.npu_prefix_cache
        kv_span = max(1, mm.npu_mem - mm.weight)
        return {
            "waiting": len(self.request),
            "batched": sum(len(b.requests) for b in self.inflight),
            "referenced_tokens": sum(r.num_computed_tokens
                                     for b in self.inflight for r in b.requests),
            "cache_total": c.total_size(),
            "cache_evictable": c.evictable_size(),
            "cache_protected": c.protected_size(),
            "kv_util": (mm.npu_used - mm.weight) / kv_span,
            "reserved_tokens": int(mm.npu_reserved // max(1, mm._bytes_per_token)),
            "npu_used": mm.npu_used,
            "npu_mem": mm.npu_mem,
            "cpu_used": mm.cpu_used,
        }

    def teardown(self):
        """Release everything at end of run and report whether it all went
        back. Returns True when the instance is clean."""
        self.memory.free_prefix_cache()
        self.memory.free_weight()
        return self.memory.is_free()

    # add a request
    def add_request(self, req, is_init=True, workflow_id=None, node_id=None,
                    priority=None, session_id=None, sub_request_index=None):
        new_req = Request(*(req), is_init=is_init)
        if self.max_model_len is not None and new_req.output > self.max_model_len:
            raise ValueError(f'Request {new_req.id} context {new_req.output} exceeds '
                             f'max_model_len={self.max_model_len}')
        # Attach program identity: DAG workflows carry workflow_id/node_id,
        # agentic sessions carry session_id/sub_request_index. Consumed by
        # the policy adapter's request-side events (admission release,
        # victim rule, admission gate); None for flat workloads.
        new_req.workflow_id = workflow_id
        new_req.node_id = node_id
        new_req.session_id = session_id
        new_req.sub_request_index = sub_request_index
        bind = getattr(self.policy_hooks, 'bind_infercept_continuation', None)
        if bind is not None:
            bind(new_req, self.memory)
        if priority is not None:
            new_req.priority = priority
        # Maintain waiting-queue sort order (required by
        # schedule_base/schedule_with_prefix). Priority mode mirrors
        # vLLM's PriorityRequestQueue: smaller priority first,
        # arrival-time tiebreak.
        if self.scheduling_policy == "priority":
            key = lambda r: (r.priority, r.arrival, r.id)
        else:
            key = lambda r: (r.arrival, r.id)
        # bisect's `key` argument is 3.10+. The container runs 3.10 so a
        # simulation is fine, but the venv the test suite runs in is 3.9, where
        # this raised TypeError -- so the suite could not pass on the machine
        # people run it on, which is where regressions are supposed to be caught
        # before they cost a 50-minute job.
        k = key(new_req)
        lo, hi = 0, len(self.request)
        while lo < hi:
            mid = (lo + hi) // 2
            if k < key(self.request[mid]):
                hi = mid
            else:
                lo = mid + 1
        self.request.insert(lo, new_req)
        return
    
    # add decode request to decode instance from prefill instnace
    def add_decode(self, req):
        req.instance_id = self.instance_id
        self.request.append(req)
        if self.enable_prefix_caching:
            self.memory.prefix_match(req)
            kv_size = self.memory.get_evict_kv(req)
            evict_size = max(0, kv_size - self.memory.avail_size(Device.NPU))
            if evict_size > 0:
                self.memory.evict_prefix_cache(evict_size, Device.NPU)
            self.memory.cache_unfinished_req(req, Device.NPU)
        else:
            kv_size = self.memory.get_total_kv(req)
            self.memory.allocate(kv_size, Device.NPU)
    
    # get first request's arrival time
    def get_first_arrival_time(self):
        return self.first_arrival_time if self.first_arrival_time != 0 else 1 # need to add event handler at first
    
    # merge requests in the request pool, ensuring they are sorted by arrival time
    def _merge_by_arrival_id(self, left, right):
        if not left:  
            return right
        if not right: 
            return left

        # Fast path: if ranges don't overlap, just concatenate
        if (left[-1].arrival, left[-1].id) <= (right[0].arrival, right[0].id):
            return left + right
        if (right[-1].arrival, right[-1].id) <= (left[0].arrival, left[0].id):
            return right + left

        # General merge
        i = j = 0
        out = []
        while i < len(left) and j < len(right):
            li, rj = left[i], right[j]
            if (li.arrival, li.id) <= (rj.arrival, rj.id):
                out.append(li); i += 1
            else:
                out.append(rj); j += 1
        if i < len(left):  
            out.extend(left[i:])
        if j < len(right): 
            out.extend(right[j:])
        return out
    
    # print total system request metrics (TTFT, TPOT, ITL)
    def print_result(self):
        # Extract ttft, tpot, and itl values from the completed requests
        ttft_values = [req.ttft for req in self.done]
        tpot_values = [req.tpot for req in self.done]
        itl_values = [itl for req in self.done for itl in req.itl]

        def _render(title: str, values, num_space=0):
            print_rule(f"[sim.tagline]{title}[/]")
            if not values:
                print_markup(f"No {title.split()[0]} data available")
                return
            mean = np.mean(values) / 1_000_000
            median = np.median(values) / 1_000_000
            p99 = np.percentile(values, 99) / 1_000_000
            label = title.split()[-1] if title != "Time to First Token" else "TTFT"
            # Map to the metric short-name used in the detail rows.
            short = {
                "Time to First Token": "TTFT",
                "Time per Output Token (excl. 1st token)": "TPOT",
                "Inter-token Latency": "ITL",
            }[title]
            spacing = " " * num_space
            print_markup(f"Mean {short} (ms){spacing}:                                                     {mean:.2f}")
            print_markup(f"Median {short} (ms){spacing}:                                                   {median:.2f}")
            print_markup(f"P99 {short} (ms){spacing}:                                                      {p99:.2f}")

        _render("Time to First Token", ttft_values)
        _render("Time per Output Token (excl. 1st token)", tpot_values)
        _render("Inter-token Latency", itl_values, num_space=1)

    # print each request results
    def print_request_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # check all the request is done
    def is_request_empty(self):
        if len(self.request) == 0 and len(self.inflight) == 0:
            return True
        else:
            return False
        
    # save requests information to an output file
    def save_output(self, output_file, is_append=False):
        if not os.path.isabs(output_file):
            output_file = f'../{output_file}'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            if not is_append:
                writer.writerow(['instance id', 'request id', 'model', 'input', 'output', 
                                'arrival', 'end_time', 'latency', 
                                'queuing_delay', 'TTFT', 'TPOT', 'ITL',
                                'cache_hit', 'n_preempted',
                                'program_id', 'turn_idx', 'first_sched_ts',
                                'first_cache_hit'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.instance_id,
                    req.id,
                    req.model,
                    req.input,
                    req.output - req.input,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.ttft,
                    req.tpot,
                    req.itl
                ,
                    req.npu_cache_hit, req.n_preempted,
                    (req.session_id if req.session_id is not None
                     else req.workflow_id),
                    (req.sub_request_index if req.sub_request_index is not None
                     else req.node_id),
                    req.first_sched_ts,
                    req.first_cache_hit])


def main():
    pass

if __name__ == "__main__":
    main()
