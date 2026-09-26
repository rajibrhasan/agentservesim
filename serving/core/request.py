# class that manages request of astra-sim
class Request:
    def __init__(self, id, model, input, output, arrival, instance_id, input_hash_ids=None, output_hash_ids=None, is_init=True):
        self.id = id
        self.model = model
        self.input = input  # Always keep original input length
        self.output = output
        self.arrival = arrival
        # Release time controls readiness; session order survives tool calls.
        self.queue_arrival = arrival
        self.infercept_session = False
        # A paused session handed this GPU prefix to its returning turn.
        # Distinct from the speculative lock taken by an admission attempt.
        self.infercept_retained_prefix = False
        self.infercept_cpu_node = None
        # Previous-turn context being recovered under InferCept's measured
        # chunk limit. Clear once that context is rebuilt, before new prefill.
        self.infercept_recompute_end = 0
        self.infercept_recompute_chunk = 0
        self.instance_id = instance_id
        self.is_init = is_init
        # The prompt as submitted. _preempt_recompute grows this by the tokens
        # the request had generated, because a RECOMPUTE-preempted request
        # really does re-prefill them -- so it is the right quantity for
        # scheduling and for is_prefill(), and the WRONG one to report. The
        # submitted length is kept separately so metrics survive preemption.
        self.original_input = input
        self.submitted_input = input
        self.num_computed_tokens = 0  # Tracks actual computed tokens (vLLM style)
        self.evict = False
        self.end_time = -1
        self.latency = -1
        self.queuing_delay = -1
        # sim time of the FIRST schedule. queuing_delay is overwritten on every
        # prefill chunk (is_init only clears at prefill completion), so it
        # measures arrival -> LAST chunk; the prefix hit is fixed at the first
        # schedule, which is what the D1 exposure window needs.
        self.first_sched_ts = -1
        # Prefix hit as recorded at FIRST scheduling, mirroring vLLM's
        # Request.num_cached_tokens (scheduler.py: `if num_cached_tokens < 0`,
        # set once and never updated -- notably NOT after a preemption). The
        # sim re-matches at every scheduling attempt and overwrites
        # npu_cache_hit, so a preempted request would otherwise report a hit
        # that includes its own recomputed work.
        self.first_cache_hit = -1
        self.ttft = -1
        # The first output token is counted once, at the prefill completion
        # that produced it. A RECOMPUTE-preempted request completes a prefill
        # again on resume, which used to add a second 'first token' to the
        # generation counter -- a 33-token request reported 34 generated.
        self.first_token_counted = False
        self.tpot = -1
        self.itl = []
        self.recent_end = 0

        # For chunked prefill
        self.chunk_len = 0  # tokens scheduled for this request in the current step

        # For prefix caching modeling
        self.input_hash_ids = input_hash_ids
        self.output_hash_ids = output_hash_ids
        self.prefix_cache_hit = 0
        self.npu_cache_hit = 0
        # KV bytes reserved for computed-but-not-yet-cached tokens (see
        # MemoryModel.reserve_kv); released when the tokens are inserted.
        self.kv_reserved = 0
        # vLLM running/waiting bookkeeping: admit_seq is set when the request
        # is first scheduled (it is then RUNNING, served in admission order);
        # None means WAITING. preempt_seq orders preempted requests at the
        # front of the waiting queue (most recent first), like
        # vLLM's waiting.prepend_request.
        self.admit_seq = None
        self.preempt_seq = None
        self.n_preempted = 0
        self.storage_cache_hit = 0
        # True once the batch that loads this request's second-tier prefix hit
        # onto the NPU has been formed: until then the restored tokens count as
        # computed but are not yet resident, and KV sizing must reserve them.
        self.storage_restored = False
        # Set on a synthetic SAGA prefetch: a recompute issued ahead of a
        # tool result. It occupies the batch like any prefill but is never
        # a program turn, so metrics must skip it.
        self.prefetch_of = None
        self.npu_last_node = None
        self.cpu_last_node = None
        self.storage_last_node = None

        # For prefix cache lock tracking
        self._prefix_locked = False
        self._prefix_npu_stats_counted = False
        self._prefix_storage_stats_counted = False

        # For agentic session tracking (informational, does not drive scheduling)
        self.session_id = None
        self.sub_request_index = None

        # Scheduling knob (unified_policy.py): harness-stamped priority,
        # smaller runs first; 0 when unstamped (mirror of vLLM's
        # Request.priority). Only consulted when the scheduler runs in
        # priority mode.
        self.priority = 0

        # For multi-agent DAG tracking. workflow_id / node_id identify which
        # workflow and agent node this request is. critical_path_slack and
        # steps_to_execution are DAG signals consumed by the DAG-aware
        # scheduler (least-slack priority) and the KV cache manager
        # (who-runs-next eviction); set by the orchestrator, default-inert.
        self.workflow_id = None
        self.node_id = None
        self.critical_path_slack = None   # ns of slack before this call delays the workflow
        self.steps_to_execution = None    # DAG distance until this agent runs next

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def add_latency(self, end_time):
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        # submitted_input, not original_input: a preempted request re-prefills
        # its generated tokens, so original_input has grown. Reporting that as
        # the prompt shrank the implied output count and inflated tpot's
        # divisor -- a 16-token prompt with 33 generated tokens was reported as
        # 32 in and 17 out.
        self.input = self.submitted_input
        if self.output == self.input + 1:
            self.tpot = 0
        else:
            self.tpot = (self.latency - self.ttft) // (self.output - self.input - 1)
    
    def add_itl(self, current): # 
        self.itl.append(current - self.recent_end)
        self.recent_end = current

    def set_que_delay(self, current):
        self.queuing_delay = current - self.arrival
    
    def set_ttft(self, current):
        """Time to first token. Set once.

        The caller fires this at every prefill completion, and a
        RECOMPUTE-preempted request completes a prefill again on resume, so
        this used to be overwritten with the resume time -- reporting the
        stall as the TTFT (1 tick became 19 in a reproduction) and resetting
        the inter-token origin with it. vLLM records first-token time once.
        """
        if self.ttft < 0:
            self.ttft = current - self.arrival
            # Also the inter-token origin, and also once: re-basing it on a
            # resume would erase the preemption stall from the ITL series,
            # where a real engine's output stream plainly shows it.
            self.recent_end = current
    
    def log(self):
        print("         scheduled request : {}".format(self.__dict__))
    
    def is_prefill(self):
        """Check if request is still in prefill phase (has tokens left to compute)"""
        return self.num_computed_tokens < self.original_input

# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, batch_time, kv_size, evict=0, load=0):
        self.batch_id = batch_id
        self.model = model
        self.total_len = total_len
        self.kv_len = kv_len
        self.batch_time = batch_time
        self.fired = [] # systems that fired this batch
        self.requests = []
        self.end = []
        # vllm
        self.kv_size = kv_size
        self.evict = evict
        self.load = load
        self.host_store_bytes = 0
        self.host_link_bytes_s = None
        # for attn prediction
        self.q_list = q_list
        self.k_list = k_list
        self.num_prefill = num_prefill
        self.num_decode = num_decode
        self.prefill_q_list = prefill_q_list
        self.prefill_k_list = prefill_k_list
        self.decode_k_list = decode_k_list

        # for debugging
        self.scheduled_tokens = None
    def log(self):
        print("-------------------------Batch Log------------------------")
        for key in self.__dict__.keys():
            if key == 'requests':
                continue
            print("         {} : {}".format(key, self.__dict__[key]))
        for req in self.requests:
            req.log()
        print("----------------------------------------------------------")
