import os, threading, json
from .utils import get_config
from .radix_tree import *
import logging
from enum import Enum
from contextlib import contextmanager

GB_TO_BYTE = 1024 * 1024 * 1024
MB_TO_BYTE = 1024 * 1024
_KV_INVARIANT = os.environ.get("SIM_KV_INVARIANT", "0") != "0"
KB_TO_BYTE = 1024

class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3

class KVCapacityError(RuntimeError):
    """A cache publication needs more reclaimable NPU memory."""


class MemoryModel():
    def __init__(self, model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0, ep_size=1, pp_size=1, kv_cache_dtype='auto'):
        self.model = model
        self.node_id = node_id
        self.sim_now = 0          # sim time (ns); refreshed by the Scheduler
        self.instance_id = instance_id
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.npu_mem = npu_mem * GB_TO_BYTE # GB -> Byte
        self.cpu_mem = cpu_mem * GB_TO_BYTE # GB -> Byte
        self.cxl_mem = cxl_mem * GB_TO_BYTE
        self.block_size = block_size
        self.fp = fp // 8 # bit -> byte of floating point
        self.kv_fp = 1 if kv_cache_dtype == 'fp8' else self.fp  # KV cache bytes per element
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage

        self.config = get_config(model)
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.head_dim = self.config.get('head_dim', self.n_embd // self.n_head)
        self.kv_head = self.config.get("num_key_value_heads", self.n_head)  # fallback to n_head if not defined
        self.q_dim = self.n_head * self.head_dim       # total Q projection output dim
        self.kv_dim = self.kv_head * self.head_dim     # total KV projection output dim
        self.vocab_size = self.config['vocab_size']
        # Accept either the Mistral-style ``num_local_experts`` or the
        # HF/Qwen-style ``num_experts`` key — profiler configs track
        # upstream HF naming which varies per family.
        self.is_moe = 'num_local_experts' in self.config or 'num_experts' in self.config

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        # Memory model
        self.weight = self.get_weight() # assume weight is loaded
        self.npu_used = self.weight
        # Bytes reserved for in-flight tokens that have been scheduled (or
        # computed in an earlier chunk) but not yet inserted into the prefix
        # cache, where this model charges KV memory. vLLM allocates blocks
        # before compute; without this reservation a multi-chunk prefill
        # holds no accounted memory until its last chunk, admission overcommits
        # the pool, and the final insert cannot be satisfied.
        self.npu_reserved = 0
        self.cpu_used = 0
        self.cpu_swap_reserved = 0
        self.host_swap = None
        # Optional policy-supplied reclaim ranking (SAGA's workflow-aware
        # LRU). None keeps the tree's own LRU, which is what vLLM does.
        self.eviction_order = None
        # Second-tier prefix cache: configured by --prefix-storage for every
        # finished request, or enabled per policy as a swap tier (InferCept)
        # that only holds contexts a retention decision moved there.
        self.second_tier_prefix_cache = None
        self.swap_tier = False
        self.cpu_mem_bw_gbs = None   # host link the swap budget is sized from
        self.last_batch_tokens = 0   # tokens of the last batch this instance formed
        if self.weight > self.npu_mem:
            raise RuntimeError(f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: Model size {self.weight*self.num_npus//GB_TO_BYTE}GB exceeds total NPU memory {self.npu_mem*self.num_npus//GB_TO_BYTE}GB")

        if enable_prefix_caching:
            one_token_kv_size = self.get_kv(1)
            self.mem_for_kv = self.npu_mem - self.weight
            self.npu_prefix_cache = RadixCache(device='NPU', 
                                               node_id=self.node_id,
                                               instance_id=self.instance_id,
                                               page_size=self.block_size,
                                               capacity=self.mem_for_kv,
                                               kv_size=one_token_kv_size,
                                               enable_kv_cache_events=True,
                                                )
            # D1 diagnostic: attribute each evicted node to its program.
            # Inert unless SIM_EVICT_TRACE is set (see _evict_trace_hook).
            self.npu_prefix_cache.on_evict = self._evict_trace_hook
            if prefix_storage is not None:
                if enable_prefix_sharing and prefix_pool is not None:
                    self.second_tier_prefix_cache = prefix_pool
                else:
                    prefix_cache_capacity = 0
                    if prefix_storage == Device.CPU:
                        device = "CPU"
                        prefix_cache_capacity = self.cpu_mem
                    elif prefix_storage == Device.CXL:
                        device = "CXL"
                        prefix_cache_capacity = self.cxl_mem
                    else:
                        raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: Device {prefix_storage} is currently not supported as a second tier prefix cache storage")
                    # print("[instance {}] prefix_cache_capacity : {}".format(instance_id, prefix_cache_capacity // GB_TO_BYTE))
                    self.second_tier_prefix_cache = RadixCache(device=device, 
                                                    node_id=self.node_id,
                                                    instance_id=self.instance_id,
                                                    page_size=1,
                                                    capacity=prefix_cache_capacity,
                                                    kv_size=(one_token_kv_size * self.num_npus),
                                                    enable_kv_cache_events=True,
                                                    )
                
        # Hash id -> token length for corresponding prefix cache block
        self._npu_cache_hashtolen = {}
        self._cpu_cache_hashtolen = {}
        self._bytes_per_token = self.get_kv(1)  # bytes per token for kv cache

        # Retention knob (unified_policy.py): set to the adapter when a
        # retention policy runs; consulted by evict_prefix_cache.
        self.kv_protection = None
    def get_weight(self):
        """Per-GPU model weight in bytes.

        Conservative upper bound across PP ranks: assumes a single rank
        holds embedding + final_layernorm + lm_head along with its share
        of transformer blocks (n_layer // pp_size). In real PP these
        non-block weights live on the first/last rank only, so middle
        ranks are lighter — but using the heaviest-rank value here keeps
        the `weight > npu_mem` check safe.
        """
        tp = self.tp_size
        pp = max(self.pp_size, 1)
        ep = self.ep_size
        fp = self.fp
        weight = 0

        _, embedding, _ = calculate_sizes(self.model, 'embedding', 1, parallel=tp, fp=fp)
        weight += embedding
        weight += self._get_weight_per_block(tp, ep, fp) * (self.n_layer // pp)
        _, ln_f, _ = calculate_sizes(self.model, 'final_layernorm', 1, parallel=tp, fp=fp)
        weight += ln_f
        _, lm_head, _ = calculate_sizes(self.model, 'lm_head', 1, parallel=tp, fp=fp)
        weight += lm_head

        self.logger.info(
            "NPU: model weight %dMB loaded",
            weight * tp // MB_TO_BYTE,
        )
        return weight

    def _get_weight_per_block(self, tp, ep, fp):
        """Per-block weight: dense layers use TP, MoE experts use EP."""
        block_weight = 0
        _, ln_w, _ = calculate_sizes(self.model, 'layernorm', 1, parallel=tp, fp=fp)
        block_weight += ln_w  # input layernorm
        _, qkv_w, _ = calculate_sizes(self.model, 'qkv_proj', 1, parallel=tp, fp=fp)
        block_weight += qkv_w
        _, o_w, _ = calculate_sizes(self.model, 'o_proj', 1, parallel=tp, fp=fp)
        block_weight += o_w
        block_weight += ln_w  # post layernorm (same weight size)
        if self.is_moe:
            _, moe_w, _ = calculate_sizes(self.model, 'moe', 1, parallel=ep, fp=fp)
            block_weight += moe_w
        else:
            _, ffn1_w, _ = calculate_sizes(self.model, 'gate_up_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn1_w
            _, ffn2_w, _ = calculate_sizes(self.model, 'down_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn2_w
        return block_weight

    def set_kv_capacity_tokens(self, tokens):
        """Set a measured physical-block budget before admitting any work."""
        if not isinstance(tokens, int) or tokens <= 0 or tokens % self.block_size:
            raise ValueError('KV capacity must be a positive whole number of blocks')
        if self.npu_used != self.weight or self.npu_reserved:
            raise RuntimeError('Cannot change KV capacity after allocating requests')
        self.mem_for_kv = self.get_kv(tokens)
        self.npu_mem = self.weight + self.mem_for_kv
        if self.enable_prefix_caching:
            self.npu_prefix_cache.capacity = self.mem_for_kv

    def get_kv(self, seq):
        # shape of kv cache
        # (kv_head, batch_size, n_embd//n_head, seq_len) per layer
        # return batch_size = 1 to caclulate max batch_size in scheduler

        # K & V multiply 2. Cast to int: this is a byte count, and it is written
        # verbatim into ASTRA-Sim trace lines (kv_load/kv_evict) whose size field
        # the Chakra converter parses with int() — a float operand (e.g. num_npus)
        # would otherwise emit "34086912.0" and crash the converter.
        return int(2 * self.kv_dim * seq * self.n_layer * self.kv_fp // self.num_npus)
    
    # get the total size of current kv cache for the request
    # used when adding prefilled request to decode instance.
    def get_total_kv(self, req):
        # ceil division: (n + block_size - 1) // block_size
        num_blocks = (req.num_computed_tokens + self.block_size - 1) // self.block_size
        return self.get_kv(num_blocks * self.block_size)

    # get size of kv block that should be 'added'. including new init requests
    # also checks evicted request and include its kv cache
    # scheduled_tokens: dict mapping request id to number of tokens scheduled this step
    # 
    # vLLM-style cumulative allocation:
    #   blocks_after = ceil((computed + scheduled) / block_size)
    #   blocks_before = ceil(computed / block_size) if computed > 0 else 0
    #   new_blocks = blocks_after - blocks_before
    def get_block_kv(self, batch_req, batch_len, scheduled_tokens=None):
        # print("[get_block_kv] current batch_req length : {}".format(batch_len))
        block_kv_size = 0
        for i in range(batch_len):
            req = batch_req[i]
            if req.evict or req.is_prefill():
                # Prefill and reloaded decode requests may allocate newly
                # computed blocks. Existing evicted KV is reloaded separately
                # by Scheduler.load_size.
                hit = req.npu_cache_hit if self.enable_prefix_caching else 0
                
                if scheduled_tokens and req.id in scheduled_tokens:
                    tokens_this_step = scheduled_tokens[req.id]
                else:
                    raise RuntimeError("[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: scheduled_tokens cannot be None")
                
                # vLLM-style cumulative block allocation
                computed_before = req.num_computed_tokens
                
                total_after = computed_before + tokens_this_step
                # A second-tier (CPU) prefix hit is counted as computed so the
                # request skips its prefill, but those tokens are not on the
                # NPU until the batch that loads them runs: the pages they
                # occupy are materialized by this step and must be reserved
                # (board job 42522214_5: restored tokens were never reserved,
                # and the completing insert overflowed a full pool).
                if (req.storage_cache_hit > req.npu_cache_hit
                        and not req.storage_restored):
                    computed_before = min(computed_before, req.npu_cache_hit)
                
                # Calculate blocks needed (cumulative)
                # The prefix cache stores whole pages only (cache_unfinished_req
                # inserts the page-aligned prefix), so the bytes this step will
                # materialize are the pages completed by it: pages(after) minus
                # pages already stored (floor), not minus ceil. Using ceil here
                # under-reserved by one page whenever a chunk ended off-page
                # (token budget minus decode tokens), and the completing insert
                # then failed with a full pool.
                blocks_after = (total_after + self.block_size - 1) // self.block_size
                blocks_before = computed_before // self.block_size
                
                
                new_blocks = max(0, blocks_after - blocks_before)
                block_kv_size += self.get_kv(new_blocks * self.block_size)
                # print("[DEBUG] hit : {} | tokens_this_step : {} | computed_before : {} | total_after : {} | new_blocks : {} | block_kv_size : {}".format(
                #     hit, tokens_this_step, computed_before, total_after, new_blocks, block_kv_size
                # ))
            else:
                # Decode: use num_computed_tokens (or input for backwards compat)
                computed = req.num_computed_tokens
                # A decode token materializes a page when it completes one
                # (stored pages are page-aligned, see above).
                num_before = computed // self.block_size
                num_after = (computed + 1 + self.block_size - 1) // self.block_size
                if num_after > num_before: # difference of the block is maximum one block
                    block_kv_size += self.get_kv(self.block_size)
        return block_kv_size
    
    # get size of kv cache that should be evicted
    def get_evict_kv(self, req):
        evict_size = 0
        # Use num_computed_tokens if available, fallback to input for backwards compat
        computed = req.num_computed_tokens
        hit = req.npu_cache_hit if self.enable_prefix_caching else 0
        needed = max(0, computed - hit)
        # ceil division: (needed + block_size - 1) // block_size
        num_blocks = (needed + self.block_size - 1) // self.block_size
        evict_size += self.get_kv(num_blocks * self.block_size)
        return evict_size

    def parked_size(self, device):
        """Bytes of prefix cache parked by retention protections that
        ADMISSION may count as usable -- all of them.

        The engine counts parked blocks as free (BlockPool.get_num_free_blocks
        is `free_block_queue.num_free_blocks + len(self._protected)`) because
        get_new_blocks reclaims them on demand through the safety valve. It
        does not condition that on the retention policy, so neither does this.
        """
        if device != Device.NPU or self.kv_protection is None:
            return 0
        fn = getattr(self.kv_protection, "admission_parked_tokens",
                     self.kv_protection.parked_tokens)
        return fn(self) * self._bytes_per_token

    def reclaimable_parked_size(self, device):
        """Bytes a feasibility check may count on the retention valve to free.

        Not parked_size: that counts everything a pin covers, including nodes a
        running request also holds, which breaking the pin cannot free. A guard
        using it reclaims and then fails anyway -- pins destroyed for an
        allocation that was never possible.
        """
        if device != Device.NPU or self.kv_protection is None:
            return 0
        fn = getattr(self.kv_protection, "reclaimable_parked_tokens", None)
        if fn is None:
            return self.parked_size(device)
        return fn(self) * self._bytes_per_token

    def reserve_kv(self, batch_req, scheduled_tokens):
        """Reserve the KV bytes this step will compute for each request of
        the batch (block granularity, same sizing as get_block_kv). Returns
        the total reserved. No-op without prefix caching (then KV is
        allocated explicitly by the scheduler)."""
        if not self.enable_prefix_caching:
            return 0
        sizes = [(req, self.get_block_kv([req], 1, scheduled_tokens))
                 for req in batch_req]
        total = sum(size for _, size in sizes)
        shortfall = self.npu_used + self.npu_reserved + total - self.npu_mem
        if shortfall > 0:
            # Feasible first. vLLM declines an allocation that does not fit even
            # counting protected blocks (allocate_slots: `if num_blocks >
            # get_num_free_blocks(): return None`) and leaves them untouched;
            # only a reachable allocation reclaims. Reclaiming first destroyed
            # pins on the way to raising.
            if shortfall > (self.evictable_size(Device.NPU)
                            + self.reclaimable_parked_size(Device.NPU)):
                raise KVCapacityError(
                    "NPU batch reservation exceeds reclaimable capacity")
            self.evict_prefix_cache(shortfall, Device.NPU)
        if self.npu_used + self.npu_reserved + total > self.npu_mem:
            raise KVCapacityError("NPU batch reservation exceeds reclaimable capacity")
        # Reserve the entire batch or none of it. Parked tokens are only
        # capacity once their retention references have actually been dropped.
        for req, size in sizes:
            req.kv_reserved += size
        self.npu_reserved += total
        return total

    def release_kv_reservation(self, req):
        """The request's computed tokens are being inserted into the prefix
        cache (charged there); drop its reservation."""
        if req.kv_reserved:
            self.npu_reserved = max(0, self.npu_reserved - req.kv_reserved)
            req.kv_reserved = 0

    def free_weight(self):
        if self.npu_used - self.weight < 0:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id}, inst={self.instance_id}] NPU: tried to free model weight {self.weight / MB_TO_BYTE:.2f}MB "
                f"but only {self.npu_used / MB_TO_BYTE:.2f}MB is used."
            )
        self.logger.info(
            "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
            self.npu_used / MB_TO_BYTE,
            self.weight / MB_TO_BYTE,
            (self.npu_used - self.weight) / MB_TO_BYTE,
        )
        self.npu_used -= self.weight

    def is_free(self):
        is_free = self.npu_used == 0 and self.cpu_used == 0
        if not is_free:
            self.logger.error(
                "Memory leak detected (node %s inst %s): NPU used: %.2fMB, CPU used: %.2fMB",
                self.node_id,
                self.instance_id,
                self.npu_used / MB_TO_BYTE,
                self.cpu_used / MB_TO_BYTE,
            )
        return

    # -------------------- Memory Management --------------------
    

    #: SIM_KV_INVARIANT=1 checks, after every NPU allocate/free, that the two
    #: accounts of what is resident still agree:
    #:
    #:     radix tokens x bytes_per_token  ==  npu_used - weight
    #:
    #: NOT minus npu_reserved. A reservation is a separate account: reserve_kv
    #: only does `npu_reserved += size`, and it reduces avail_size so the space
    #: is not promised twice, but it never enters npu_used -- the tokens are
    #: charged there when they are INSERTED, which is also when the reservation
    #: is dropped. Subtracting it here manufactured a drift exactly equal to the
    #: outstanding reservation on the first step that had one.
    #:
    #: They are meant to be one fact kept in two places -- the tree knows which
    #: tokens are cached, npu_used knows how many bytes are spent -- and a run
    #: that drifts wedges later, somewhere unrelated, with a pool full of memory
    #: nothing will reclaim. Measured on rtx70b_swe50_j0.02 with continuum: the
    #: tree claimed 22,312 MB against 17,883 MB charged, and the wedge surfaced
    #: at whichever allocation happened to come next. Off by default; this is a
    #: per-operation check, not something to carry in a measured run.
    def _check_kv_invariant(self, where):
        if (not _KV_INVARIANT or not self.enable_prefix_caching
                or getattr(self, "_tearing_down", False)
                or getattr(self, "_recon_depth", 0) > 0):
            return
        tree = self.npu_prefix_cache.total_size() * self._bytes_per_token
        charged = self.npu_used - self.weight
        drift = tree - charged
        # Any divergence at all. A one-block tolerance hid the origin: the gap
        # compounds in sub-block increments at nearly every reconciliation and
        # had already reached 4,192 MB by the time a single step moved it
        # enough to trip. The first non-zero step is the one worth seeing.
        if abs(drift) > 0:
            self._kv_invariant_drift = getattr(self, "_kv_invariant_drift", 0)
            prev = self._kv_invariant_drift
            self._kv_invariant_drift = drift
            if drift != prev:
                raise RuntimeError(
                    f"[KV INVARIANT] drift opened at {where}: tree "
                    f"{tree / MB_TO_BYTE:.2f}MB vs charged {charged / MB_TO_BYTE:.2f}MB "
                    f"(npu_used {self.npu_used / MB_TO_BYTE:.2f} - weight "
                    f"{self.weight / MB_TO_BYTE:.2f}; reserved "
                    f"{self.npu_reserved / MB_TO_BYTE:.2f}, not part of it); "
                    f"this step moved it by "
                    f"{(drift - prev) / MB_TO_BYTE:+.2f}MB (was "
                    f"{prev / MB_TO_BYTE:+.2f}MB)")

    def allocate(self, size, device):
        if device == Device.NPU:
            if self.npu_used + size > self.npu_mem:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to load {size / MB_TO_BYTE:.2f}MB but only {(self.npu_mem - self.npu_used) / MB_TO_BYTE:.2f}MB is available."
                )
            self.logger.info(
                "NPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used + size) / MB_TO_BYTE,
            )
            self.npu_used += size
            self._check_kv_invariant("allocate")
        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.allocate(size)
            else:
                if self.cpu_used + size > self.cpu_mem:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to load {size / MB_TO_BYTE:.2f}MB "
                        f"but only {(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used + size) / MB_TO_BYTE,
                )
                self.cpu_used += size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.allocate(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to allocate KV cache in unsupported device {device}")
    
    def free(self, size, device):
        if device == Device.NPU:
            if self.npu_used - size < self.weight:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to free {size / MB_TO_BYTE:.2f}MB but only {(self.npu_used - self.weight) / MB_TO_BYTE:.2f}MB is used."
                )
            self.logger.info(
                "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used - size) / MB_TO_BYTE,
            )
            self.npu_used -= size
            self._check_kv_invariant("free")

        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.free(size)
            else:
                if self.cpu_used - size < 0:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to free {size / MB_TO_BYTE:.2f}MB "
                        f"but only {self.cpu_used / MB_TO_BYTE:.2f}MB is used."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used - size) / MB_TO_BYTE,
                )
                self.cpu_used -= size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.free(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to free KV cache in unsupported device {device}")
    
    def is_avail(self, size, device):
        if device == Device.NPU:
            if self.npu_mem - self.npu_used >= size:
                return True
            else:
                return False 
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.is_avail(size)
            else:
                if self.cpu_mem - self.cpu_used >= size:
                    return True
                else:
                    return False 
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.is_avail(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")
    
    def need_size(self, size, device):
        if device == Device.NPU:
            needed = (size - (self.npu_mem - self.npu_used))
            if needed > 0:
                return needed
            else:
                return 0
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.need_size(size)
            else:
                needed = (size - (self.cpu_mem - self.cpu_used))
                if needed > 0:
                    return needed
                else:
                    return 0
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.need_size(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")

    def avail_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            # D1 diagnostic: SIM_RESERVE_IGNORE=1 makes availability ignore the
            # in-step reservation, to test whether npu_reserved (~10,143 tokens
            # at preemption time, 12% of the B200 pool) is what drives the 9x
            # excess preemption vs vLLM. Default keeps current behaviour.
            _res = 0 if os.environ.get("SIM_RESERVE_IGNORE") else self.npu_reserved
            return max(0, self.npu_prefix_cache.avail_size() - _res)
        elif device == Device.CPU or device == Device.CXL:
            return max(0, self.second_tier_prefix_cache.avail_size() - self.cpu_swap_reserved)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get available size of prefix cache in unsupported device {device}")
    
    def enable_swap_tier(self):
        """Give this instance a CPU tier that holds only swapped contexts.

        InferCept moves a paused program's KV to host memory and restores it
        for the successor; the successor then prefix-hits on this tier and the
        scheduler charges the load at the measured host link rate. Unlike
        --prefix-storage, finished requests are not cached here by default.
        """
        if not self.enable_prefix_caching:
            raise RuntimeError('the swap tier requires prefix caching')
        if self.second_tier_prefix_cache is not None:
            return
        self.second_tier_prefix_cache = RadixCache(device="CPU",
                                                   node_id=self.node_id,
                                                   instance_id=self.instance_id,
                                                   page_size=self.block_size,
                                                   capacity=self.cpu_mem,
                                                   kv_size=(self.get_kv(1) * self.num_npus),
                                                   enable_kv_cache_events=True)
        self.swap_tier = True

    # -------------------- Prefix Cache Management --------------------

    def storage_cache_evicted_req(self, req):
        if self.enable_prefix_caching:
            new_last_node = self.second_tier_prefix_cache.cache_unfinished_req(req, update=False) # do not update hit counts
            # should lock evicted kv cache in cpu
            self.second_tier_prefix_cache.inc_lock_ref(new_last_node)
            req.cpu_last_node = new_last_node
            self.apply_kv_cache_events()

    # ---- D1 diagnostic (SIM_MEM_TRACE=<path>): occupancy + hit trace.
    # Zero cost when the env var is unset. Remove once D1 is closed.
    def _evict_trace_hook(self, owners, ntok):
        """D1 diagnostic: one line per node leaving the NPU radix tree,
        attributed to EVERY program whose prefix covers it, stamped with sim time,
        so evictions can be lined up against that program's tool gap."""
        f = getattr(self, "_evt_fh", None)
        if f is None:
            path = os.environ.get("SIM_EVICT_TRACE")
            if not path:
                self._evt_fh = False
                return
            try:
                # one handle for the run: this fires per evicted NODE, which is
                # hundreds of thousands of lines over a cell.
                self._evt_fh = f = open("%s.inst%s" % (path, self.instance_id), "a")
            except Exception:
                self._evt_fh = False
                return
        if f is False:
            return
        try:
            f.write('{"t":%.6f,"owners":%s,"tok":%d}\n'
                    % (self.sim_now / 1e9,
                       json.dumps(sorted(owners) if owners else []), ntok))
        except Exception:
            pass

    def _mem_trace(self, tag, **kw):
        path = os.environ.get("SIM_MEM_TRACE")
        if not path or not self.enable_prefix_caching:
            return
        if tag == "cache_unfinished":          # fires per step per request
            self._mt_n = getattr(self, "_mt_n", 0) + 1
            if self._mt_n % 100:
                return
        try:
            c = self.npu_prefix_cache
            rec = {"tag": tag,
                   "total_tok": c.total_size(),
                   "evictable_tok": c.evictable_size(),
                   "protected_tok": c.protected_size(),
                   "cap_tok": int(self.mem_for_kv // self._bytes_per_token)}
            rec.update(kw)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def evictable_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.evictable_size() * self._bytes_per_token
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get evictable size of prefix cache in unsupported device {device}")


    def lock_prefix(self, req, device): 
        # Increment lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.inc_lock_ref(req.npu_last_node)
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
        elif (device == Device.CPU or device == Device.CXL) and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.inc_lock_ref(req.cpu_last_node)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to lock prefix cache in unsupported device {device}")
    
    def unlock_prefix(self, req, device):
        # Decrement lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
            req.npu_last_node = None
            req._prefix_locked = False
        elif device == Device.CPU and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.dec_lock_ref(req.cpu_last_node)
            req.cpu_last_node = None
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to unlock prefix cache in unsupported device {device}")
    
    @contextmanager
    def _npu_publication(self, req, finished=False):
        """Make room before any caller publishes new radix entries.

        Retention reclamation is shared by prefill, decode, completion, and
        prefill/decode transfer. Keep the matched prefix alive while reclaiming
        other entries, and honour reservations belonging to other requests.
        A failed check leaves the request's reservation and ownership intact.
        """
        self.apply_kv_cache_events()
        cache = self.npu_prefix_cache
        ids = req.input_hash_ids + req.output_hash_ids
        ids = ids[:-1] if finished else ids[:req.num_computed_tokens]
        if cache.page_size != 1 and not cache.save_unfull_chunk:
            ids = ids[:len(ids) // cache.page_size * cache.page_size]
        match = cache.match_prefix(ids)
        cache.inc_lock_ref(match.last_device_node)
        try:
            new_bytes = self.get_kv(len(ids) - match.hit_length)
            other_reserved = self.npu_reserved - req.kv_reserved
            shortfall = self.npu_used + other_reserved + new_bytes - self.npu_mem
            if shortfall > 0:
                # Feasible first -- same rule as reserve_kv and the admission
                # guard: an impossible publication leaves the pins alone.
                if shortfall > (self.evictable_size(Device.NPU)
                                + self.reclaimable_parked_size(Device.NPU)):
                    raise KVCapacityError(
                        f"NPU publication rejected before insertion for request {req.id}: "
                        f"new={new_bytes} other_reserved={other_reserved} "
                        f"used={self.npu_used} capacity={self.npu_mem}; "
                        f"shortfall={shortfall} exceeds reclaimable "
                        f"{self.evictable_size(Device.NPU) + self.reclaimable_parked_size(Device.NPU)}")
                self.evict_prefix_cache(shortfall, Device.NPU)
            if self.npu_used + other_reserved + new_bytes > self.npu_mem:
                raise KVCapacityError(
                    f"NPU publication rejected before insertion for request {req.id}: "
                    f"new={new_bytes} other_reserved={other_reserved} "
                    f"used={self.npu_used} capacity={self.npu_mem}; "
                    f"locked_tokens={cache.protected_size()} "
                    "(running and retention references combined)"
                )
            yield
            self.apply_kv_cache_events()
        finally:
            cache.dec_lock_ref(match.last_device_node)

    def release_infercept_host(self, req):
        if req.infercept_cpu_node is not None:
            self.second_tier_prefix_cache.dec_lock_ref(req.infercept_cpu_node)
            req.infercept_cpu_node = None

    def cache_unfinished_req(self, req, device):
        # Get new_last_node via cache_unfinished_req (replaces last node)
        # Decrement old node's lock ref count, increment new node's lock ref count
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            with self._npu_publication(req, finished=False):
                self.release_kv_reservation(req)
                self.npu_prefix_cache._current_owner = (
                    req.session_id if req.session_id is not None else req.workflow_id)
                new_last_node = self.npu_prefix_cache.cache_unfinished_req(req)
            
                old_node = req.npu_last_node
                # print(f"[CACHE_UNFINISHED] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
                if old_node is not None and req._prefix_locked:
                    self.npu_prefix_cache.dec_lock_ref(old_node)
                self.npu_prefix_cache.inc_lock_ref(new_last_node)
                # print(f"[CACHE_UNFINISHED] req={req.id} AFTER: old_node_id={old_node.id}(lock_ref={old_node.lock_ref}) new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
                req.npu_last_node = new_last_node
                req._prefix_locked = True
                self._mem_trace("cache_unfinished", req=req.id)
                if self.logger.isEnabledFor(logging.DEBUG):
                    # print(f"cache_unfinished_req of req {req.id}")
                    # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
                    self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_unfinished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of unfinished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

        if device == Device.NPU and req.storage_restored:
            self.release_infercept_host(req)

    def cache_finished_req(self, req, device):
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            with self._npu_publication(req, finished=True):
                self.release_kv_reservation(req)
                self.npu_prefix_cache._current_owner = (
                    req.session_id if req.session_id is not None else req.workflow_id)
                self.npu_prefix_cache.cache_finished_req(req)
                # Only dec_lock_ref if the request was locked
                node = req.npu_last_node
                if not req._prefix_locked:
                    # Never locked → skip dec
                    pass
                    # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref={node.lock_ref if node else 'N/A'} (SKIPPED dec - not locked)")
                else:
                    # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
                    if node is not None:
                        self.npu_prefix_cache.dec_lock_ref(node)
                        req.npu_last_node = None
                    req._prefix_locked = False
                # node = req.npu_last_node
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
                # self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
                    # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_AFTER={node.lock_ref if node else 'N/A'}")
                # print(f"[CACHE_FINISHED] req={req.id} evictable_size={self.npu_prefix_cache.evictable_size()} protected_size={self.npu_prefix_cache.protected_size()} total_size={self.npu_prefix_cache.total_size()}")
                if self.logger.isEnabledFor(logging.DEBUG):
                    print(f"cache_finished_req of req {req.id}")
                    print(f"===============NPU PREFIX CACHE of Instance[{self.instance_id}]=================")
                    self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_finished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_finished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of finished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

        if device == Device.NPU:
            self.release_infercept_host(req)

    def _reclaim_key(self, device):
        """The eviction order a policy installed for the NPU tier, if any.

        Asked per reclamation rather than cached: SAGA's ranking is a function
        of the current residency and clock, so a key computed once at
        attachment would rank the pool as it looked then.
        """
        if device is not Device.NPU or self.eviction_order is None:
            return None
        return self.eviction_order.node_key(self, getattr(self, "sim_now", 0) or 0)

    def evict_prefix_cache(self, bytes, device):
        if not self.enable_prefix_caching or bytes <= 0:
            return

        if device == Device.NPU:
            cache = self.npu_prefix_cache
        elif device == Device.CPU:
            cache = self.second_tier_prefix_cache
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to evict prefix cache to unsupported device {device}")

        # Each cache instance carries its own bytes-per-token in kv_size:
        # per-rank for NPU, full-cluster for the second-tier pool.
        space_needed = int((bytes + cache.kv_size - 1) // cache.kv_size)
        if device == Device.NPU:
            self._mem_trace("evict_pre", want_tok=space_needed)
        if self.kv_protection is not None and device == Device.NPU:
            # Retention safety valve: break parked protections when the
            # LRU alone cannot supply the eviction target.
            self.kv_protection.ensure_evictable_tokens(self, space_needed)
        cache.evict(space_needed, key=self._reclaim_key(device))
        if device == Device.NPU:
            self._mem_trace("evict_post", want_tok=space_needed)

        self.apply_kv_cache_events()

    # -------------------- Prefix Cache Helpers --------------------

    def prefix_match(self, req): # req.prefix_cache_hit initialization 
        if not self.enable_prefix_caching:
            return
        
        tokens = self._match_key(req)
        if tokens is None:
            return
        old_node = req.npu_last_node
        # Cap the match at input-1 tokens, mirroring vLLM v1
        # (v1/core/kv_cache_manager.py: "When all tokens hit the cache, we
        # must recompute the last token to obtain logits. Thus, set
        # max_cache_hit_length to prompt_length - 1"). Without the cap a
        # prompt that is an exact block multiple and fully cached gets
        # num_computed_tokens == input, is_prefill() turns False, and the
        # waiting-queue loop never schedules it: it sits until some other
        # arrival forms a batch it can ride in, so its TTFT becomes the gap
        # to the next arrival. With the cap the last block is recomputed and
        # the request takes the ordinary prefill path, as it does in vLLM.
        # original_input, not input: after a RECOMPUTE preemption the prompt
        # IS the old prompt plus what the request had generated, and vLLM
        # matches against the request's whole current token sequence.
        max_hit = max(0, req.original_input - 1)
        res = self.npu_prefix_cache.match_prefix(tokens[:max_hit])
        req.npu_cache_hit = res.hit_length
        req.npu_last_node = res.last_device_node
        # print(f"[PREFIX_MATCH] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={res.last_device_node.id}(lock_ref={res.last_device_node.lock_ref}) hit={res.hit_length} num_computed={req.num_computed_tokens}")

        if self.second_tier_prefix_cache is not None:
            res_storage = self.second_tier_prefix_cache.match_prefix(tokens[:max_hit])
            req.storage_cache_hit = res_storage.hit_length
            req.storage_last_node = res_storage.last_device_node
        else:
            req.storage_cache_hit = 0
            req.storage_last_node = None
        
        req.prefix_cache_hit = max(req.npu_cache_hit, req.storage_cache_hit)
        self._mem_trace("match", req=req.id, inp=req.input, hit=req.npu_cache_hit)
        # if req.num_computed_tokens < req.prefix_cache_hit:
        #     req.num_computed_tokens = req.prefix_cache_hit
        if req.num_computed_tokens == 0:
            req.num_computed_tokens = req.prefix_cache_hit
            # print(f"Request[{req.id}] prefix cache hit: {req.prefix_cache_hit} tokens (NPU: {req.npu_cache_hit}, {self.prefix_storage}: {req.storage_cache_hit})")
        # for debugging
        
        # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
        # self.npu_prefix_cache.pretty_print()
        # print("===============CPU PREFIX CAHCE=================")
        # self.second_tier_prefix_cache.pretty_print()
    
    @staticmethod
    def _match_key(req):
        """The token ids a request presents to the prefix cache.

        vLLM matches on the request's WHOLE current token sequence
        (kv_cache_manager.get_computed_blocks hashes request.all_token_ids).
        After a RECOMPUTE preemption that sequence is the original prompt plus
        the tokens the request had generated -- blocks vLLM freed tail-first
        but left hashed and re-hittable, so it rematches nearly all of them.

        Keying on input_hash_ids alone capped the rematch at the ORIGINAL
        prompt while _preempt_recompute had already grown original_input by
        the generated count, so every preemption re-prefilled the generated
        context from scratch. That inflated recomputation and, with it,
        congestion and further preemptions.
        """
        ids = req.input_hash_ids
        if ids is None:
            return None
        extra = req.original_input - len(ids)
        if extra > 0 and req.output_hash_ids:
            return list(ids) + list(req.output_hash_ids[:extra])
        return ids

    def peek_prefix_hit(self, req):
        """Read-only NPU prefix hit for a request that is not scheduled
        yet (admission-gate probe). Same key and cap as prefix_match."""
        if not self.enable_prefix_caching or req.input_hash_ids is None:
            return 0
        tokens = self._match_key(req)
        return self.npu_prefix_cache.peek_prefix_length(
            tokens[:max(0, req.original_input - 1)])

    def peek_storage_hit(self, req):
        """Read-only second-tier prefix hit for a request that is not
        scheduled yet.

        The admission gate runs before STEP 1's prefix_match, so a waiting
        request's storage_cache_hit is still its initial zero. Reading that
        field in the gate made InferCept's FCFS restore a no-op that looked
        like it was admitting everything (job 42596388: 1.3M blocks swapped,
        zero restores gated). Same key and cap as prefix_match, and it
        mutates nothing.
        """
        if (not self.enable_prefix_caching or req.input_hash_ids is None
                or self.second_tier_prefix_cache is None):
            return 0
        tokens = self._match_key(req)
        return self.second_tier_prefix_cache.peek_prefix_length(
            tokens[:max(0, req.original_input - 1)])

    def erase_prefix_info(self, req):
        # A request that has not computed beyond its prefix hit must also
        # forget the hit-derived num_computed_tokens; otherwise it keeps
        # believing those tokens are computed while the (now unlocked)
        # prefix can be evicted, and its later insert re-creates them
        # with no reservation (pool over-subscription under pressure).
        if req.num_computed_tokens <= req.prefix_cache_hit:
            req.num_computed_tokens = 0
        if not self.enable_prefix_caching:
            return
        
        req.prefix_cache_hit = 0
        req.npu_cache_hit = 0
        req.storage_cache_hit = 0
        req.storage_restored = False
        req.npu_last_node = None
        req.storage_last_node = None

    def free_prefix_cache(self):
        if not self.enable_prefix_caching:
            return
        if self.host_swap is not None:
            self.host_swap.close()
        # End of run: the pool is given back while the tree still holds it, so
        # the two accounts are MEANT to diverge here. Without this the
        # invariant check reports teardown as the drift and hides the real one.
        self._tearing_down = True
        # free evictable prefix cache, if evictable_size != total_size there is locked prefix cache
        self.free(self.npu_prefix_cache.evictable_size() * self._bytes_per_token, Device.NPU)
        if not self.enable_prefix_sharing and self.prefix_storage is not None:
            self.free(self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token * self.num_npus, self.prefix_storage)
        elif self.swap_tier:
            self.free(self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token * self.num_npus, Device.CPU)
    
    # Count load/unload events from prefix cache and update memory usage
    def apply_kv_cache_events(self):
        """Reconcile the tree's store/remove events against npu_used.

        Capacity failures leave their events pending for recovery. The depth
        guard suppresses invariant checks while an update is in progress and
        must unwind even on failure; otherwise one failed publication silently
        disables diagnostics for the rest of the run. Reclamation is staged
        inside the same transaction rather than recursively charging events.
        """
        self._recon_depth = getattr(self, "_recon_depth", 0) + 1
        try:
            out = self._apply_kv_cache_events()
        except BaseException:
            # Unwind, but do NOT check: the accounts are mid-flight, and a
            # second error raised from here would mask the real one (the
            # insert-does-not-fit report is what the caller needs to see).
            self._recon_depth = max(0, self._recon_depth - 1)
            raise
        self._recon_depth = max(0, self._recon_depth - 1)
        if self._recon_depth == 0:
            self._check_kv_invariant("apply_kv_cache_events")
        return out

    def _apply_kv_cache_events(self):
        # Stage the hash-map edits and the memory delta together. Draining
        # events is not a commit: capacity recovery may fail after the tree
        # has already changed. Keep every event for the next attempt until
        # both accounts can be updated, including removals made by reclamation.
        cache = self.npu_prefix_cache
        events = cache.take_events()
        changed = {}
        delta = 0

        def stage(pending):
            nonlocal delta
            for ev in pending:
                if isinstance(ev, BlockStored):
                    tlen = len(ev.token_ids)
                    for h in ev.block_hashes:
                        old = changed.get(h, self._npu_cache_hashtolen.get(h))
                        changed[h] = [old[0], old[1] + 1] if old else [tlen, 1]
                    delta += self.get_kv(tlen)
                elif isinstance(ev, BlockRemoved):
                    for h in ev.block_hashes:
                        old = changed.get(h, self._npu_cache_hashtolen.get(h))
                        if old is None:
                            raise RuntimeError(f"NPU prefix cache remove unknown block hash {h}")
                        delta -= self.get_kv(old[0])
                        changed[h] = [old[0], old[1] - 1] if old[1] > 1 else None

        try:
            stage(events)
            shortfall = self.npu_used + delta - self.npu_mem
            if shortfall > 0:
                tokens = int((shortfall + cache.kv_size - 1) // cache.kv_size)
                if self.kv_protection is not None:
                    self.kv_protection.ensure_evictable_tokens(self, tokens)
                # Do not call evict_prefix_cache: it recursively reconciles
                # before this transaction's insertions have been charged.
                cache.evict(tokens, key=self._reclaim_key(Device.NPU))
                reclaimed = cache.take_events()
                events.extend(reclaimed)
                stage(reclaimed)
            if self.npu_used + delta > self.npu_mem:
                raise KVCapacityError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
                    f"NPU prefix-cache publication does not fit: net change "
                    f"{delta / MB_TO_BYTE:.2f}MB, used {self.npu_used / MB_TO_BYTE:.2f}MB "
                    f"of {self.npu_mem / MB_TO_BYTE:.2f}MB; "
                    f"cache total {cache.total_size() * self._bytes_per_token / MB_TO_BYTE:.2f}MB, "
                    f"protected {cache.protected_size() * self._bytes_per_token / MB_TO_BYTE:.2f}MB"
                )
            if self.npu_used + delta < self.weight:
                raise RuntimeError("NPU cache events would free model weights")
        except BaseException:
            # Preserve chronological order: a later removal must follow the
            # insertion it cancels. Neither the hash map nor npu_used changed.
            cache.kv_event_queue = events + cache.kv_event_queue
            raise

        for h, value in changed.items():
            if value is None:
                self._npu_cache_hashtolen.pop(h, None)
            else:
                self._npu_cache_hashtolen[h] = value
        if delta > 0:
            self.allocate(delta, Device.NPU)
        elif delta < 0:
            self.free(-delta, Device.NPU)

        cpu_byte_alloc = 0
        cpu_byte_free = 0
        # Second-tier (CPU/CXL) prefix cache events.
        if self.second_tier_prefix_cache is None:
            return

        if (self.prefix_storage is Device.CPU or self.swap_tier) and not self.enable_prefix_sharing:
            # Non-shared CPU second_tier: bridge events into the instance's
            # cpu_used counter so allocations are bounded by cpu_mem.
            for ev in self.second_tier_prefix_cache.take_events():
                if isinstance(ev, BlockStored):
                    tlen = len(ev.token_ids)
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            self._cpu_cache_hashtolen[h][1] += 1
                        else:
                            self._cpu_cache_hashtolen[h] = [tlen, 1]
                    cpu_byte_alloc += self.get_kv(tlen) * self.num_npus
                elif isinstance(ev, BlockRemoved):
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            tlen = self._cpu_cache_hashtolen[h][0]
                            self._cpu_cache_hashtolen[h][1] -= 1
                            if self._cpu_cache_hashtolen[h][1] <= 0:
                                del self._cpu_cache_hashtolen[h]
                            cpu_byte_free += self.get_kv(tlen) * self.num_npus
                        else:
                            self.logger.warning(f"CPU prefix cache remove unknown block hash {h}")

            if cpu_byte_free > 0:
                self.free(cpu_byte_free, Device.CPU)
            if cpu_byte_alloc > 0:
                self.allocate(cpu_byte_alloc, Device.CPU)
        else:
            # Shared pool or CXL: the cache itself accounts via
            # total_memory_usage (= kv_stored + total_size * kv_size),
            # so no instance-side counter update is needed. Drain the
            # queue to prevent it from growing unboundedly.
            self.second_tier_prefix_cache.take_events()

    def return_prefix_info(self):
        if not self.enable_prefix_caching:
            return (0, 0, 0, 0)
        if self.second_tier_prefix_cache is None:
            return (self.npu_prefix_cache.return_prefix_info(), (0, 0))
        return (self.npu_prefix_cache.return_prefix_info(), self.second_tier_prefix_cache.return_prefix_info())

        
def full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype='auto'):
    """Bytes of KV cache per token aggregated over the full TP cluster.

    Mirrors MemoryModel.get_kv(1) * num_npus but computes directly, avoiding
    the per-rank floor-division roundoff. ``fp`` is the model weight dtype
    in bits (16, 32, ...). ``kv_cache_dtype='fp8'`` forces 1 byte per element
    for the KV cache regardless of weight dtype.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    kv_head = config.get('num_key_value_heads', n_head)
    kv_dim = kv_head * head_dim
    n_layer = config['num_hidden_layers']
    kv_fp = 1 if kv_cache_dtype == 'fp8' else fp // 8
    # 2 (K + V) * kv_dim * n_layer * bytes_per_elem
    return 2 * kv_dim * n_layer * kv_fp


# calculate the per-rank input, weight, output size of each layer
def calculate_sizes(model, layer_name, length, kv_len=None, pim=False, parallel=1, fp=2):
    """Calculate input, weight, and output tensor sizes for a given layer.

    Args:
        parallel: parallelism degree for weight/activation sharding.
            For dense layers this is TP; for MoE experts this is EP.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    vocab_size = config['vocab_size']
    kv_head = config.get("num_key_value_heads", n_head)  # fallback to n_head if not defined
    q_dim = n_head * head_dim       # total Q projection output dim
    kv_dim = kv_head * head_dim     # total KV projection output dim
    ffn_dim = config.get("intermediate_size", config.get("ffn_dim"))  # dense FFN dim
    moe_ffn_dim = config.get("moe_intermediate_size", ffn_dim)  # per-expert FFN dim (may differ from dense)
    # Same both-name fallback as MemoryModel.__init__ — HF / Qwen use
    # ``num_experts`` while Mistral uses ``num_local_experts``.
    num_local_experts = config.get(
        "num_local_experts", config.get("num_experts", 1)
    )

    p = max(int(parallel), 1)

    # NOTE (vLLM-style assumptions):
    # NOTE (vLLM-style assumptions):
    # - Embedding / LM head: vocab-parallel → split vocab_size across ranks.
    # - Q/K/V: ColumnParallelLinear         → split output dim across ranks.
    # - o_proj: RowParallelLinear           → split input dim across ranks.
    # - LayerNorm weights: replicated (NOT sharded).
    # - MoE experts: parallel = EP degree, each rank holds num_local_experts // p experts.

    # ----------------- Embedding & Norms -----------------
    if layer_name == "embedding":
        input_size = length * fp * 2  # token_ids are int32 or int64
        weight_size = (vocab_size // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name in ["input_layernorm", "post_layernorm", "final_layernorm", "layernorm"]:
        input_size = length * n_embd * fp
        weight_size = 1 * n_embd * fp  # scale only
        output_size = length * n_embd * fp

    elif layer_name == "qk_norm":
        input_size = length * (q_dim + kv_dim) // p * fp
        weight_size = 2 * head_dim * fp
        output_size = length * (q_dim + kv_dim) // p * fp

    # ----------------- RoPE & Attention Core -----------------
    elif layer_name == "rotary_emb":
        input_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp
        weight_size = 0
        output_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp

    elif layer_name == "attention":
        if not pim:
            input_size = (
                (n_head // p) * length * head_dim * fp +
                (kv_head // p) * kv_len * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * length * head_dim * fp
        else:
            input_size = (
                (n_head // p) * 1 * head_dim * fp +
                (kv_head // p) * 1 * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * 1 * head_dim * fp

    # ----------------- QKV Projection (fused) -----------------
    elif layer_name == "qkv_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * ((q_dim + 2 * kv_dim) // p) * fp
        output_size = length * ((q_dim + 2 * kv_dim) // p) * fp

    elif layer_name == "o_proj":
        input_size = length * (q_dim // p) * fp
        weight_size = (q_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "gate_up_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * 2 * (ffn_dim // p) * fp
        output_size = length * 2 * (ffn_dim // p) * fp

    elif layer_name == "act_fn":
        input_size = length * 2 * (ffn_dim // p) * fp
        weight_size = 0
        output_size = length * (ffn_dim // p) * fp

    elif layer_name == "down_proj":
        input_size = length * (ffn_dim // p) * fp
        weight_size = (ffn_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "sampler":
        input_size = length * (vocab_size // p) * fp
        weight_size = 0
        output_size = length * 4  # int32 token IDs

    elif layer_name == "moe":
        experts_per_rank = num_local_experts // p
        input_size = length * n_embd * fp
        weight_size = (n_embd * num_local_experts * fp  # gate (replicated)
                     + experts_per_rank * 3 * n_embd * moe_ffn_dim * fp)  # local experts
        output_size = length * n_embd * fp

    # ----------------- LM Head -----------------
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        weight_size = n_embd * (vocab_size // p) * fp
        output_size = length * (vocab_size // p) * fp

    else:
        raise ValueError(f"No matching layer name {layer_name} found for model {model}.")

    return input_size, weight_size, output_size
