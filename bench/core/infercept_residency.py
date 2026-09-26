"""Chunked InferCept residency using vLLM's authoritative physical allocator.

The scheduler keeps transferring requests out of runnable queues. Source GPU
references survive until all-rank acknowledgement; incoming blocks are not
published in the prefix index until then. CPU staging belongs to the connector
and is excluded from this class's allocatable CPU capacity.
"""
from collections import Counter, deque
from dataclasses import dataclass, field

from .infercept_transfer import LayerSwapPlan


@dataclass
class PausedKV:
    request: object
    computed_tokens: int
    cpu: dict = field(default_factory=dict)
    discarded: bool = False
    cancelled: bool = False


@dataclass
class TransferChunk:
    state: PausedKV
    storing: bool
    indices: tuple
    gpu: tuple
    cpu: tuple


@dataclass
class PendingSwap:
    ticket: int
    chunks: tuple
    handed_off: bool = False

    @property
    def gpu(self):
        return tuple(block for chunk in self.chunks for block in chunk.gpu)

    def contains(self, state):
        return any(chunk.state is state for chunk in self.chunks)


class InferceptResidency:
    def __init__(self, manager, connector, cpu_blocks):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        managers = manager.coordinator.single_type_managers
        if len(managers) != 1 or not isinstance(managers[0].kv_cache_spec, FullAttentionSpec):
            raise ValueError('chunked residency requires one full-attention KV group')
        if not isinstance(cpu_blocks, int) or cpu_blocks < 0:
            raise ValueError('allocatable CPU capacity must be nonnegative')
        self.manager, self.single = manager, managers[0]
        self.pool, self.connector = manager.block_pool, connector
        self.block_size = self.single.block_size
        self.cpu_blocks = cpu_blocks
        self.free_cpu = deque(range(cpu_blocks))
        self.states = {}
        self.pending = None
        self.stats = {'stored_blocks': 0, 'loaded_blocks': 0, 'discarded_blocks': 0}

    def pause(self, request):
        if request.request_id in self.states:
            raise ValueError('request already has an interception residency record')
        state = PausedKV(request, request.num_computed_tokens)
        self.states[request.request_id] = state
        return state

    def _state(self, request_id, count):
        if self.pending is not None:
            raise RuntimeError('finish the pending transfer before reserving another')
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError('chunk must contain a positive number of blocks')
        state = self.states[request_id]
        if state.cancelled:
            raise RuntimeError('request was cancelled')
        return state

    def store_tail(self, request_id, count):
        return self.transfer(stores=((request_id, count),))

    def load_prefix(self, request_id, count):
        return self.transfer(loads=((request_id, count),))

    def transfer_for_iteration(self, *, stores=(), loads=(), discard_remainders=()):
        """Transfer while making outgoing pages allocatable this iteration.

        The connector stages each outgoing layer before the model may write
        that layer. This permits the scheduler to allocate pages handed off by
        paused requests to incoming or newly scheduled work in the same
        forward pass. Requests participating in the transfer must remain out
        of the runnable batch until the all-rank acknowledgement arrives.
        """
        return self.transfer(stores=stores, loads=loads, handoff=True,
                             discard_remainders=discard_remainders)

    def transfer(self, *, stores=(), loads=(), handoff=False,
                 discard_remainders=()):
        """Reserve a joint transaction across paused and returning requests.

        Host destinations may reuse slots being read by this transaction: the
        layer worker stages outgoing bytes until all incoming reads finish.
        With ``handoff=False``, source ownership remains until ACK. With
        ``handoff=True``, pages owned only by outgoing requests become
        allocatable for the same forward; layer events protect their contents.
        """
        stores, loads = tuple(stores), tuple(loads)
        discard_remainders = frozenset(discard_remainders)
        ids = [rid for rid, _ in (*loads, *stores)]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('transfer requires distinct nonempty request actions')
        store_ids = {rid for rid, _ in stores}
        if discard_remainders - store_ids or (discard_remainders and not handoff):
            raise ValueError('only handed-off swap-out requests can discard remainders')
        incoming, outgoing = [], []
        for rid, count in loads:
            state = self._state(rid, count)
            start = len(self.single.req_to_blocks.get(rid, ()))
            indices = tuple(range(start, start + count))
            if any(index not in state.cpu for index in indices):
                raise ValueError('restore must extend the resident prefix contiguously')
            incoming.append(TransferChunk(state, False, indices, (),
                                          tuple(state.cpu[index] for index in indices)))
        recycled = tuple(slot for chunk in incoming for slot in chunk.cpu)
        available = tuple(self.free_cpu) + recycled
        total_load = sum(len(chunk.indices) for chunk in incoming)
        total_store = sum(count for _, count in stores)
        if total_store > len(available):
            raise MemoryError('allocatable host capacity is insufficient')
        if total_store > self.connector.scratch_blocks:
            raise ValueError('swap-out plan exceeds reserved host staging')
        offset = 0
        for rid, count in stores:
            state = self._state(rid, count)
            blocks = self.single.req_to_blocks.get(rid, ())
            if count > len(blocks):
                raise MemoryError('resident tail is insufficient')
            indices = tuple(range(len(blocks) - count, len(blocks)))
            if any(index in state.cpu for index in indices):
                raise RuntimeError('tail is already stored on CPU')
            outgoing.append(TransferChunk(state, True, indices, tuple(blocks[-count:]),
                                          available[offset:offset + count]))
            offset += count
        source = tuple(block for chunk in outgoing for block in chunk.gpu)
        source_refs = Counter(block.block_id for block in source)
        candidates = {block.block_id: block for block in source
                      if not block.is_null and block.ref_cnt == source_refs[block.block_id]}
        free_count = min(total_load, self.pool.get_num_free_blocks())
        shortfall = total_load - free_count
        if shortfall > len(candidates):
            raise MemoryError('GPU capacity cannot hold the incoming chunks')
        recycled_gpu = tuple(candidates.values())[:shortfall]
        # Validate capacity before allocation, which may break retention pins.
        fresh_gpu = tuple(self.pool.get_new_blocks(free_count)) if free_count else ()
        self.pool.touch(recycled_gpu)  # incoming reservations, distinct from DMA holds
        gpu = fresh_gpu + recycled_gpu
        offset = 0
        for chunk in incoming:
            count = len(chunk.indices)
            chunk.gpu = gpu[offset:offset + count]
            offset += count
        self.pool.touch(source)  # DMA source holds, one per request reference.
        try:
            ticket = self.connector.queue_plan(LayerSwapPlan(
                store_gpu=tuple(block.block_id for block in source),
                store_cpu=tuple(slot for chunk in outgoing for slot in chunk.cpu),
                load_cpu=recycled, load_gpu=tuple(block.block_id for block in gpu)))
        except Exception:
            self.pool.free_blocks(source)
            self.pool.free_blocks(gpu)
            raise
        if handoff:
            # Every eligible source can be assigned to new work, not only the
            # subset selected as an incoming load destination. Remove its old
            # hash before another prefix lookup can adopt bytes being replaced.
            self.pool.evict_blocks(set(candidates))
            for chunk in outgoing:
                rid = chunk.state.request.request_id
                blocks = self.single.req_to_blocks[rid]
                if tuple(blocks[-len(chunk.gpu):]) != chunk.gpu:
                    raise RuntimeError('source ownership changed during transfer handoff')
                del blocks[-len(chunk.gpu):]
                self.single.num_cached_block[rid] = min(
                    self.single.num_cached_block.get(rid, 0), len(blocks))
                if rid in discard_remainders:
                    remainder = tuple(blocks)
                    self.manager.free(chunk.state.request)
                    self.pool.evict_blocks({block.block_id for block in remainder
                                            if block.ref_cnt == 0})
                    chunk.state.request.num_computed_tokens = 0
                    chunk.state.discarded = True
                    boundary = min((*chunk.state.cpu, *chunk.indices))
                    chunk.state.request.infercept_compute_limit = (
                        boundary * self.block_size)
                    self.stats['discarded_blocks'] += len(remainder)
                else:
                    chunk.state.request.num_computed_tokens = min(
                        chunk.state.computed_tokens, len(blocks) * self.block_size)
                self.pool.free_blocks(chunk.gpu)  # request ownership
            for chunk in outgoing:
                self.pool.free_blocks(chunk.gpu)  # layer fences replace DMA holds
        else:
            # A conservative transaction may still exchange a source and
            # destination ID. Prevent prefix adoption before that overwrite.
            self.pool.evict_blocks({block.block_id for block in recycled_gpu})
        reserved = {slot for chunk in outgoing for slot in chunk.cpu}
        self.free_cpu = deque(slot for slot in self.free_cpu if slot not in reserved)
        self.pending = PendingSwap(ticket, tuple(incoming + outgoing), handoff)
        return ticket

    def discard_prefix(self, request_id):
        """Discard resident KV, preserving CPU chunks and known token history.

        Recompute may run only up to the first CPU chunk. The caller must then
        restore that chunk and refresh the worker block table before admission.
        Shared blocks remain owned by their other users; they are not evicted.
        """
        if self.pending is not None:
            raise RuntimeError('cannot discard while a transfer is pending')
        state = self.states[request_id]
        blocks = tuple(self.single.req_to_blocks.get(request_id, ()))
        self.manager.free(state.request)
        self.pool.evict_blocks({b.block_id for b in blocks if b.ref_cnt == 0})
        state.request.num_computed_tokens = 0
        state.discarded = True
        state.request.infercept_compute_limit = (
            min(state.cpu) * self.block_size if state.cpu else None)
        self.stats['discarded_blocks'] += len(blocks)

    def finish(self):
        """Publish an acknowledged transaction; keep all buffers until ACK."""
        pending = self.pending
        if pending is None:
            return True
        # Validate every owner before consuming the ACK or mutating any request.
        for chunk in pending.chunks:
            if chunk.state.cancelled:
                continue
            blocks = self.single.req_to_blocks.get(chunk.state.request.request_id, ())
            if chunk.storing and not pending.handed_off:
                if tuple(blocks[-len(chunk.gpu):]) != chunk.gpu:
                    raise RuntimeError('source ownership changed during paused transfer')
            elif chunk.storing:
                # Handed-off pages left the request at transfer time; a discarded
                # remainder leaves nothing resident, otherwise the prefix stays.
                expected = 0 if chunk.state.discarded else chunk.indices[0]
                if len(blocks) != expected:
                    raise RuntimeError('resident remainder changed during handed-off store')
            elif len(blocks) != chunk.indices[0]:
                raise RuntimeError('resident prefix changed during paused restore')
        if not self.connector.take_acknowledgement(pending.ticket):
            return False
        # Reads precede writes: recycled host slots lose their old owner first.
        for chunk in pending.chunks:
            state = chunk.state
            request, rid = state.request, state.request.request_id
            if state.cancelled:
                if not chunk.storing:
                    self.pool.free_blocks(chunk.gpu)
                elif not pending.handed_off:
                    # finish_requests() already released request ownership;
                    # only the transfer's source hold survives until ACK.
                    self.pool.free_blocks(chunk.gpu)
                del self.states[rid]
                continue
            blocks = self.single.req_to_blocks.get(rid, [])
            if chunk.storing:
                if not pending.handed_off:
                    del blocks[-len(chunk.gpu):]
                    self.single.num_cached_block[rid] = min(
                        self.single.num_cached_block.get(rid, 0), len(blocks))
                    self.pool.free_blocks(chunk.gpu)  # request ownership
                    self.pool.free_blocks(chunk.gpu)  # DMA hold
                state.cpu.update(zip(chunk.indices, chunk.cpu))
                self.stats['stored_blocks'] += len(chunk.gpu)
            else:
                blocks.extend(chunk.gpu)
                for index in chunk.indices:
                    del state.cpu[index]
                self.stats['loaded_blocks'] += len(chunk.gpu)
            # A restore extends recomputed context across the CPU boundary;
            # a handed-off store must not resurrect its discarded prefix.
            if not chunk.storing or not state.discarded:
                request.num_computed_tokens = min(
                    state.computed_tokens, len(blocks) * self.block_size)
            if not chunk.storing:
                self.manager.cache_blocks(request, request.num_computed_tokens)
            request.infercept_compute_limit = (
                min(state.cpu) * self.block_size if state.cpu else None)
        occupied = [slot for state in self.states.values() for slot in state.cpu.values()]
        if len(occupied) != len(set(occupied)):
            raise RuntimeError('CPU slots acquired multiple owners')
        occupied = set(occupied)
        self.free_cpu = deque(slot for slot in range(self.cpu_blocks) if slot not in occupied)
        self.pending = None
        return True

    def resume(self, request_id):
        state = self.states[request_id]
        if state.cpu or (self.pending is not None and self.pending.contains(state)):
            raise RuntimeError('cannot resume while context remains on CPU or in flight')
        state.request.infercept_compute_limit = None
        del self.states[request_id]

    def reconcile_full_prompt(self, request, prompt):
        """Keep only full KV blocks whose tokens match the recorded next prompt.

        A replay's recorded assistant tokens need not match actual generation.
        Release divergent GPU/CPU suffixes before replacing token history.
        An in-flight copy must finish before any ownership is changed.
        """
        if self.pending is not None:
            raise RuntimeError('full-prompt reconciliation requires completed DMA')
        common = 0
        for old, new in zip(request.all_token_ids, prompt):
            if old != new:
                break
            common += 1
        state = self.states.get(request.request_id)
        computed = state.computed_tokens if state is not None else request.num_computed_tokens
        tokens = min(common, computed, max(0, len(prompt) - 1))
        keep = tokens // self.block_size
        blocks = self.single.req_to_blocks.get(request.request_id, [])
        dropped = tuple(blocks[keep:])
        del blocks[keep:]
        self.pool.free_blocks(dropped)
        if blocks:
            self.single.num_cached_block[request.request_id] = min(
                self.single.num_cached_block.get(request.request_id, 0), len(blocks))
        else:
            # Presence, even with value zero, selects vLLM's already-admitted
            # fast path, which forbids new prefix hits. An empty owner must
            # rematch normally on its next full-prompt admission.
            self.single.num_cached_block.pop(request.request_id, None)
        request.num_computed_tokens = min(request.num_computed_tokens, keep * self.block_size)
        if state is not None:
            for index in tuple(state.cpu):
                if index >= keep:
                    self.free_cpu.append(state.cpu.pop(index))
            state.computed_tokens = keep * self.block_size
        request.infercept_compute_limit = (
            min(state.cpu) * self.block_size if state is not None and state.cpu else None)

    def cancel(self, request_id):
        """Called alongside native request free; never release DMA buffers early."""
        state = self.states.get(request_id)
        if state is None:
            return
        state.cancelled = True
        if self.pending is None or not self.pending.contains(state):
            self.free_cpu.extend(state.cpu.values())
            del self.states[request_id]
