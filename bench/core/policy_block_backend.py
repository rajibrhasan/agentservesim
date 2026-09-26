"""Physical block ownership for policy-directed vLLM CPU transfers.

Pools are vLLM BlockPool objects. The submit callback queues worker DMA metadata,
and completion must be acknowledged by every worker rank before ready() succeeds.
Each handle owns its own references, including when two requests share a block.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

from policies.kv_transfer_runtime import KVLocation


@dataclass
class Copy:
    source: KVLocation
    destination: KVLocation
    remaining_ranks: set


class BlockTransferBackend:
    def __init__(self, engine, gpu_pool, cpu_pool, bytes_per_block,
                 world_size, submit):
        if bytes_per_block <= 0 or world_size <= 0:
            raise ValueError('positive block size and worker count required')
        self.engine = engine
        self.pools = {'gpu': gpu_pool, 'cpu': cpu_pool}
        self.bytes_per_block = bytes_per_block
        self.world_size = world_size
        self.submit = submit
        self._next_handle = 0
        self._next_ticket = 0
        self._allocations: Dict[int, Tuple[KVLocation, tuple]] = {}
        self._copies: Dict[int, Copy] = {}

    def _own(self, tier, blocks):
        handle = self._next_handle
        self._next_handle += 1
        location = KVLocation(self.engine, tier, handle,
                              len(blocks) * self.bytes_per_block)
        self._allocations[handle] = (location, tuple(blocks))
        return location

    def capture_gpu(self, block_ids):
        """Retain real source refs at a boundary where the caller stops writes.

        Includes unfinished physical blocks. Caller still owns its original
        references and must release those through its request allocator after
        capture; this handle alone cannot free somebody else's references.
        """
        ids = tuple(block_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('source must contain distinct physical blocks')
        pool = self.pools['gpu']
        if any(not isinstance(i, int) or not 0 <= i < len(pool.blocks) for i in ids):
            raise ValueError('invalid source block ID')
        blocks = tuple(pool.blocks[i] for i in ids)
        if any(b.is_null or b.ref_cnt <= 0 for b in blocks):
            raise ValueError('source must be resident and owned')
        pool.touch(blocks)
        return self._own('gpu', blocks)

    def reserve(self, request_id, engine, tier, size_bytes):
        if engine != self.engine:
            raise ValueError('cross-engine transfers require a migration transport')
        if tier not in self.pools or size_bytes <= 0 or size_bytes % self.bytes_per_block:
            raise ValueError('reservation must contain whole physical blocks')
        count = size_bytes // self.bytes_per_block
        pool = self.pools[tier]
        # vLLM counts reclaimable protected blocks here. get_new_blocks owns
        # pin breaking; a failed feasibility check must not break any pins.
        if count > pool.get_num_free_blocks():
            raise MemoryError(f'{tier} pool cannot reserve {count} blocks')
        blocks = pool.get_new_blocks(count)
        return self._own(tier, blocks)

    def block_ids(self, location):
        record = self._allocations.get(location.handle)
        if record is None or record[0] != location:
            raise ValueError('unknown or mismatched allocation')
        return tuple(b.block_id for b in record[1])

    def start(self, source, destination):
        if source.tier == destination.tier or source.size_bytes != destination.size_bytes:
            raise ValueError('copy requires equally sized CPU and GPU allocations')
        src = self.block_ids(source)
        dst = self.block_ids(destination)
        ticket = self._next_ticket
        self._next_ticket += 1
        self._copies[ticket] = Copy(source, destination, set(range(self.world_size)))
        try:
            # submit must enqueue atomically. If it raises, no worker may retain
            # the buffers; this is the TransferBackend.start failure contract.
            self.submit(ticket, source.tier, src, dst)
        except Exception:
            del self._copies[ticket]
            raise
        return ticket

    def acknowledge(self, ticket, rank):
        if not 0 <= rank < self.world_size:
            raise ValueError('unknown worker rank')
        self._copies[ticket].remaining_ranks.discard(rank)

    def ready(self, ticket):
        return not self._copies[ticket].remaining_ranks

    def release(self, location):
        record = self._allocations.get(location.handle)
        if record is None:
            return
        if record[0] != location:
            raise ValueError('mismatched allocation')
        related = [t for t, c in self._copies.items()
                   if location in (c.source, c.destination)]
        if any(self._copies[t].remaining_ranks for t in related):
            raise RuntimeError('cannot free a buffer referenced by DMA')
        self.pools[location.tier].free_blocks(record[1])
        del self._allocations[location.handle]
        for ticket in related:
            del self._copies[ticket]
