
from collections import deque
from dataclasses import dataclass
import asyncio
import hashlib
import json
import math
import time
import uuid


@dataclass
class Export:
    blocks: tuple
    deadlines: dict
    uncertain: bool = False


@dataclass
class Import:
    request: object
    tokens: int
    received_blocks: int = 0
    published: bool = False
    consumer_tag: str = ''
    seen: bool = False
    uncertain: bool = False


class PrefixMigration:
    def __init__(self, engine, layouts):
        self.engine = engine
        self.exports = {}
        self.imports = {}
        if not layouts or any(layout != layouts[0] for layout in layouts):
            raise ValueError('migration requires matching per-rank storage layouts')
        v = engine.core.vllm_config
        identity = {
            'model': v.model_config.model,
            'revision': v.model_config.revision,
            'config': v.model_config.hf_config.to_dict(),
            'dtype': str(v.model_config.dtype),
            'kv_dtype': v.cache_config.cache_dtype,
            'block_size': engine.block_size,
            'layouts': layouts,
        }
        self.fingerprint = hashlib.sha256(json.dumps(
            identity, sort_keys=True, default=str).encode()).hexdigest()
        self.bytes_per_block = sum(
            math.prod(shape) for _, shape, _ in layouts[0])
        # Worker layouts describe raw int8 storage. Limit the total RPC payload,
        # rather than allowing a TP-wide context to become one huge message.
        self.chunk_blocks = max(1, (32 * 1024 * 1024) // (
            self.bytes_per_block * engine.world_size))

    def _request(self, tokens):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        if not tokens or len(tokens) > self.engine.core.vllm_config.model_config.max_model_len:
            raise ValueError('migration token sequence exceeds engine context limits')
        if any(not isinstance(t, int) or t < 0 for t in tokens):
            raise ValueError('migration requires actual nonnegative token IDs')
        return Request(request_id='migration-' + uuid.uuid4().hex,
                       prompt_token_ids=list(tokens),
                       sampling_params=SamplingParams(max_tokens=1),
                       pooling_params=None, mm_features=None,
                       block_hasher=self.engine.core.request_block_hasher)

    def begin_export(self, tokens, handle=None):
        handle = handle or uuid.uuid4().hex
        if handle in self.exports:
            raise ValueError('duplicate migration export handle')
        manager = self.engine.scheduler.kv_cache_manager
        request = self._request(tokens)
        blocks, count = manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1)
        if not count:
            return None
        blocks = tuple(blocks[0])
        pool = manager.block_pool
        deadlines = {b.block_id: pool._protected[b.block_id]
                     for b in blocks if b.block_id in pool._protected}
        pool.touch(blocks)
        self.exports[handle] = Export(blocks, deadlines)
        return {'handle': handle, 'tokens': count, 'fingerprint': self.fingerprint,
                'chunk_blocks': min(self.chunk_blocks, self.engine.cpu_capacity)}

    def begin_import(self, tokens, count, fingerprint, consumer_tag, handle=None):
        handle = handle or uuid.uuid4().hex
        if handle in self.imports:
            raise ValueError('duplicate migration import handle')
        if fingerprint != self.fingerprint:
            raise ValueError('migration requires matching model, TP and KV layouts')
        if not consumer_tag:
            raise ValueError('migration requires a destination consumer tag')
        size = self.engine.block_size
        if count <= 0 or count % size or count >= len(tokens):
            raise ValueError('migration must contain a reusable full-block prefix')
        request = self._request(tokens)
        manager = self.engine.scheduler.kv_cache_manager
        blocks = manager.allocate_slots(request, 0,
            num_external_computed_tokens=count, delay_cache_blocks=True)
        if blocks is None:
            raise MemoryError('destination cannot reserve the migrated prefix')
        self.imports[handle] = Import(request, count, consumer_tag=str(consumer_tag))
        return {'handle': handle,
                'chunk_blocks': min(self.chunk_blocks, self.engine.cpu_capacity)}

    def _scratch(self, count):
        if count <= 0 or count > self.chunk_blocks:
            raise ValueError('migration chunk exceeds the transfer bound')
        if count > len(self.engine.free_cpu):
            raise MemoryError('CPU staging pool is occupied by suspended requests')
        return tuple(self.engine.free_cpu.popleft() for _ in range(count))

    def read(self, handle, offset, count):
        record = self.exports[handle]
        if offset < 0 or offset + count > len(record.blocks):
            raise ValueError('export range exceeds the held prefix')
        cpu = self._scratch(count)
        gpu = [b.block_id for b in record.blocks[offset:offset + count]]
        record.uncertain = True
        self.engine._copy(gpu, cpu, True)
        # A failed worker call retains its staging reservation: it must not be
        # reused while a failed or unreachable worker might still access it.
        data = self.engine._rpc('policy_kv_read_cpu', list(cpu))
        record.uncertain = False
        data.sort(key=lambda r: r['rank'])
        self.engine.free_cpu.extend(cpu)
        return data

    def write(self, handle, offset, payloads):
        record = self.imports[handle]
        if record.published or offset != record.received_blocks:
            raise ValueError('migration chunks must arrive once in prefix order')
        if [r['rank'] for r in payloads] != list(range(self.engine.world_size)):
            raise ValueError('migration payload must contain every rank in order')
        lengths = [sum(len(value) for value in r['data'].values()) for r in payloads]
        if len(set(lengths)) != 1 or not lengths[0] or lengths[0] % self.bytes_per_block:
            raise ValueError('migration payload is not a whole number of blocks')
        count = lengths[0] // self.bytes_per_block
        if (offset + count) * self.engine.block_size > record.tokens:
            raise ValueError('migration payload exceeds destination reservation')
        cpu = self._scratch(count)
        record.uncertain = True
        replies = self.engine._rpc('policy_kv_write_cpu', list(cpu), payloads)
        if any(r['blocks'] != count for r in replies):
            raise RuntimeError('incomplete migration staging acknowledgement')
        gpu = self.engine.scheduler.kv_cache_manager.get_block_ids(
            record.request.request_id)[0][offset:offset + count]
        self.engine._copy(gpu, cpu, False)
        record.uncertain = False
        self.engine.free_cpu.extend(cpu)
        record.received_blocks += count

    def publish(self, handle):
        record = self.imports[handle]
        if record.received_blocks * self.engine.block_size != record.tokens:
            raise RuntimeError('cannot publish an incomplete migration')
        if not record.published:
            self.engine.scheduler.kv_cache_manager.cache_blocks(record.request, record.tokens)
            record.published = True

    def release_export(self, handle):
        if handle not in self.exports:
            return
        if self.exports[handle].uncertain:
            raise RuntimeError('cannot release a source with an unacknowledged transfer')
        record = self.exports.pop(handle)
        pool = self.engine.scheduler.kv_cache_manager.block_pool
        pool.free_blocks(reversed(record.blocks))
        # Preserve a surviving source policy deadline on cancellation or copy.
        # Other request references can legitimately prevent immediate parking.
        for block in reversed(record.blocks):
            deadline = record.deadlines.get(block.block_id)
            if deadline is not None and deadline > time.time():
                pool.protect_blocks([block.block_id], deadline)

    def release_import(self, handle):
        if handle not in self.imports:
            return
        if self.imports[handle].uncertain:
            raise RuntimeError('cannot release a destination with an unacknowledged transfer')
        record = self.imports.pop(handle)
        self.engine.scheduler.kv_cache_manager.free(record.request)

    def release_consumed(self):
        requests = self.engine.scheduler.requests
        by_tag = {str((r.sampling_params.extra_args or {}).get('kv_tag', r.request_id)): r
                  for r in requests.values()}
        for handle, record in tuple(self.imports.items()):
            if not record.published:
                continue
            consumer = by_tag.get(record.consumer_tag)
            if consumer is not None:
                record.seen = True
                if consumer.num_computed_tokens > 0:
                    self.release_import(handle)
            elif record.seen:
                self.release_import(handle)

    def before_abort(self, request_ids):
        for rid in request_ids:
            request = self.engine.scheduler.requests.get(rid)
            if request is None:
                continue
            tag = str((request.sampling_params.extra_args or {}).get('kv_tag', rid))
            for record in self.imports.values():
                if record.consumer_tag == tag:
                    record.seen = True


async def migrate_prefix(source, destination, token_ids, consumer_tag):
    """Return a destination receipt; the caller must submit or release it.

    All engine failures propagate. Source references are released only after
    destination publication or a confirmed destination cancellation.
    """
    async def call(engine, method, *args):
        return await engine.engine_core.call_utility_async('policy_migration', method, list(args))

    handle = uuid.uuid4().hex
    receipt = None
    try:
        export = await call(source, 'begin_export', token_ids, handle)
        if export is None:
            return None
        receipt = await call(destination, 'begin_import', token_ids,
                             export['tokens'], export['fingerprint'], consumer_tag, handle)
        # Both engines have the same layout; negotiate bounded staging space.
        chunk = min(export['chunk_blocks'], receipt['chunk_blocks'])
        block_size = await call(source, 'block_size')
        blocks = export['tokens'] // block_size
        for offset in range(0, blocks, chunk):
            payload = await call(source, 'read', export['handle'], offset,
                                 min(chunk, blocks - offset))
            await call(destination, 'write', receipt['handle'], offset, payload)
        await call(destination, 'publish', receipt['handle'])
    except BaseException:
        async def cleanup():
            # Client-chosen IDs also permit cleanup when cancellation interrupts
            # a reply after the engine has already made its reservation.
            await call(destination, 'release_import', handle)
            await call(source, 'release_export', handle)
        await asyncio.shield(cleanup())
        raise
    await call(source, 'release_export', export['handle'])
    return receipt


class MigrationHost:
    """Prefix migration on a stock-scheduled engine.

    SAGA keeps native vLLM scheduling and only needs the acknowledged prefix
    transport. Installed through additional_config["agent_policy"] with
    name: kv-migration and engine_cls naming this class: schedule() is the
    native scheduler's, model calls are not timed (time_model_calls false), and
    a CPU staging pool is reserved through the policy worker extension for
    exports and imports.
    """

    def __init__(self, core, config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        if config.get('name') != 'kv-migration':
            raise ValueError('MigrationHost serves name: kv-migration')
        if config.get('time_model_calls', False):
            raise ValueError('kv-migration does not time model calls')
        self.core, self.scheduler = core, core.scheduler
        groups = self.scheduler.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec):
            raise ValueError('prefix migration requires one full-attention KV group')
        self.block_size = groups[0].kv_cache_spec.block_size
        self.world_size = core.vllm_config.parallel_config.world_size
        replies = self._rpc('policy_kv_init', int(config['cpu_bytes_per_rank']), 'batched')
        self._worker_init = sorted(replies, key=lambda r: r['rank'])
        self.cpu_capacity = min(r['cpu_blocks'] for r in replies)
        if self.cpu_capacity <= 0:
            raise ValueError('CPU staging pool has no blocks')
        self.free_cpu = deque(range(self.cpu_capacity))
        self._migration = None
        self.stats = {'copied_blocks': 0, 'copy_seconds': 0.0}

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
        self.stats['copied_blocks'] += len(gpu)
        self.stats['copy_seconds'] += max(float(r.get('seconds', 0.0)) for r in replies)

    def schedule(self):
        output = self.scheduler.schedule()
        if self._migration is not None:
            self._migration.release_consumed()
        return output

    def complete(self):
        pass

    def before_abort(self, request_ids):
        if self._migration is not None:
            self._migration.before_abort(request_ids)

    def abort_completed(self):
        if self._migration is not None:
            self._migration.release_consumed()

    def migration(self, operation, *args):
        if operation == 'afs_update':
            self.scheduler.set_afs_shares(*args)
            return True
        if operation == 'block_size':
            return self.block_size
        allowed = {'begin_export', 'begin_import', 'read', 'write', 'publish',
                   'release_export', 'release_import'}
        if operation not in allowed:
            raise ValueError('unsupported migration operation')
        if self._migration is None:
            self._migration = PrefixMigration(self, [r['layout'] for r in self._worker_init])
        return getattr(self._migration, operation)(*args)

    def snapshot(self, tag=None):
        if tag is not None:
            raise KeyError('kv-migration keeps no per-call metrics')
        return {**self.stats, 'cpu_blocks_used': self.cpu_capacity - len(self.free_cpu),
                'afs_preemptions': getattr(self.scheduler, 'afs_preemptions', 0),
                'migration_exports': len(self._migration.exports) if self._migration else 0,
                'migration_imports': len(self._migration.imports) if self._migration else 0}
