"""Native vLLM scheduler/allocator tests; CPU DMA emulation checks data lifetime.

Run with the isolated vLLM checkout's .venv/bin/python. The GPU smoke separately
checks the actual CUDA worker operations and generated output parity.
"""
from pathlib import Path
from types import SimpleNamespace
import struct

import pytest

pytest.importorskip('vllm')
import torch
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from bench.core.policy_engine import PolicyEngine


@pytest.mark.parametrize('operation,arguments', [
    ('block_size', []),
    ('begin_export', [list(range(33)), 'receipt']),
    ('begin_import', [list(range(33)), 32, 'fingerprint', 'consumer', 'receipt']),
    ('write', ['receipt', 0, [{'rank': 0, 'data': {'kv': b'\x00\xff'}}]]),
    ('release_import', ['receipt']),
])
def test_migration_native_utility_dispatch(operation, arguments):
    """Exercise the RPC signature adapter, including wire-decoded byte payloads."""
    import msgspec
    from types import MethodType
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    core = SimpleNamespace(_agent_policy=SimpleNamespace(
        migration=lambda op, *args: (op, list(args))))
    method = MethodType(EngineCore.policy_migration, core)
    wire_args = msgspec.msgpack.decode(msgspec.msgpack.encode([operation, arguments]))
    converted = EngineCoreProc._convert_msgspec_args(method, wire_args)
    assert method(*converted) == (operation, arguments)


class DMA:
    def __init__(self, scheduler, cpu_blocks=16):
        self.s = scheduler
        self.cpu_blocks = cpu_blocks
        self.gpu = [[None] * 16 for _ in range(scheduler.kv_cache_config.num_blocks)]
        self.cpu = [[None] * 16 for _ in range(cpu_blocks)]
        self.copies = []

    def collective_rpc(self, method, args=()):
        if method == 'policy_kv_init':
            return [{'rank': 0, 'cpu_blocks': self.cpu_blocks,
                     'layout': [('layer', (64,), 'torch.int8')]}]
        if method == 'policy_model_elapsed':
            return [{'rank': 0, 'seconds': 0.1}]
        if method == 'policy_kv_read_cpu':
            values = [(-1 if value is None else value)
                      for block in args[0] for value in self.cpu[block]]
            return [{'rank': 0, 'data': {'layer': struct.pack(f'<{len(values)}i', *values)}}]
        if method == 'policy_kv_write_cpu':
            ids, payloads = args
            values = struct.unpack(f'<{len(ids) * 16}i', payloads[0]['data']['layer'])
            for i, block in enumerate(ids):
                self.cpu[block] = [None if v == -1 else v for v in values[i * 16:(i + 1) * 16]]
            return [{'rank': 0, 'blocks': len(ids)}]
        assert method == 'policy_kv_copy'
        gpu, cpu, out = args
        for g, c in zip(gpu, cpu):
            assert self.s.kv_cache_manager.block_pool.blocks[g].ref_cnt > 0
            if out:
                self.cpu[c] = self.gpu[g].copy()
            else:
                self.gpu[g] = self.cpu[c].copy()
        self.copies.append((out, tuple(gpu), tuple(cpu)))
        return [{'rank': 0, 'blocks': len(gpu)}]


@pytest.fixture
def native(tmp_path):
    return make_native(tmp_path)


def make_native(tmp_path, scheduler_type=Scheduler):
    init_none_hash(sha256)
    source = Path(__file__).resolve().parents[1] / 'configs/model/meta-llama/Llama-3.1-8B.json'
    (tmp_path / 'config.json').write_text(source.read_text())
    model = ModelConfig(model=str(tmp_path), skip_tokenizer_init=True,
                        max_model_len=256, dtype='float32')
    config = VllmConfig(model_config=model, device_config=DeviceConfig(device='cpu'),
        scheduler_config=SchedulerConfig(max_model_len=256, max_num_seqs=1,
            max_num_batched_tokens=64, enable_chunked_prefill=True,
            async_scheduling=False, is_encoder_decoder=False),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=True),
        parallel_config=ParallelConfig())
    kv = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(['layer'], FullAttentionSpec(block_size=16,
                        num_kv_heads=1, head_size=8, dtype=torch.float32))])
    config.cache_config.num_gpu_blocks = kv.num_blocks
    s = scheduler_type(vllm_config=config, kv_cache_config=kv, block_size=16,
                  structured_output_manager=StructuredOutputManager(config), log_stats=True)
    s.scheduler_reserve_full_isl = False
    dma = DMA(s)
    core = SimpleNamespace(scheduler=s, vllm_config=config,
                            model_executor=dma, batch_queue=None,
                            request_block_hasher=get_request_block_hasher(16, sha256))
    cfg = {'name': 'autellix', 'service_boundaries_s': [0.05, 1],
           'quanta_s': [0.05, 0.05, 0.05], 'starvation_ratio': 1000000,
           'cpu_bytes_per_rank': 4096}
    return s, dma, PolicyEngine(core, cfg)


def add(s, rid, prompt=None, output=4):
    params = SamplingParams(max_tokens=output, ignore_eos=True,
                            extra_args={'program_id': rid, 'kv_tag': rid})
    req = Request(request_id=rid, prompt_token_ids=prompt or list(range(17)),
                  sampling_params=params, pooling_params=None, mm_features=None,
                  arrival_time=0, block_hasher=get_request_block_hasher(16, sha256))
    s.add_request(req)
    return req


def step(s, dma, policy):
    output = policy.schedule()
    ids = list(output.num_scheduled_tokens)
    sampled = []
    for rid in ids:
        req = s.requests[rid]
        blocks = s.kv_cache_manager.get_block_ids(rid)[0]
        start = req.num_computed_tokens - output.num_scheduled_tokens[rid]
        # Every reused computed token must still hold exactly the right KV.
        for t in range(start):
            assert dma.gpu[blocks[t // 16]][t % 16] == req.all_token_ids[t]
        for t in range(start, req.num_computed_tokens):
            dma.gpu[blocks[t // 16]][t % 16] = req.all_token_ids[t]
        sampled.append([100 + len(req.output_token_ids)]
                       if req.num_computed_tokens == req.num_tokens else [])
    model = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=sampled,
        logprobs=None, prompt_logprobs_dict={}, pooler_output=[])
    s.update_from_output(output, model)
    policy.complete()
    return output


def test_preemption_swaps_real_allocator_blocks_and_preserves_tail(native):
    s, dma, p = native
    requests = [add(s, name) for name in ('a', 'b', 'c')]
    for _ in range(30):
        if not s.requests:
            break
        step(s, dma, p)
    assert not s.requests
    assert all(list(r.output_token_ids) == [100, 101, 102, 103] for r in requests)
    assert p.stats['swap_out_blocks'] > 0
    assert p.stats['swap_in_blocks'] == p.stats['swap_out_blocks']
    assert not p.swapped
    assert len(p.free_cpu) == p.cpu_capacity
    assert all(p.snapshot(name)['execution_s'] == pytest.approx(0.4) for name in ('a','b','c'))


def test_continuum_cancelled_wait_restores_original_pin_expiry(native):
    import time

    s, dma, p = native
    manager = s.kv_cache_manager
    manager.enable_kv_protection = manager.block_pool.enable_kv_protection = True
    s._kv_release_at_arrival = True
    add(s, 'previous', list(range(33)), output=1)
    step(s, dma, p)
    deadline = time.time() + 2
    assert manager.kv_protect('previous', deadline) == 2
    blocks = tuple(manager.block_pool._protected)
    params = SamplingParams(max_tokens=1, extra_args={
        'kv_release_tag': 'previous', 'kv_release_event': 'scheduled'})
    request = Request(request_id='waiting', prompt_token_ids=list(range(34)),
        sampling_params=params, pooling_params=None, mm_features=None,
        block_hasher=get_request_block_hasher(16, sha256))
    s.add_request(request)
    assert all(manager.block_pool._protected[b] == manager.KV_HOLD_DEADLINE for b in blocks)
    s.finish_requests('waiting', RequestStatus.FINISHED_ABORTED)
    assert all(manager.block_pool._protected[b] == deadline for b in blocks)


@pytest.mark.parametrize('early_result,discard_kv', [(False, False), (True, False), (False, True)])
def test_infercept_session_preserves_sampled_token_arrival_and_kv(tmp_path, early_result, discard_kv):
    from bench.core.infercept_scheduler import InferceptSessionScheduler

    s, dma, _ = make_native(tmp_path, InferceptSessionScheduler)
    native_policy = SimpleNamespace(schedule=s.schedule, complete=lambda: None)

    def turn(prompt, output, arrival):
        return Request(request_id='agent', prompt_token_ids=prompt,
            sampling_params=SamplingParams(max_tokens=output, ignore_eos=True,
                extra_args={'infercept_session': True}),
            pooling_params=None, mm_features=None, arrival_time=arrival,
            resumable=True, block_hasher=get_request_block_hasher(16, sha256))

    first = turn(list(range(17)), 1, 0)
    s.add_request(first)
    if early_result:
        s.add_request(turn([55, 56], 3, 100))
    step(s, dma, native_policy)
    original_blocks = tuple(s.kv_cache_manager.get_block_ids('agent')[0])
    if not early_result:
        assert first.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        assert 'agent' in s.intercepted_since
        assert all(s.kv_cache_manager.block_pool.blocks[b].ref_cnt > 0 for b in original_blocks)
        if discard_kv:
            s.kv_cache_manager.free(first)
            first.num_computed_tokens = 0
        s.add_request(turn([55, 56], 3, 100))
    assert list(first.all_token_ids) == list(range(17)) + [100, 55, 56]
    assert first.arrival_time == 0
    assert first.max_tokens == 3
    assert first.num_computed_tokens == (0 if discard_kv and not early_result else 17)
    assert first.prompt_token_ids == list(first.all_token_ids)
    assert 'agent' not in s.intercepted_since
    for _ in range(3):
        step(s, dma, native_policy)
    assert list(first.output_token_ids) == [100, 101, 102]
    assert first.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert not s.intercepted_since
    assert not s.requests
    assert all(b.is_null or b.ref_cnt == 0 for b in s.kv_cache_manager.block_pool.blocks)


def test_infercept_resumed_sessions_keep_original_fcfs_under_contention(tmp_path):
    from bench.core.infercept_scheduler import InferceptSessionScheduler

    s, dma, _ = make_native(tmp_path, InferceptSessionScheduler)
    native_policy = SimpleNamespace(schedule=s.schedule, complete=lambda: None)

    def turn(rid, arrival, prompt, output=1):
        return Request(request_id=rid, prompt_token_ids=prompt,
            sampling_params=SamplingParams(max_tokens=output, ignore_eos=True,
                extra_args={'infercept_session': True}),
            pooling_params=None, mm_features=None, arrival_time=arrival,
            resumable=True, block_hasher=get_request_block_hasher(16, sha256))

    # A pauses first, then B. A resumes and is preempted; B's tool result
    # arrives before the next step. Both are ready, in different native queues.
    a = turn('a', 0, list(range(17)))
    b = turn('b', 1, [42] * 17)
    s.add_request(a)
    step(s, dma, native_policy)
    s.add_request(b)
    step(s, dma, native_policy)
    assert a.status == b.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    s.add_request(turn('a', 10, [56], output=3))
    step(s, dma, native_policy)
    s.running.remove(a)
    s._preempt_request(a, 10)
    # A real preemption occupies a scheduling step without A in its output.
    # Hold spare capacity for that step so the native worker-cache bookkeeping
    # observes the same gap before A can be re-admitted.
    pool = s.kv_cache_manager.block_pool
    pressure = pool.get_new_blocks(pool.get_num_free_blocks())
    try:
        assert not step(s, dma, native_policy).num_scheduled_tokens
    finally:
        pool.free_blocks(pressure)
    s.add_request(turn('b', 11, [55]))
    output = step(s, dma, native_policy)
    assert list(output.num_scheduled_tokens) == ['a']
    s.finish_requests(['a', 'b'], RequestStatus.FINISHED_ABORTED)
    assert all(block.is_null or block.ref_cnt == 0
               for block in s.kv_cache_manager.block_pool.blocks)


def paused_infercept_native(tmp_path):
    from bench.core.infercept_connector import InferceptConnector
    from bench.core.infercept_residency import InferceptResidency
    from bench.core.infercept_scheduler import InferceptSessionScheduler
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    s, dma, _ = make_native(tmp_path, InferceptSessionScheduler)
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    request = Request(request_id='agent', prompt_token_ids=list(range(33)),
        sampling_params=SamplingParams(max_tokens=3, ignore_eos=True,
            extra_args={'infercept_session': True}),
        pooling_params=None, mm_features=None, arrival_time=0, resumable=True,
        block_hasher=get_request_block_hasher(16, sha256))
    s.add_request(request)
    for _ in range(3):
        step(s, dma, runner)
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert request.num_computed_tokens == 35
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
            'cpu_bytes_per_rank': 4096, 'scratch_blocks': 3}),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1, world_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=False), speculative_config=None)
    connector = InferceptConnector(config, KVConnectorRole.SCHEDULER, s.kv_cache_config)
    residency = InferceptResidency(s.kv_cache_manager, connector, dma.cpu_blocks)
    residency.pause(request)
    return s, dma, runner, request, residency


def acknowledge_chunk(dma, residency):
    from bench.core.infercept_connector import SwapAcknowledgement

    connector = residency.connector
    metadata = connector.build_connector_meta(None)
    plan = metadata.plan
    # Match the CUDA worker: stage stores before overwriting GPU destinations,
    # then read old host contents before committing stores to recycled slots.
    staged = [dma.gpu[block].copy() for block in plan.store_gpu]
    if plan.load_gpu:
        dma.collective_rpc('policy_kv_copy', (plan.load_gpu, plan.load_cpu, False))
    for slot, values in zip(plan.store_cpu, staged):
        dma.cpu[slot] = values
    connector.update_connector_output(SimpleNamespace(kv_connector_worker_meta=
        SwapAcknowledgement(metadata.ticket, frozenset((0,)))))
    assert residency.finish()


def test_infercept_chunked_tail_swap_restores_native_kv_after_reuse(tmp_path):
    s, dma, runner, request, residency = paused_infercept_native(tmp_path)
    pool = s.kv_cache_manager.block_pool
    original = tuple(s.kv_cache_manager.get_block_ids('agent')[0])
    residency.store_tail('agent', 1)
    assert pool.blocks[original[-1]].ref_cnt == 2
    assert not residency.finish()
    assert request.num_computed_tokens == 35
    acknowledge_chunk(dma, residency)
    assert request.num_computed_tokens == 32
    assert len(s.kv_cache_manager.get_block_ids('agent')[0]) == 2
    residency.store_tail('agent', 2)
    acknowledge_chunk(dma, residency)
    assert request.num_computed_tokens == 0
    # Reuse every free physical page. Recovery must come from host bytes, not
    # from still-valid stale GPU contents or surviving prefix-cache entries.
    pressure = pool.get_new_blocks(pool.get_num_free_blocks())
    for block in pressure:
        dma.gpu[block.block_id] = [-999] * 16
    pool.free_blocks(pressure)
    for count, computed in ((1, 16), (2, 35)):
        residency.load_prefix('agent', count)
        previous = request.num_computed_tokens
        assert not residency.finish()
        assert request.num_computed_tokens == previous
        acknowledge_chunk(dma, residency)
        assert request.num_computed_tokens == computed
    residency.resume('agent')
    assert not residency.states
    assert len(residency.free_cpu) == dma.cpu_blocks
    s.add_request(Request(request_id='agent', prompt_token_ids=[55],
        sampling_params=SamplingParams(max_tokens=1, ignore_eos=True,
            extra_args={'infercept_session': True}), pooling_params=None,
        mm_features=None, arrival_time=10, resumable=True,
        block_hasher=get_request_block_hasher(16, sha256)))
    step(s, dma, runner)  # validates every reused KV token against native tokens
    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert all(block.is_null or block.ref_cnt == 0 for block in pool.blocks)
    assert residency.stats == {'stored_blocks': 3, 'loaded_blocks': 3, 'discarded_blocks': 0}


@pytest.mark.parametrize('loading', [False, True])
def test_infercept_cancelled_chunk_keeps_dma_storage_until_ack(tmp_path, loading):
    s, dma, _, request, residency = paused_infercept_native(tmp_path)
    residency.store_tail('agent', 3)
    if loading:
        acknowledge_chunk(dma, residency)
        residency.load_prefix('agent', 2)
    pending = residency.pending
    residency.cancel('agent')
    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert all(block.ref_cnt > 0 for block in pending.gpu)
    assert not residency.finish()
    acknowledge_chunk(dma, residency)
    assert not residency.states and residency.pending is None
    assert len(residency.free_cpu) == len(set(residency.free_cpu)) == dma.cpu_blocks
    assert all(block.is_null or block.ref_cnt == 0
               for block in s.kv_cache_manager.block_pool.blocks)


def test_infercept_recompute_stops_before_cpu_resident_region(native):
    s, dma, _ = native
    s.max_num_scheduled_tokens = 16
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    request = add(s, 'recompute', prompt=list(range(65)), output=1)
    request.infercept_compute_limit = 32
    assert step(s, dma, runner).num_scheduled_tokens == {'recompute': 16}
    assert step(s, dma, runner).num_scheduled_tokens == {'recompute': 16}
    assert request.num_computed_tokens == 32
    assert not step(s, dma, runner).num_scheduled_tokens
    assert not request.output_token_ids
    request.infercept_compute_limit = None
    while s.requests:
        step(s, dma, runner)
    assert list(request.output_token_ids) == [100]


def test_infercept_mixed_discard_and_host_swap_keeps_a_recompute_boundary(tmp_path):
    s, dma, runner, request, residency = paused_infercept_native(tmp_path)
    residency.store_tail('agent', 1)
    acknowledge_chunk(dma, residency)
    residency.discard_prefix('agent')
    assert request.num_computed_tokens == 0
    assert request.infercept_compute_limit == 32
    s.add_request(Request(request_id='agent', prompt_token_ids=[55],
        sampling_params=SamplingParams(max_tokens=1, ignore_eos=True,
            extra_args={'infercept_session': True}), pooling_params=None,
        mm_features=None, arrival_time=10, resumable=True,
        block_hasher=get_request_block_hasher(16, sha256)))
    assert step(s, dma, runner).num_scheduled_tokens == {'agent': 32}
    assert request.num_computed_tokens == 32 and not request.output_token_ids
    assert not step(s, dma, runner).num_scheduled_tokens
    residency.load_prefix('agent', 1)
    acknowledge_chunk(dma, residency)
    assert request.num_computed_tokens == 35
    residency.resume('agent')
    assert request.infercept_compute_limit is None
    step(s, dma, runner)
    assert list(request.output_token_ids) == [100]
    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert len(residency.free_cpu) == dma.cpu_blocks
    assert residency.stats == {'stored_blocks': 1, 'loaded_blocks': 1, 'discarded_blocks': 2}


def test_infercept_blocked_recompute_does_not_block_other_waiters(native):
    s, dma, _ = native
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    paused = add(s, 'cpu-pending', prompt=[5] * 33, output=1)
    paused.infercept_compute_limit = 0
    add(s, 'ready', prompt=[6] * 17, output=1)
    assert step(s, dma, runner).num_scheduled_tokens == {'ready': 17}
    assert paused.num_computed_tokens == 0
    s.finish_requests('cpu-pending', RequestStatus.FINISHED_ABORTED)


def test_abort_swapped_request_releases_host_blocks(native):
    s, dma, p = native
    add(s, 'a'); add(s, 'b')
    step(s, dma, p)
    step(s, dma, p)
    assert 'a' in p.swapped
    s.finish_requests('a', RequestStatus.FINISHED_ABORTED)
    step(s, dma, p)
    assert 'a' not in p.swapped
    assert len(p.free_cpu) == p.cpu_capacity


def test_full_cpu_pool_keeps_victim_running_instead_of_recomputing(native):
    s, dma, p = native
    a = add(s, 'a'); add(s, 'b')
    step(s, dma, p)
    free = p.free_cpu
    p.free_cpu = type(free)()
    before = a.num_computed_tokens
    output = step(s, dma, p)
    p.free_cpu = free
    assert list(output.num_scheduled_tokens) == ['a']
    assert a.num_computed_tokens == before + 1
    assert a.num_preemptions == 0
    assert p.stats['cpu_capacity_holds'] == 1


def test_incomplete_worker_ack_does_not_release_gpu(native):
    s, dma, p = native
    a = add(s, 'a'); add(s, 'b')
    step(s, dma, p)
    original = s.kv_cache_manager.get_block_ids('a')
    dma.collective_rpc = lambda *args, **kwargs: []
    with pytest.raises(RuntimeError, match='acknowledgements'):
        p.schedule()
    assert s.kv_cache_manager.get_block_ids('a') == original
    assert a.num_computed_tokens == 17


def test_resident_requeue_only_appends_new_blocks_at_boundary(native):
    s, dma, p = native
    request = add(s, 'a', prompt=list(range(16)), output=4)
    step(s, dma, p)
    original = s.kv_cache_manager.get_block_ids('a')[0].copy()
    output = step(s, dma, p)
    cached = output.scheduled_cached_reqs
    assert cached.req_ids == ['a']
    assert cached.resumed_req_ids == set()
    all_blocks = s.kv_cache_manager.get_block_ids('a')[0]
    assert len(all_blocks) == len(original) + 1
    assert cached.new_block_ids == [(all_blocks[len(original):],)]
    output = step(s, dma, p)
    assert output.scheduled_cached_reqs.new_block_ids == [([],)]
    assert request.num_preemptions == 0


def test_abort_all_reclaims_cpu_without_another_scheduling_step(native):
    s, dma, p = native
    add(s, 'a'); add(s, 'b')
    step(s, dma, p)
    step(s, dma, p)
    assert p.swapped
    s.finish_requests(list(s.requests), RequestStatus.FINISHED_ABORTED)
    p.abort_completed()
    assert not s.requests
    assert not p.swapped
    assert not p._tags
    assert len(p.free_cpu) == p.cpu_capacity


def test_abort_during_output_processing_defers_accounting_until_complete(native):
    s, dma, p = native
    add(s, 'a'); add(s, 'b')
    step(s, dma, p)
    p.schedule()
    assert p.swapped
    s.finish_requests(list(s.requests), RequestStatus.FINISHED_ABORTED)
    p.abort_completed()
    p.complete()
    assert not p.runtime.calls
    assert not p.swapped
    assert len(p.free_cpu) == p.cpu_capacity


def test_swapped_chunked_prefill_resumes_without_recomputation(native):
    s, dma, p = native
    s.max_num_scheduled_tokens = 16
    requests = [add(s, name, prompt=[i] * 65) for i, name in enumerate(('a', 'b'))]
    computed = {r.request_id: 0 for r in requests}
    for _ in range(40):
        if not s.requests:
            break
        output = step(s, dma, p)
        for rid, n in output.num_scheduled_tokens.items():
            computed[rid] += n
    assert not s.requests
    assert computed == {'a': 68, 'b': 68}
    assert all(list(r.output_token_ids) == [100, 101, 102, 103] for r in requests)
    assert p.stats['swap_out_blocks'] > 0
    assert not p.swapped


def test_prefix_migration_copies_bytes_and_hands_off_real_ownership(tmp_path):
    s1, dma1, p1 = make_native(tmp_path)
    s2, dma2, p2 = make_native(tmp_path)
    request = add(s1, 'source', prompt=list(range(33)))
    while s1.requests:
        step(s1, dma1, p1)
    tokens = list(request.all_token_ids)
    export = p1.migration('begin_export', tokens)
    assert export['tokens'] == 32
    receipt = p2.migration('begin_import', tokens, export['tokens'],
                           export['fingerprint'], 'destination')
    import_handle = receipt['handle']
    probe = p2._migration._request(tokens)
    assert s2.kv_cache_manager.get_computed_blocks(probe)[1] == 0
    for offset in range(2):
        payload = p1.migration('read', export['handle'], offset, 1)
        p2.migration('write', import_handle, offset, payload)
        if offset == 0:
            with pytest.raises(RuntimeError, match='incomplete'):
                p2.migration('publish', import_handle)
            assert s2.kv_cache_manager.get_computed_blocks(probe)[1] == 0
    p2.migration('publish', import_handle)
    assert s2.kv_cache_manager.get_computed_blocks(probe)[1] == 32
    assert p1._migration.exports  # source is still held until destination ACK
    p1.migration('release_export', export['handle'])
    assert not p1._migration.exports
    add(s2, 'destination', prompt=tokens)
    output = step(s2, dma2, p2)
    assert output.num_scheduled_tokens['destination'] == len(tokens) - 32
    assert not p2._migration.imports
    while s2.requests:
        step(s2, dma2, p2)
    assert len(p1.free_cpu) == p1.cpu_capacity
    assert len(p2.free_cpu) == p2.cpu_capacity


def test_migration_rejects_incompatible_model_before_allocating(native):
    s, dma, p = native
    before = s.kv_cache_manager.block_pool.get_num_free_blocks()
    with pytest.raises(ValueError, match='matching model'):
        p.migration('begin_import', list(range(33)), 32, 'wrong', 'consumer')
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == before


def test_migration_failed_copy_keeps_destination_owned(native):
    s, dma, p = native
    p.migration('begin_export', list(range(33)))
    receipt = p.migration('begin_import', list(range(33)), 32,
                           p._migration.fingerprint, 'consumer')
    record = p._migration.imports[receipt['handle']]
    gpu = s.kv_cache_manager.get_block_ids(record.request.request_id)
    original = dma.collective_rpc
    dma.collective_rpc = lambda method, args=(): ([] if method == 'policy_kv_copy'
                                                  else original(method, args))
    payload = [{'rank': 0, 'data': {'layer': struct.pack('<16i', *range(16))}}]
    with pytest.raises(RuntimeError, match='acknowledgements'):
        p.migration('write', receipt['handle'], 0, payload)
    with pytest.raises(RuntimeError, match='unacknowledged'):
        p.migration('release_import', receipt['handle'])
    assert s.kv_cache_manager.get_block_ids(record.request.request_id) == gpu


def test_abort_migration_consumer_before_first_schedule_releases_receipt(native):
    s, dma, p = native
    p.migration('begin_export', list(range(33)))
    receipt = p.migration('begin_import', list(range(33)), 32,
                           p._migration.fingerprint, 'consumer')
    payload = [{'rank': 0, 'data': {'layer': struct.pack('<32i', *range(32))}}]
    p.migration('write', receipt['handle'], 0, payload)
    p.migration('publish', receipt['handle'])
    add(s, 'consumer', prompt=list(range(33)))
    p.before_abort(['consumer'])
    s.finish_requests('consumer', RequestStatus.FINISHED_ABORTED)
    p.abort_completed()
    assert not p._migration.imports


def test_migration_consumer_needs_only_uncached_capacity(native):
    s, dma, p = native
    p.migration('begin_export', list(range(33)))
    receipt = p.migration('begin_import', list(range(33)), 32,
                           p._migration.fingerprint, 'consumer')
    payload = [{'rank': 0, 'data': {'layer': struct.pack('<32i', *range(32))}}]
    p.migration('write', receipt['handle'], 0, payload)
    p.migration('publish', receipt['handle'])
    pool = s.kv_cache_manager.block_pool
    unrelated = pool.get_new_blocks(pool.get_num_free_blocks() - 1)
    add(s, 'consumer', prompt=list(range(33)))
    try:
        output = step(s, dma, p)
        assert output.num_scheduled_tokens == {'consumer': 1}
        assert not p._migration.imports
    finally:
        pool.free_blocks(unrelated)


@pytest.mark.parametrize('full_gpu', [False, True])
@pytest.mark.parametrize('cancelled', [None, 'agent', 'other'])
def test_infercept_joint_transfer_recycles_full_host_pool(tmp_path, cancelled, full_gpu):
    from collections import deque

    s, dma, runner, first, residency = paused_infercept_native(tmp_path)
    residency.cpu_blocks = 3
    residency.free_cpu = deque(range(3))
    residency.store_tail('agent', 3)
    acknowledge_chunk(dma, residency)
    assert not residency.free_cpu
    old_host = [values.copy() for values in dma.cpu[:3]]
    second = Request(request_id='other', prompt_token_ids=list(range(40, 73)),
        sampling_params=SamplingParams(max_tokens=3, ignore_eos=True,
            extra_args={'infercept_session': True}), pooling_params=None,
        mm_features=None, arrival_time=1, resumable=True,
        block_hasher=get_request_block_hasher(16, sha256))
    s.add_request(second)
    for _ in range(3):
        step(s, dma, runner)
    assert second.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    residency.pause(second)
    outgoing_ids = s.kv_cache_manager.get_block_ids('other')[0]
    outgoing = [dma.gpu[block].copy() for block in outgoing_ids]
    pool = s.kv_cache_manager.block_pool
    pressure = pool.get_new_blocks(pool.get_num_free_blocks()) if full_gpu else []
    if full_gpu:
        # An unrelated owner makes these pages ineligible for overwrite.
        shared = tuple(pool.blocks[i] for i in outgoing_ids)
        pool.touch(shared)
        with pytest.raises(MemoryError, match='GPU capacity'):
            residency.transfer(loads=(('agent', 3),), stores=(('other', 3),))
        assert residency.pending is None and residency.connector.pending is None
        assert all(block.ref_cnt == 2 for block in shared)
        assert dma.cpu[:3] == old_host
        pool.free_blocks(shared)
    residency.transfer(loads=(('agent', 3),), stores=(('other', 3),))
    incoming_ids = tuple(b.block_id for b in residency.pending.chunks[0].gpu)
    if full_gpu:
        assert incoming_ids == tuple(outgoing_ids)
        assert all(pool.blocks[i].block_hash is None for i in incoming_ids)
        assert pool.get_num_free_blocks() == 0
    assert not residency.finish()
    assert dma.cpu[:3] == old_host
    if cancelled:
        residency.cancel(cancelled)
        s.finish_requests(cancelled, RequestStatus.FINISHED_ABORTED)
    acknowledge_chunk(dma, residency)
    assert [dma.gpu[block] for block in incoming_ids] == old_host
    assert dma.cpu[:3] == outgoing
    assert len(residency.free_cpu) == (3 if cancelled == 'other' else 0)
    if cancelled != 'agent':
        assert first.num_computed_tokens == 35
        residency.resume('agent')
    if cancelled != 'other':
        assert second.num_computed_tokens == 0
        assert len(residency.states['other'].cpu) == 3
    for rid in ('agent', 'other'):
        residency.cancel(rid)
        s.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
    assert len(set(residency.free_cpu)) == len(residency.free_cpu) == 3
    pool.free_blocks(pressure)
    assert all(block.is_null or block.ref_cnt == 0
               for block in s.kv_cache_manager.block_pool.blocks)


def test_infercept_joint_store_preserves_shared_prefix_references(tmp_path):
    from bench.core.infercept_transfer import LayerSwapWorker

    s, dma, runner, first, residency = paused_infercept_native(tmp_path)
    second = Request(request_id='shared', prompt_token_ids=list(range(33)),
        sampling_params=SamplingParams(max_tokens=3, ignore_eos=True,
            extra_args={'infercept_session': True}), pooling_params=None,
        mm_features=None, arrival_time=1, resumable=True,
        block_hasher=get_request_block_hasher(16, sha256))
    s.add_request(second)
    for _ in range(3):
        step(s, dma, runner)
    residency.pause(second)
    ids = s.kv_cache_manager.get_block_ids('agent')[0]
    assert s.kv_cache_manager.get_block_ids('shared')[0][:2] == ids[:2]
    residency.connector.scratch_blocks = 6
    residency.transfer(stores=(('agent', 3), ('shared', 3)))
    plan = residency.connector.pending.plan
    assert len(set(plan.store_gpu)) < len(plan.store_gpu)
    # The real worker accepts repeated reads but never repeated destinations.
    LayerSwapWorker._ids(plan.store_gpu, len(dma.gpu), 'source', shared_source=True)
    with pytest.raises(ValueError, match='duplicate'):
        LayerSwapWorker._ids(plan.store_gpu, len(dma.gpu), 'destination')
    assert all(s.kv_cache_manager.block_pool.blocks[i].ref_cnt == 4 for i in ids[:2])
    acknowledge_chunk(dma, residency)
    assert first.num_computed_tokens == second.num_computed_tokens == 0
    assert not set(residency.states['agent'].cpu.values()) & set(residency.states['shared'].cpu.values())
    for rid in ('agent', 'shared'):
        residency.load_prefix(rid, 3)
        acknowledge_chunk(dma, residency)
        residency.resume(rid)
        s.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
    assert len(residency.free_cpu) == dma.cpu_blocks
    assert all(block.is_null or block.ref_cnt == 0
               for block in s.kv_cache_manager.block_pool.blocks)


def test_infercept_iteration_handoff_exposes_pages_before_ack(tmp_path):
    """Layer fences, rather than stale request refs, protect same-forward reuse."""
    from collections import deque

    s, dma, runner, incoming, residency = paused_infercept_native(tmp_path)
    residency.cpu_blocks = 4
    residency.free_cpu = deque(range(4))
    residency.store_tail('agent', 3)
    acknowledge_chunk(dma, residency)
    outgoing = Request(request_id='handoff', prompt_token_ids=list(range(40, 73)),
        sampling_params=SamplingParams(max_tokens=3, ignore_eos=True,
            extra_args={'infercept_session': True}), pooling_params=None,
        mm_features=None, arrival_time=1, resumable=True,
        block_hasher=get_request_block_hasher(16, sha256))
    s.add_request(outgoing)
    for _ in range(3):
        step(s, dma, runner)
    residency.pause(outgoing)
    pool = s.kv_cache_manager.block_pool
    pressure = pool.get_new_blocks(pool.get_num_free_blocks())
    source_ids = tuple(s.kv_cache_manager.get_block_ids('handoff')[0])

    residency.transfer_for_iteration(
        loads=(('agent', 2),), stores=(('handoff', 3),))
    pending = residency.pending
    load_ids = tuple(block.block_id for block in pending.chunks[0].gpu)
    assert set(load_ids).issubset(source_ids)
    assert s.kv_cache_manager.get_block_ids('handoff')[0] == []
    assert outgoing.num_computed_tokens == 0
    # One source page not reserved for incoming KV is available to the native
    # scheduler before DMA acknowledgement and must survive finish().
    same_forward = pool.get_new_blocks(1)
    assert same_forward[0].block_id in source_ids
    acknowledge_chunk(dma, residency)
    assert incoming.num_computed_tokens == 32
    assert len(residency.states['handoff'].cpu) == 3
    assert same_forward[0].ref_cnt == 1

    pool.free_blocks(same_forward)
    pool.free_blocks(pressure)
    for rid in ('agent', 'handoff'):
        residency.cancel(rid)
        s.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
    assert len(residency.free_cpu) == 4
    assert all(block.is_null or block.ref_cnt == 0 for block in pool.blocks)


def test_overprovisioned_call_starts_when_a_slot_frees_inside_the_window(tmp_path):
    """Autellix multi-step scheduling: with N-step replanning, the next queued
    call is admitted natively as soon as a selected call finishes, instead of
    waiting for the next replan."""
    starts = {}
    for overprovision in (0, 1):
        case = tmp_path / str(overprovision)
        case.mkdir()
        s, dma, p = make_native(case)
        s.max_num_running_reqs = 2
        cfg = {'name': 'autellix', 'service_boundaries_s': [0.05, 1],
               'quanta_s': [10, 10, 10], 'starvation_ratio': 1000000,
               'cpu_bytes_per_rank': 4096, 'replan_steps': 8,
               'overprovision_calls': overprovision}
        p = PolicyEngine(p.core, cfg)
        add(s, 'a', output=2)
        add(s, 'b', output=6)
        add(s, 'c', output=1)
        for step_index in range(12):
            if not s.requests:
                break
            output = step(s, dma, p)
            if 'c' in output.num_scheduled_tokens and 'c' not in starts:
                starts[overprovision] = step_index
        assert not s.requests
        assert p.stats['overprovisioned_admissions'] == overprovision
        starts.setdefault(overprovision, None)
    # a finishes at step 2. With overprovisioning c is admitted natively right
    # after; without it c waits until the selection empties (b finishes at
    # step 6) and a replan runs.
    assert starts[1] is not None and starts[1] <= 3
    assert starts[0] is not None and starts[0] > starts[1]


def test_migration_host_moves_a_prefix_between_stock_scheduled_engines(tmp_path):
    """SAGA engines keep native scheduling; the host supplies only the transport."""
    from bench.core.policy_migration import MigrationHost

    s1, dma1, p1 = make_native(tmp_path)
    s2, dma2, p2 = make_native(tmp_path)
    config = {'name': 'kv-migration', 'cpu_bytes_per_rank': 4096, 'time_model_calls': False}
    hosts = [MigrationHost(p1.core, config), MigrationHost(p2.core, config)]
    runners = [SimpleNamespace(schedule=h.schedule, complete=h.complete) for h in hosts]
    request = add(s1, 'source', prompt=list(range(33)))
    while s1.requests:
        step(s1, dma1, runners[0])
    tokens = list(request.all_token_ids)
    export = hosts[0].migration('begin_export', tokens)
    receipt = hosts[1].migration('begin_import', tokens, export['tokens'],
                                 export['fingerprint'], 'destination')
    for offset in range(2):
        hosts[1].migration('write', receipt['handle'], offset,
                           hosts[0].migration('read', export['handle'], offset, 1))
    hosts[1].migration('publish', receipt['handle'])
    hosts[0].migration('release_export', export['handle'])
    add(s2, 'destination', prompt=tokens)
    output = step(s2, dma2, runners[1])
    assert output.num_scheduled_tokens['destination'] == len(tokens) - 32
    while s2.requests:
        step(s2, dma2, runners[1])
    assert not hosts[1]._migration.imports
    assert len(hosts[0].free_cpu) == hosts[0].cpu_capacity
    assert hosts[0].snapshot()['cpu_blocks_used'] == 0 and hosts[0].stats['copied_blocks'] == 2
    with pytest.raises(KeyError):
        hosts[0].snapshot('some-tag')
    with pytest.raises(ValueError, match='time model calls'):
        MigrationHost(p1.core, {**config, 'time_model_calls': True})
