"""Native tests for InferCept's min-waste controller (InferceptPolicyScheduler).

These drive vLLM's real scheduler and block allocator on CPU with the DMA
emulation from test_vllm_policy_integration; the GPU smoke checks the CUDA
worker separately. Waste inputs are fixed profiles written by the test, never
measured hardware numbers, so the assertions are about the decision logic and
ownership, not about fidelity on any GPU.
"""
import json
import os
import time
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip('vllm')
import torch
from vllm.config import (CacheConfig, DeviceConfig, KVTransferConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
                                        KVCacheTensor)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from test_vllm_policy_integration import DMA, step

BLOCK = 16
BLOCK_BYTES = 64


def write_profile(tmp_path, a=1.0, c=1.0, S=64):
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({'a': a, 'c': c, 'S': S, 'method': 'test fixture'}))
    return str(path)


def make_policy_native(tmp_path, *, host_blocks=8, scratch_blocks=2,
                       bandwidth_bytes_s=1e9, profile=None, extra=None):
    from bench.core.infercept_scheduler import InferceptPolicyScheduler
    from pathlib import Path

    init_none_hash(sha256)
    source = Path(__file__).resolve().parents[1] / 'configs/model/meta-llama/Llama-3.1-8B.json'
    (tmp_path / 'config.json').write_text(source.read_text())
    model = ModelConfig(model=str(tmp_path), skip_tokenizer_init=True,
                        max_model_len=256, dtype='float32')
    policy = {'profile': profile or write_profile(tmp_path),
              'bandwidth_bytes_s': bandwidth_bytes_s}
    config = VllmConfig(
        model_config=model, device_config=DeviceConfig(device='cpu'),
        scheduler_config=SchedulerConfig(
            max_model_len=256, max_num_seqs=2, max_num_batched_tokens=64,
            enable_chunked_prefill=True, async_scheduling=False, is_encoder_decoder=False),
        cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=True),
        parallel_config=ParallelConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector='InferceptConnector', kv_role='kv_both',
            kv_connector_module_path='bench.core.infercept_connector',
            kv_connector_extra_config={'cpu_bytes_per_rank': host_blocks * BLOCK_BYTES,
                                       'scratch_blocks': scratch_blocks}),
        additional_config={'infercept_policy': policy, **(extra or {})})
    kv = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[KVCacheTensor(size=16 * BLOCK_BYTES, shared_by=['layer'])],
        kv_cache_groups=[KVCacheGroupSpec(['layer'], FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float32))])
    config.cache_config.num_gpu_blocks = kv.num_blocks
    s = InferceptPolicyScheduler(
        vllm_config=config, kv_cache_config=kv, block_size=BLOCK,
        structured_output_manager=StructuredOutputManager(config), log_stats=True)
    s.scheduler_reserve_full_isl = False
    dma = DMA(s, cpu_blocks=host_blocks - scratch_blocks)
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    return s, dma, Harness(s, dma, runner)


class Harness:
    """Drive schedule/complete and emulate the CUDA worker's transfer protocol.

    The controller queues a plan inside schedule(); the native scheduler then
    dispatches it through build_connector_meta. The worker stages outgoing
    pages layer by layer before the model writes them, so the emulation
    snapshots store sources at dispatch and acknowledges after the step.
    """
    def __init__(self, s, dma, runner):
        self.s, self.dma, self.runner = s, dma, runner
        self.staged = {}
        original = s.connector.build_connector_meta

        def build(scheduler_output):
            metadata = original(scheduler_output)
            if getattr(metadata, 'ticket', -1) >= 0:
                self.staged[metadata.ticket] = (
                    metadata.plan, [dma.gpu[b].copy() for b in metadata.plan.store_gpu])
            return metadata
        s.connector.build_connector_meta = build

    @property
    def schedule(self):
        return self.runner.schedule

    def complete(self):
        self.runner.complete()

    def acknowledge(self):
        from bench.core.infercept_connector import SwapAcknowledgement

        connector = self.s.connector
        ticket = connector.inflight
        if ticket is None or connector.acknowledged == ticket:
            return False
        plan, copies = self.staged.pop(ticket)
        if plan.load_gpu:
            self.dma.collective_rpc('policy_kv_copy', (plan.load_gpu, plan.load_cpu, False))
        for slot, values in zip(plan.store_cpu, copies):
            self.dma.cpu[slot] = values
        connector.update_connector_output(SimpleNamespace(
            kv_connector_worker_meta=SwapAcknowledgement(ticket, frozenset((0,)))))
        return True

    def step(self):
        output = step(self.s, self.dma, self.runner)
        self.acknowledge()
        if os.environ.get('INFERCEPT_TRACE'):
            self.trace(output)
        return output

    def trace(self, output):
        s = self.s
        pend = s.residency.pending
        chunks = None if pend is None else [
            (('store' if c.storing else 'load'), tuple(c.indices), tuple(b.block_id for b in c.gpu))
            for c in pend.chunks]
        reqs = {r.request_id: (r.status.name, r.num_computed_tokens, r.num_tokens,
                               r.infercept_compute_limit, len(s.kv_cache_manager.get_block_ids(r.request_id)[0]))
                for r in s.requests.values()}
        states = {rid: (sorted(st.cpu), st.discarded, st.computed_tokens)
                  for rid, st in s.residency.states.items()}
        print(f'STEP scheduled={dict(output.num_scheduled_tokens)} reqs={reqs} states={states} '
              f'pending={chunks} resumed={sorted(s._resumed)} stats={s.infercept_stats} res={s.residency.stats} '
              f'free_cpu={len(s.residency.free_cpu)} inflight={s.connector.inflight}')

    def settle(self, limit=40):
        """Step until no transfer is in flight and nothing is schedulable."""
        for _ in range(limit):
            output = self.step()
            if self.s.residency.pending is None and not output.num_scheduled_tokens:
                return
        raise AssertionError('scheduler did not settle')


def session_turn(rid, prompt, output, arrival):
    return Request(request_id=rid, prompt_token_ids=prompt,
                   sampling_params=SamplingParams(max_tokens=output, ignore_eos=True,
                                                  extra_args={'infercept_session': True}),
                   pooling_params=None, mm_features=None, arrival_time=arrival,
                   resumable=True, block_hasher=get_request_block_hasher(BLOCK, sha256))


def plain(rid, prompt, output, arrival=0):
    return Request(request_id=rid, prompt_token_ids=prompt,
                   sampling_params=SamplingParams(max_tokens=output, ignore_eos=True),
                   pooling_params=None, mm_features=None, arrival_time=arrival,
                   block_hasher=get_request_block_hasher(BLOCK, sha256))


def pause_session(s, h, rid='agent'):
    request = session_turn(rid, list(range(33)), 3, 0)
    s.add_request(request)
    for _ in range(3):
        h.step()
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert request.num_computed_tokens == 35
    assert rid in s.residency.states
    return request


def all_released(s):
    pool = s.kv_cache_manager.block_pool
    return all(block.is_null or block.ref_cnt == 0 for block in pool.blocks)


def test_policy_scheduler_requires_its_configuration(tmp_path):
    from bench.core.infercept_scheduler import InferceptPolicyScheduler
    from test_vllm_policy_integration import make_native

    with pytest.raises(ValueError, match='infercept_policy'):
        make_native(tmp_path, InferceptPolicyScheduler)


@pytest.mark.parametrize('host_blocks,scratch_blocks', [(8, 2), (0, 0)])
def test_idle_system_plans_no_transfer_and_takes_the_free_discard(tmp_path, host_blocks, scratch_blocks):
    # Nothing else runs: no forward pass can hide a transfer, so the budget is
    # zero, and chunked discard of a single-chunk context wastes nothing
    # (Eq. 4 charges only memory held while earlier chunks recompute), so it
    # beats any positive preserve waste.
    s, dma, h = make_policy_native(tmp_path, host_blocks=host_blocks, scratch_blocks=scratch_blocks)
    request = pause_session(s, h)
    h.step()
    assert s.infercept_stats['swap_plans'] == 0
    assert s.infercept_stats['unhidden_transfer_blocks'] == 0
    assert s.infercept_stats['discard_decisions'] == 1
    assert s.infercept_stats['preserve_decisions'] == 0
    assert request.num_computed_tokens == 0
    assert s.residency.stats['discarded_blocks'] == 3
    assert s.residency.states['agent'].discarded

    # The tool result arrives; the discarded context is recomputed and
    # generation continues with the same sampled history.
    s.add_request(session_turn('agent', [55], 3, 100))
    assert list(request.all_token_ids) == list(range(33)) + [100, 101, 102, 55]
    h.settle()
    assert list(request.output_token_ids) == [100, 101, 102]
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert request not in s.running
    assert s.infercept_stats['swap_plans'] == 0
    # Paused again with nothing else running: the free discard repeats.
    assert s.infercept_stats['discard_decisions'] == 2
    measurement = s.policy_metrics()['turn_kv_measurements'][1]
    assert measurement['kv_reused_tokens'] == 0
    assert measurement['prefill_computed_tokens'] == 37
    assert measurement['recomputed_context_tokens'] == 35

    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert not s.residency.states and all_released(s)
    assert len(s.residency.free_cpu) == s.residency.cpu_blocks


def test_preserves_under_load_until_the_gap_grows(tmp_path):
    # A slow forward profile makes recompute expensive for the other running
    # request; a full host pool removes the swap option, so the decision is
    # the paper's min(preserve, discard) alone.
    s, dma, h = make_policy_native(tmp_path, profile=write_profile(tmp_path, a=10.0))
    s.residency.free_cpu.clear()
    other = plain('other', [7] * 17, 12, arrival=0)
    s.add_request(other)
    request = pause_session(s, h)
    original = tuple(s.kv_cache_manager.get_block_ids('agent')[0])
    h.step()
    assert s.infercept_stats['preserve_decisions'] >= 1
    assert s.infercept_stats['discard_decisions'] == 0
    assert s.infercept_stats['swap_plans'] == 0
    assert tuple(s.kv_cache_manager.get_block_ids('agent')[0]) == original

    s.intercepted_since['agent'] = time.monotonic() - 1e6
    h.step()
    assert s.infercept_stats['discard_decisions'] == 1
    assert request.num_computed_tokens == 0
    assert s.residency.stats['discarded_blocks'] == 3

    s.add_request(session_turn('agent', [55], 3, 100))
    h.settle()
    assert list(request.output_token_ids) == [100, 101, 102]
    assert list(other.output_token_ids) == [100 + i for i in range(12)]
    assert s.infercept_stats['swap_plans'] == 0

    s.finish_requests(['agent', 'other'], RequestStatus.FINISHED_ABORTED)
    assert not s.residency.states and all_released(s)


def test_ample_gpu_capacity_suppresses_outgoing_budget(tmp_path):
    # A short measured overlap window relative to free physical capacity.
    # The old planner filled the outgoing budget even though no allocation
    # needed those pages. Keep real native allocator/scheduler execution here.
    s, dma, h = make_policy_native(tmp_path, bandwidth_bytes_s=64000)
    s.add_request(plain('other', [7] * 17, 24, arrival=0))
    pause_session(s, h)
    for _ in range(3):
        h.step()
    assert s.infercept_stats['planned_store_blocks'] == 0
    assert s.residency.stats['stored_blocks'] == 0


def test_swaps_a_paused_context_out_and_restores_it_before_the_next_turn(tmp_path):
    s, dma, h = make_policy_native(tmp_path, host_blocks=8, scratch_blocks=2)
    # A long-running request is already resident when the session pauses: its
    # context makes recompute costly (preserve wins over discard) and its
    # forward passes hide the transfers (the budget is positive).
    other = plain('other', [7] * 17, 24, arrival=0)
    s.add_request(other)
    request = pause_session(s, h)
    state = s.residency.states['agent']
    for _ in range(6):
        h.step()
        if len(state.cpu) == 3:
            break
    assert sorted(state.cpu) == [0, 1, 2], 'all three resident blocks must reach the host'
    assert s.residency.stats == {'stored_blocks': 3, 'loaded_blocks': 0, 'discarded_blocks': 0}
    assert s.infercept_stats['swap_plans'] == 2, 'staging holds two blocks, so 2 + 1'
    assert s.infercept_stats['planned_store_blocks'] == 3
    assert s.infercept_stats['unhidden_transfer_blocks'] == 0
    assert s.infercept_stats['discard_decisions'] == 0
    assert s.kv_cache_manager.get_block_ids('agent')[0] == []
    assert request.num_computed_tokens == 0

    s.add_request(session_turn('agent', [55], 3, 100))
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    for _ in range(8):
        h.step()
        if 'agent' not in s.residency.states:
            break
    assert s.residency.stats['loaded_blocks'] == 3
    assert s.infercept_stats['planned_load_blocks'] == 3
    assert s.infercept_stats['unhidden_transfer_blocks'] == 0, 'the other request hid every load'
    assert not state.cpu and 'agent' not in s.residency.states
    h.settle()
    assert list(request.output_token_ids) == [100, 101, 102]
    assert list(other.output_token_ids) == [100 + i for i in range(24)]
    assert request not in s.running
    # The session paused again while the other request still ran, so the
    # controller swapped it out a second time; that context stays on the host.
    metrics = s.policy_metrics()
    returned = next(m for m in metrics['turn_kv_measurements']
                    if m['program_id'] == 'agent' and m['turn_idx'] == 1)
    assert returned['input_toks'] == 37
    assert returned['cpu_restored_tokens'] == 35
    assert returned['gpu_reused_tokens'] == 0
    assert returned['kv_reused_tokens'] == 35
    assert returned['prefill_computed_tokens'] == 2
    assert metrics['loaded_blocks'] == 3 and metrics['stored_blocks'] == 6
    assert metrics['cpu_blocks_used'] == 3 and metrics['discard_decisions'] == 0

    s.finish_requests(['agent', 'other'], RequestStatus.FINISHED_ABORTED)
    assert all_released(s)


def test_waiting_ownership_recovery_leaves_fitting_head_alone(tmp_path):
    s, dma, h = make_policy_native(tmp_path)
    s.scheduler_reserve_full_isl = True
    younger = session_turn('younger', list(range(160)), 1, 1)
    s.add_request(younger)
    for _ in range(3):
        h.step()
    s.add_request(session_turn('younger', [777], 1, 2))
    head = plain('head', [999] * 17, 1, arrival=0)
    s.add_request(head)
    before = tuple(s.residency.single.req_to_blocks['younger'])
    s._relieve_waiting_ownership()
    assert tuple(s.residency.single.req_to_blocks['younger']) == before
    assert younger.num_preemptions == 0
    assert s.infercept_stats['waiting_ownership_preemptions'] == 0
    assert len(s.residency.free_cpu) == s.residency.cpu_blocks


def test_handed_off_store_with_discarded_remainder_recomputes_then_restores(tmp_path):
    # The other request is still waiting when the plan is made, so recompute
    # is free by Eq. 4: the budgeted tail is swapped out and the remainder is
    # discarded in the same iteration. The return recomputes up to the first
    # host chunk, restores the rest, and continues with the same history.
    s, dma, h = make_policy_native(tmp_path, host_blocks=8, scratch_blocks=2)
    request = pause_session(s, h)
    other = plain('other', [7] * 17, 6, arrival=1)
    s.add_request(other)
    h.step()
    state = s.residency.states['agent']
    assert state.discarded and request.num_computed_tokens == 0
    assert s.residency.stats['discarded_blocks'] == 1
    h.step()
    assert sorted(state.cpu) == [1, 2]
    assert s.residency.stats['stored_blocks'] == 2
    assert request.infercept_compute_limit == 16

    s.add_request(session_turn('agent', [55], 3, 100))
    # Recompute runs up to the first host chunk, the request is parked like a
    # pause, the tail is restored, and it re-enters through the waiting queue.
    for _ in range(10):
        h.step()
        if s.residency.stats['loaded_blocks'] == 2 and 'agent' not in s.residency.states:
            break
    assert s.residency.stats['loaded_blocks'] == 2
    assert request.num_computed_tokens >= 35, "restore must include the recomputed prefix"
    assert request.infercept_compute_limit is None
    h.settle()
    assert list(request.output_token_ids) == [100, 101, 102]
    assert list(other.output_token_ids) == [100 + i for i in range(6)]
    assert request not in s.running
    measured = next(m for m in s.policy_metrics()['turn_kv_measurements']
                    if m['program_id'] == 'agent' and m['turn_idx'] == 1)
    assert measured['cpu_restored_tokens'] == 19
    assert measured['gpu_reused_tokens'] == 0
    assert measured['prefill_computed_tokens'] == 18
    assert measured['recomputed_context_tokens'] == 16

    s.finish_requests(['agent', 'other'], RequestStatus.FINISHED_ABORTED)
    assert all_released(s)
    assert len(s.residency.free_cpu) == s.residency.cpu_blocks


def test_turn_measurement_for_preserved_gpu_context(tmp_path):
    s, dma, h = make_policy_native(tmp_path, profile=write_profile(tmp_path, a=10.0))
    s.residency.free_cpu.clear()
    s.add_request(plain('other', [7] * 17, 12))
    pause_session(s, h)
    s.add_request(session_turn('agent', [55], 3, 100))
    h.settle()
    measured = next(m for m in s.policy_metrics()['turn_kv_measurements']
                    if m['program_id'] == 'agent' and m['turn_idx'] == 1)
    assert measured['gpu_reused_tokens'] == 35
    assert measured['cpu_restored_tokens'] == 0
    assert measured['prefill_computed_tokens'] == 2
    assert measured['recomputed_context_tokens'] == 0


def test_abort_during_transfer_keeps_dma_storage_until_acknowledged(tmp_path):
    s, dma, h = make_policy_native(tmp_path, host_blocks=8, scratch_blocks=2)
    pause_session(s, h)
    s.add_request(plain('other', [7] * 17, 2, arrival=1))
    step(s, dma, h.runner)          # dispatches a store; deliberately not acknowledged
    assert s.residency.pending is not None and s.connector.inflight is not None
    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert s.residency.states['agent'].cancelled
    assert s.residency.pending is not None, 'cancellation must not release in-flight buffers'
    assert h.acknowledge()
    h.settle()
    assert 'agent' not in s.residency.states
    s.finish_requests('other', RequestStatus.FINISHED_ABORTED)
    assert all_released(s)
    assert len(s.residency.free_cpu) == s.residency.cpu_blocks


def test_engine_metrics_utility_falls_back_to_the_scheduler(tmp_path):
    """The engine's policy_metrics utility serves schedulers without an adapter."""
    from vllm.v1.engine.core import EngineCore

    s, _, _ = make_policy_native(tmp_path)
    core = SimpleNamespace(_agent_policy=None, scheduler=s)
    metrics = MethodType(EngineCore.policy_metrics, core)()
    assert metrics['iterations'] == 0 and metrics['paused_requests'] == 0
    with pytest.raises(RuntimeError):
        MethodType(EngineCore.policy_metrics, core)('some-tag')
    bare = SimpleNamespace(_agent_policy=None, scheduler=SimpleNamespace())
    with pytest.raises(RuntimeError):
        MethodType(EngineCore.policy_metrics, bare)()


def test_full_prompt_head_can_reclaim_younger_ready_context(tmp_path):
    # Captured RTX state: no runners/DMA; older preempted head cannot fit,
    # while younger resumed WAITING calls retain GPU ownership behind it.
    s, dma, h = make_policy_native(tmp_path)
    s.scheduler_reserve_full_isl = True
    younger = session_turn('younger', list(range(160)), 1, 1)
    s.add_request(younger)
    for _ in range(3):
        h.step()
    assert younger.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    s.add_request(session_turn('younger', [777], 1, 2))
    assert younger.status == RequestStatus.WAITING
    assert 'younger' not in s.residency.states
    head = plain('head', [999] * 113, 1, arrival=0)
    s.add_request(head)
    assert not s.running and s.residency.pending is None
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == 5
    # The pre-fix scheduler repeats an empty step without releasing owners.
    recover = s._relieve_waiting_ownership
    s._relieve_waiting_ownership = lambda: None
    for _ in range(3):
        assert not h.step().num_scheduled_tokens
    assert younger.num_preemptions == 0
    s._relieve_waiting_ownership = recover
    output = h.step()
    assert 'head' in output.num_scheduled_tokens
    assert younger.arrival_time == 1
    assert younger.num_preemptions == 1
    for _ in range(12):
        h.step()
        if 'head' not in s.requests and younger.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break
    assert 'head' not in s.requests
    assert younger.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert list(younger.output_token_ids) == [100]
    s.finish_requests('younger', RequestStatus.FINISHED_ABORTED)
    assert all_released(s)


def test_restore_progress_with_inadmissible_queued_compute(tmp_path):
    s, dma, h = make_policy_native(tmp_path)
    s.scheduler_reserve_full_isl = True
    returning = session_turn('returning', list(range(208)), 1, 0)
    s.add_request(returning)
    for _ in range(4):
        h.step()
    assert returning.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    planner = s._plan_transfers
    s._plan_transfers = lambda now: s.residency.transfer_for_iteration(
        stores=(('returning', 2),))
    h.step()
    assert s.residency.finish()
    s._plan_transfers = planner
    s.add_request(session_turn('returning', [777], 1, 1))
    younger = plain('younger', [999] * 129, 1, arrival=2)
    s.add_request(younger)
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == 4
    assert not s.running
    for _ in range(16):
        h.step()
        if returning.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break
    assert returning.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert s.infercept_stats['planned_load_blocks'] == 2
    assert list(returning.output_token_ids) == [100]
    s.finish_requests('returning', RequestStatus.FINISHED_ABORTED)
    for _ in range(8):
        h.step()
        if 'younger' not in s.requests:
            break
    assert 'younger' not in s.requests
    assert all_released(s)


def test_zero_free_blocks_reclaims_partial_restore_owner(tmp_path):
    s, dma, h = make_policy_native(tmp_path)
    s.scheduler_reserve_full_isl = True
    younger = session_turn('younger', list(range(160)), 1, 1)
    s.add_request(younger)
    for _ in range(3):
        h.step()
    planner = s._plan_transfers
    s._plan_transfers = lambda now: s.residency.transfer_for_iteration(
        stores=(('younger', 1),))
    h.step()
    assert s.residency.finish()
    s._plan_transfers = lambda now: None
    s.add_request(session_turn('younger', [777], 1, 2))
    head = session_turn('head', [999] * 96, 1, 0)
    s.add_request(head)
    for _ in range(2):
        h.step()
    assert head.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    s.add_request(session_turn('head', [888], 1, 3))
    assert younger.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    cpu_before = dict(s.residency.states['younger'].cpu)
    s._plan_transfers = planner
    recovery = s._relieve_waiting_ownership
    s._relieve_waiting_ownership = lambda: None
    for _ in range(3):
        assert not h.step().num_scheduled_tokens
        assert s.residency.pending is None
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    s._relieve_waiting_ownership = recovery
    h.step()
    assert head.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert s.residency.states['younger'].discarded
    assert s.residency.states['younger'].cpu == cpu_before
    assert younger.num_preemptions == 1
    s.finish_requests('head', RequestStatus.FINISHED_ABORTED)
    for _ in range(20):
        h.step()
        if younger.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break
    assert younger.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert list(younger.output_token_ids) == [100]
    assert s.residency.stats['loaded_blocks'] == 1
    s.finish_requests('younger', RequestStatus.FINISHED_ABORTED)
    assert all_released(s)
    assert len(s.residency.free_cpu) == s.residency.cpu_blocks


@pytest.mark.parametrize('common,new_length', [(64, 88), (16, 16), (0, 32)])
def test_recorded_full_prompt_reconciles_gpu_and_cpu_after_ack(tmp_path, common, new_length):
    s, dma, h = make_policy_native(tmp_path, host_blocks=12, scratch_blocks=3)
    request = session_turn('agent', list(range(80)), 1, 0)
    s.add_request(request)
    for _ in range(2):
        h.step()
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    planner = s._plan_transfers
    s._plan_transfers = lambda now: s.residency.transfer_for_iteration(stores=(('agent', 3),))
    h.step()
    assert s.residency.pending is not None
    s._plan_transfers = planner
    prompt = list(range(common)) + [999] * (new_length - common)
    update = session_turn('agent', prompt, 1, 2)
    update.sampling_params.extra_args['infercept_full_prompt'] = True
    s.add_request(update)
    # Replacing token history or freeing copy buffers before ACK is forbidden.
    assert list(request.all_token_ids) == list(range(80)) + [100]
    assert 'agent' in s._pending_full_prompts
    for _ in range(12):
        h.step()
        if 'agent' not in s._pending_full_prompts and request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break
    assert request.prompt_token_ids == prompt
    assert request.num_prompt_tokens == new_length
    assert list(request.all_token_ids) == prompt + [100]
    assert request.arrival_time == 0
    assert s.residency.stats['loaded_blocks'] == (2 if common == 64 else 0)
    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert all_released(s)
    assert len(s.residency.free_cpu) == s.residency.cpu_blocks


def test_zero_host_full_prompt_continuation_without_transfers(tmp_path):
    s, dma, h = make_policy_native(tmp_path, host_blocks=0, scratch_blocks=0)
    request = pause_session(s, h)
    h.step()
    prompt = list(range(33)) + [88] * 16
    update = session_turn('agent', prompt, 3, 100)
    update.sampling_params.extra_args['infercept_full_prompt'] = True
    s.add_request(update)
    h.settle()
    assert request.prompt_token_ids == prompt
    assert list(request.all_token_ids) == prompt + [100, 101, 102]
    assert s.infercept_stats['swap_plans'] == 0
    assert s.residency.stats['stored_blocks'] == s.residency.stats['loaded_blocks'] == 0
    assert s.residency.cpu_blocks == 0
    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert all_released(s)


@pytest.mark.parametrize('reserve_full', [False, True])
@pytest.mark.parametrize('cached_prompt_length', [17, 49])
def test_discarded_full_prompt_can_rematch_another_requests_cache(tmp_path, reserve_full, cached_prompt_length):
    s, dma, h = make_policy_native(tmp_path, host_blocks=0, scratch_blocks=0)
    s.scheduler_reserve_full_isl = reserve_full
    request = pause_session(s, h)
    h.step()
    assert request.num_computed_tokens == 0
    # Another call populates the cache after the session discarded its KV.
    prompt = list(range(49))
    s.add_request(plain('other', prompt[:cached_prompt_length], 1))
    h.settle()
    update = session_turn('agent', prompt, 3, 100)
    update.sampling_params.extra_args['infercept_full_prompt'] = True
    s.add_request(update)
    for _ in range(12):
        h.step()
        if len(request.output_token_ids) == 3:
            break
    assert list(request.all_token_ids) == prompt + [100, 101, 102]
    measured = next(m for m in s.policy_metrics()['turn_kv_measurements']
                    if m['program_id'] == 'agent' and m['turn_idx'] == 1)
    assert measured['gpu_reused_tokens'] >= (cached_prompt_length // BLOCK) * BLOCK
    assert s.infercept_stats['swap_plans'] == 0
    s.finish_requests('agent', RequestStatus.FINISHED_ABORTED)
    assert all_released(s)


def test_infercept_native_schedule_trace_includes_turn_and_residency(tmp_path):
    from bench.core.schedule_diagnostics import attach
    s, dma, h = make_policy_native(tmp_path, host_blocks=0, scratch_blocks=0)
    attach(s, tmp_path / 'trace')
    h.runner.schedule = s.schedule
    pause_session(s, h)
    h.step()
    rows = [json.loads(line) for line in next((tmp_path / 'trace').glob('*.jsonl')).read_text().splitlines()]
    assert any(row['after']['turns']['agent']['turn'] == 0 for row in rows)
    assert any(row['after']['requests']['agent']['generated_tokens'] == 3 for row in rows)
    assert rows[-1]['after']['infercept_counters']['discard_decisions'] == 1
    assert rows[-1]['after']['residency']['agent']['discarded']
    assert s.infercept_stats['swap_plans'] == 0


def test_discarded_history_uses_profile_chunks_before_new_prompt():
    from bench.core.infercept_scheduler import InferceptPolicyScheduler
    request = SimpleNamespace(request_id='recover', num_computed_tokens=0,
                              num_tokens=2048, status=RequestStatus.RUNNING,
                              infercept_compute_limit=None)
    decode = SimpleNamespace(num_computed_tokens=128, num_tokens=129)
    state = SimpleNamespace(request=request, cpu={}, discarded=True,
                            computed_tokens=1536)
    scheduler = SimpleNamespace(
        paper_scheduling=False,
        running=[request, decode], profile=SimpleNamespace(S=384),
        _resumed={'recover'},
        residency=SimpleNamespace(states={'recover':state}, pending=None,
                                  block_size=16),
        kv_cache_manager=SimpleNamespace(enable_caching=False))
    limits=[]
    for _ in range(4):
        InferceptPolicyScheduler._prepare_resumed(scheduler)
        limits.append(request.infercept_compute_limit)
        request.num_computed_tokens=request.infercept_compute_limit
    assert limits == [384, 768, 1152, 1536]


def test_paper_budget_and_resident_fcfs_requeue(tmp_path, monkeypatch):
    monkeypatch.setenv('INFERCEPT_PAPER_SCHEDULING', '1')
    s, dma, h = make_policy_native(tmp_path, host_blocks=0, scratch_blocks=0,
                                   profile=write_profile(tmp_path, S=16))
    younger = plain('younger', list(range(33)), 3, arrival=10)
    s.add_request(younger)
    first = h.step()
    assert first.num_scheduled_tokens == {'younger': 16}
    blocks = s.kv_cache_manager.get_block_ids('younger')
    # An older session becomes ready while younger remains partially prefetched.
    older = plain('older', list(range(100, 140)), 3, arrival=0)
    s.add_request(older)
    second = h.step()
    assert second.num_scheduled_tokens == {'older': 16}
    assert younger.num_computed_tokens == 16
    assert s.kv_cache_manager.get_block_ids('younger') == blocks
    assert younger.num_preemptions == 0
    assert younger not in s.running
    outputs = [first, second]
    for _ in range(20):
        if not s.requests:
            break
        outputs.append(h.step())
    assert not s.requests
    assert all(sum(o.num_scheduled_tokens.values()) <= 16 for o in outputs)
    assert any(len(o.num_scheduled_tokens) == 2 for o in outputs)
    assert all(not o.scheduled_cached_reqs.resumed_req_ids for o in outputs)
    assert all(list(r.output_token_ids) == [100, 101, 102] for r in (older, younger))
    assert s.max_num_scheduled_tokens == 64  # temporary cap, not config mutation


def test_paper_recovery_chunk_reserves_decode_queries():
    from bench.core.infercept_scheduler import InferceptPolicyScheduler
    req = SimpleNamespace(request_id='recover', num_computed_tokens=0,
                          is_prefill_chunk=True,
                          num_tokens=2048, status=RequestStatus.RUNNING,
                          infercept_compute_limit=None)
    state = SimpleNamespace(request=req, cpu={}, discarded=True, computed_tokens=1536)
    scheduler = SimpleNamespace(paper_scheduling=True,
        running=[req, SimpleNamespace(num_computed_tokens=128, num_tokens=129,
                                     is_prefill_chunk=False)],
        profile=SimpleNamespace(S=384), _resumed={'recover'},
        residency=SimpleNamespace(states={'recover': state}, pending=None, block_size=16),
        kv_cache_manager=SimpleNamespace(enable_caching=False))
    InferceptPolicyScheduler._prepare_resumed(scheduler)
    assert req.infercept_compute_limit == 383


@pytest.mark.parametrize('blocked_status', [
    RequestStatus.WAITING_FOR_REMOTE_KVS,
    RequestStatus.WAITING_FOR_STREAMING_REQ,
])
def test_resident_requeue_preserves_blocked_lifecycle(tmp_path, monkeypatch, blocked_status):
    """A previous partial-prefill marker must not unblock DMA or a tool wait."""
    monkeypatch.setenv('INFERCEPT_PAPER_SCHEDULING', '1')
    s, dma, h = make_policy_native(tmp_path, profile=write_profile(tmp_path, S=16))
    request = session_turn('agent', list(range(33)), 3, 0)
    s.add_request(request)
    h.step()
    h.step()
    assert request.num_computed_tokens == 32
    assert 'agent' in s._policy_resident_requeues
    s._park_for_load(request)
    request.status = blocked_status
    blocks = s.kv_cache_manager.get_block_ids('agent')
    s._requeue_partial_prefills()
    assert request.status == blocked_status
    assert 'agent' not in s._policy_resident_requeues
    assert s.kv_cache_manager.get_block_ids('agent') == blocks


def test_paper_partial_recompute_reaches_cpu_restore(tmp_path, monkeypatch):
    """Several recovery chunks must hand off to DMA and finish the next turn."""
    monkeypatch.setenv('INFERCEPT_PAPER_SCHEDULING', '1')
    s, dma, h = make_policy_native(tmp_path, host_blocks=8, scratch_blocks=2,
                                   profile=write_profile(tmp_path, S=16))
    request = session_turn('agent', list(range(65)), 3, 0)
    s.add_request(request)
    for _ in range(12):
        h.step()
        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    other = plain('other', [777] * 17, 6, arrival=1)
    # Select discard deterministically; this test exercises recovery, not
    # elapsed-time-dependent preserve/discard cost estimation.
    monkeypatch.setattr(s, '_waste', lambda state, now: (100.0, 100.0, 0.0))
    s.add_request(other)
    h.step()
    h.step()
    state = s.residency.states['agent']
    assert state.discarded and state.cpu
    assert min(state.cpu) * BLOCK > s.profile.S
    s.add_request(session_turn('agent', [55], 3, 100))
    for _ in range(40):
        h.step()
        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert list(request.output_token_ids) == [100, 101, 102]
    assert s.residency.stats['loaded_blocks'] >= 2
    assert s.num_waiting_for_streaming_input == 1
