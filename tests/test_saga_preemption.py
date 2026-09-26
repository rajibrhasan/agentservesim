"""AFS yields a native running generation locally, then resumes it intact."""
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from bench.core import saga_scheduler
from test_vllm_policy_integration import add, make_native, step


def test_residency_survives_unpin_but_not_block_reuse(tmp_path):
    s, dma, _ = make_native(tmp_path, saga_scheduler.SagaScheduler)
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    req = add(s, 'p0', prompt=list(range(33)), output=1)
    req.sampling_params.extra_args['program_id'] = 'p'
    while req.request_id in s.requests:
        step(s, dma, runner)
    def resident():
        return s.kv_protection_stats()['policy_observation']['cached_blocks_by_program']
    assert resident()['p'] == 2
    assert not s.kv_protection_stats()['policy_observation']['protected_blocks_by_tag']
    # Fill the small native pool with different content until the physical
    # blocks have been recycled. Old block IDs must not imply old residency.
    for i in range(10):
        other = add(s, f'q{i}', prompt=[1000 + i] * 33, output=1)
        other.sampling_params.extra_args['program_id'] = f'q{i}'
        while other.request_id in s.requests:
            step(s, dma, runner)
    assert 'p' not in resident()


def test_local_preemption_waits_500ms_and_preserves_output(tmp_path, monkeypatch):
    now = [10.0]
    monkeypatch.setattr(saga_scheduler, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    s, dma, _ = make_native(tmp_path, saga_scheduler.SagaScheduler)
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    s.set_afs_shares({'low': 0.1, 'high': 0.9})
    low = add(s, 'low', output=8)
    low.sampling_params.extra_args['saga_tenant'] = 'low'
    step(s, dma, runner)
    high = add(s, 'high', prompt=[42] * 17, output=2)
    high.sampling_params.extra_args['saga_tenant'] = 'high'
    assert list(step(s, dma, runner).num_scheduled_tokens) == ['low']
    now[0] += 0.5
    assert list(step(s, dma, runner).num_scheduled_tokens) == ['low']
    now[0] += 0.001
    output = step(s, dma, runner)
    assert list(output.num_scheduled_tokens) == ['high']
    assert output.preempted_req_ids == {'low'}
    assert low.num_preemptions == 1
    for _ in range(20):
        if not s.requests:
            break
        step(s, dma, runner)
    assert not s.requests
    assert list(low.output_token_ids) == list(range(100, 108))
    assert list(high.output_token_ids) == [100, 101]
    assert all(b.is_null or b.ref_cnt == 0 for b in s.kv_cache_manager.block_pool.blocks)


def test_equal_shares_do_not_preempt(tmp_path, monkeypatch):
    s, dma, _ = make_native(tmp_path, saga_scheduler.SagaScheduler)
    s.set_afs_shares({'default': 1.0})
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    low = add(s, 'first', output=3)
    step(s, dma, runner)
    add(s, 'second', output=1).sampling_params.extra_args['saga_ready_s'] = 0.0
    assert list(step(s, dma, runner).num_scheduled_tokens) == ['first']
    assert low.num_preemptions == 0


def test_aged_high_share_uses_free_capacity_without_preemption(tmp_path):
    s, dma, _ = make_native(tmp_path, saga_scheduler.SagaScheduler)
    s.max_num_running_reqs = 2
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    s.set_afs_shares({'low': 0.1, 'high': 0.9})
    low = add(s, 'low', output=8)
    low.sampling_params.extra_args['saga_tenant'] = 'low'
    step(s, dma, runner)
    high = add(s, 'high', prompt=[42] * 17, output=2)
    high.sampling_params.extra_args.update(saga_tenant='high', saga_ready_s=0.0)
    output = step(s, dma, runner)
    assert set(output.num_scheduled_tokens) == {'low', 'high'}
    assert not output.preempted_req_ids
    assert low.num_preemptions == 0


def test_free_slot_but_no_token_budget_still_preempts(tmp_path):
    s, dma, _ = make_native(tmp_path, saga_scheduler.SagaScheduler)
    s.max_num_running_reqs = 2
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    s.set_afs_shares({'low': 0.1, 'high': 0.9})
    low = add(s, 'low', output=8)
    low.sampling_params.extra_args['saga_tenant'] = 'low'
    step(s, dma, runner)
    s.max_num_scheduled_tokens = 1
    high = add(s, 'high', prompt=[42] * 17, output=2)
    high.sampling_params.extra_args.update(saga_tenant='high', saga_ready_s=0.0)
    output = step(s, dma, runner)
    assert output.num_scheduled_tokens == {'high': 1}
    assert output.preempted_req_ids == {'low'}


def test_free_slot_but_kv_pressure_still_preempts(tmp_path):
    s, dma, _ = make_native(tmp_path, saga_scheduler.SagaScheduler)
    s.max_num_running_reqs = 2
    runner = SimpleNamespace(schedule=s.schedule, complete=lambda: None)
    s.set_afs_shares({'low': 0.1, 'high': 0.9})
    low = add(s, 'low', prompt=[11] * 224, output=8)
    low.sampling_params.extra_args['saga_tenant'] = 'low'
    for _ in range(4):
        step(s, dma, runner)
    assert low.num_computed_tokens == 224
    high = add(s, 'high', prompt=[42] * 17, output=2)
    high.sampling_params.extra_args.update(saga_tenant='high', saga_ready_s=0.0)
    output = step(s, dma, runner)
    assert output.num_scheduled_tokens == {'high': 17}
    assert output.preempted_req_ids == {'low'}
