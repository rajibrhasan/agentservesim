from types import SimpleNamespace as NS

import pytest

from serving.core import trace_generator as trace
from serving.core.power_model import total_ring_data


@pytest.mark.parametrize('tp,local_ep,ep_total,tp_dim,ep_dim,dispatch,combine', [
    (1, 1, 1, None, None, 'NONE', 'NONE'),
    (2, 2, 2, None, None, 'NONE', 'ALLREDUCE'),
    (2, 2, 2, [True, False], [False, True], 'NONE', 'ALLREDUCE:1,0'),
    (2, 2, 4, [True, False], [True, True], 'ALLGATHER:1,1', 'REDUCESCATTER:1,1'),
])
def test_expert_collectives_follow_dp_layout(monkeypatch, tp, local_ep,
        ep_total, tp_dim, ep_dim, dispatch, combine):
    monkeypatch.setattr(trace, 'get_device', lambda *a: 'LOCAL')
    monkeypatch.setattr(trace, '_lookup_moe', lambda *a: 100)
    monkeypatch.setattr(trace, 'calculate_sizes', lambda *a, **kw: (32, 64, 32))
    ctx = NS(ep_total=ep_total, local_ep=local_ep, tp_size=tp,
             tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=0,
             config={'hidden_size': 4096, 'num_local_experts': 16}, fp=2,
             placement={}, model='phi', perf_db={},
             gate=NS(route_ep=lambda *a: NS(local_tokens=[8]*local_ep,
                                           activated_experts=[2]*local_ep)))
    power = NS(link_data_bytes=0, npu_latencies_ns=[], dram_weight_bytes=0)
    lines=[]
    trace._emit_moe_block(ctx, NS(total_len=16, step_adjust=(1.0, None)), lines, power, 0, '0')
    assert lines[0].split()[2] == dispatch
    assert lines[-1].split()[2] == combine
    if ep_total == local_ep and ep_total > 1:
        assert int(lines[0].split()[3]) == 0
        assert int(lines[-1].split()[3]) == 16*4096*2
        assert power.link_data_bytes == total_ring_data(16*4096*2, tp)
    elif ep_total > local_ep:
        assert int(lines[0].split()[3]) == (16//ep_total)*(4096+16)*2
        assert int(lines[-1].split()[3]) == 16*4096*2
    else:
        assert int(lines[-1].split()[3]) == 0
        assert power.link_data_bytes == 0
    assert power.npu_latencies_ns == [100]


@pytest.mark.parametrize('tokens', [1, 4, 8, 1025])
def test_step_baseline_matches_emitted_expert_critical_path(monkeypatch, tokens):
    from serving.core.gate_function import GateRouter
    db = {'architecture': {'sequence': {'mlp_moe': ['moe']}, 'catalog': {}}}
    cfg = {'num_hidden_layers': 32, 'num_local_experts': 16,
           'num_experts_per_tok': 2, 'hidden_size': 4096}
    monkeypatch.setattr(trace, '_lookup_moe', lambda db, t, e: t * 100 + e * 10)
    monkeypatch.setattr(trace, 'get_device', lambda *a: 'LOCAL')
    monkeypatch.setattr(trace, 'calculate_sizes', lambda *a, **kw: (32, 64, 32))
    gate = GateRouter(0, 0, 16, 2)
    ctx = NS(ep_total=2, local_ep=2, tp_size=2, tp_dim=None, ep_dim=None,
             dp_sum_total_len=0, config=cfg, fp=2, placement={}, model='phi',
             perf_db=db, gate=gate)
    power = NS(link_data_bytes=0, npu_latencies_ns=[], dram_weight_bytes=0)
    trace._emit_moe_block(ctx, NS(total_len=tokens, step_adjust=(1.0, None)), [], power, 0, '0')
    actual = trace.step_kernel_ns(db, cfg, 2, 0, 0, tokens, 16, 16, 16,
                                 tokens, moe_ep_size=2)
    assert actual == 32 * power.npu_latencies_ns[0]
    with pytest.raises(NotImplementedError):
        trace.step_kernel_ns(db, cfg, 2, 0, 0, tokens, 16, 16, 16, tokens)


def test_signed_moe_residual_scales_expert_compute(monkeypatch):
    monkeypatch.setattr(trace, '_lookup_moe', lambda *a: 100)
    monkeypatch.setattr(trace, 'get_device', lambda *a: 'LOCAL')
    monkeypatch.setattr(trace, 'calculate_sizes', lambda *a, **kw: (32, 64, 32))
    ctx = NS(ep_total=2, local_ep=2, tp_size=2, tp_dim=None, ep_dim=None,
             dp_sum_total_len=0, config={'hidden_size': 4096, 'num_local_experts': 16},
             fp=2, placement={}, model='phi', perf_db={},
             gate=NS(route_ep=lambda *a: NS(local_tokens=[8,8], activated_experts=[2,2])))
    power = NS(link_data_bytes=0, npu_latencies_ns=[], dram_weight_bytes=0)
    lines=[]
    trace._emit_moe_block(ctx, NS(total_len=16, step_adjust=(0.75, 0)), lines, power, 0, '0')
    assert power.npu_latencies_ns == [75]
    assert all(line.split()[1] == '75' for line in lines if line.startswith('expert\t'))
    assert int(lines[-1].split()[3]) == 16*4096*2


def test_moe_negative_residual_uses_supported_kernel_baseline(monkeypatch):
    from serving.core.gate_function import GateRouter
    cfg = {'num_hidden_layers': 32, 'num_local_experts': 16,
           'num_experts_per_tok': 2, 'hidden_size': 4096}
    db = {'architecture': {'sequence': {'mlp_moe': ['moe']}, 'catalog': {}}}
    monkeypatch.setattr(trace, '_lookup_moe', lambda *a: 100)
    monkeypatch.setattr(trace, '_lookup_step_overhead', lambda *a: -800)
    monkeypatch.setenv('STEP_OVERHEAD_MODE', 'auto')
    ctx = NS(perf_db=db, config=cfg, tp_size=2, local_ep=2, ep_total=2,
             is_moe=True, gate=GateRouter(0, 0, 16, 2))
    batch = NS(step_adjust=None, prefill_chunk=0, kv_prefill=0, n_decode=4,
               kv_decode_mean=16, kv_decode_max=16, kv_decode_min=16, lm_head_len=4)
    assert trace._step_adjustment(ctx, batch) == (0.75, 0)
