from dataclasses import replace
import math

import pytest

import policies
from policies.base import PolicyConfig, SystemSignals
from policies.program import ProgramControlBlock
from policies.saga import SagaToolTTL


def pcb(tool='search', **kwargs):
    return ProgramControlBlock(program_id='p', tool_name=tool, **kwargs)


def policy(**kwargs):
    result = SagaToolTTL(cold_start_s=2, **kwargs)
    result.signals = SystemSignals(kv_utilization=0.5)
    return result


def test_learns_only_completed_gaps_and_separates_tools():
    p = policy()
    gap = pcb(in_gap=True, gap_started_ts=10)
    assert p.on_turn_complete(gap, 'p:0', 10)[1] == 12
    p.observe_arrival(gap, 18)
    assert p.on_turn_complete(gap, 'p:1', 20)[1] == pytest.approx(28)
    assert p.on_turn_complete(pcb('other'), 'p:1', 20)[1] == 22
    p.observe_arrival(gap, 18)
    assert p.on_turn_complete(gap, 'p:1', 20)[2]['samples'] == 1


def test_log_normal_percentile_and_pressure():
    p = policy(ema_weight=0.5)
    p.observe_arrival(pcb(gap_started_ts=0), 1)
    p.observe_arrival(pcb(turn_idx=1, gap_started_ts=2), 6)
    # Equal weight logs 0, ln(4): mean ln(2), variance ln(2)^2.
    expected = math.exp(math.log(2) * (1 + p._z))
    assert p.on_turn_complete(pcb(), 'p:2', 10)[1] == pytest.approx(10 + expected)
    p.signals = SystemSignals(kv_utilization=0.9)
    assert p.on_turn_complete(pcb(), 'p:2', 10)[1] == pytest.approx(10 + expected / 2)


def test_cap_after_pressure_and_invalid_observations():
    p = policy()
    p.observe_arrival(pcb(gap_started_ts=0), 1000)
    p.signals = SystemSignals(kv_utilization=1)
    assert p.on_turn_complete(pcb(), 'p:1', 0)[1] == pytest.approx(300)
    assert p.on_turn_complete(pcb(None), 'p:2', 0) is None
    with pytest.raises(ValueError, match='follow gap'):
        p.observe_arrival(pcb(gap_started_ts=10), 9)
    p.signals = SystemSignals()
    with pytest.raises(ValueError, match='measured'):
        p.on_turn_complete(pcb(), 'p:1', 0)


def test_shared_registry_keeps_legacy_fixed_ttl():
    from bench.core.policy_driver import engine_flags
    cfg = PolicyConfig(tau_s=2)
    learned = policies.resolve('kv', 'saga-tool-ttl', cfg)
    legacy = policies.resolve('kv', 'saga-ttl', cfg)
    assert isinstance(learned, SagaToolTTL)
    assert not isinstance(legacy, SagaToolTTL)
    assert engine_flags('saga-tool-ttl', 'fcfs')['kv_protection']
    with pytest.raises(ValueError, match='cold start'):
        policies.resolve('kv', 'saga-tool-ttl', replace(cfg, tau_s=None))


def test_request_and_program_adapters_construct_same_component():
    from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
    from serving.core.program_policy_adapter import build_retention
    request = UnifiedPolicyAdapter(retention_value='saga-tool-ttl', tau_s=2,
                                   scheduling_value='fcfs', routing_value=None,
                                   num_instances=1, block_size=16)
    program = build_retention('saga-tool-ttl', tau_s=2)
    assert type(request.retention_exec.policy) is type(program) is SagaToolTTL


def test_routing_does_not_erase_learning_in_request_plane():
    from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
    adapter = UnifiedPolicyAdapter(retention_value='saga-tool-ttl', tau_s=2,
                                    scheduling_value='fcfs', routing_value='rr',
                                    num_instances=2, block_size=16)
    table = adapter.programs
    table.on_turn_release('p', 0, 0)
    table.on_turn_complete('p', 0, now=1, tool_name='tool')
    row = {'index': 1, 'session_id': 'p', 'sub_request_index': 1,
           'arrival_time_ns': 9_000_000_000}
    adapter.select_instance(row, lambda: 0, 9_000_000_000)
    adapter.on_turn_routed(row)
    learned = adapter.retention_exec.policy
    assert learned._history['tool'] == (math.log(8), 0, 1)
    assert table.get('p').gap_n == 1


def test_routing_does_not_erase_learning_in_real_driver():
    import asyncio
    from unittest.mock import Mock
    from bench.core.policy_driver import PolicyDriver, TupleConfig

    async def check():
        driver = PolicyDriver(TupleConfig(retention='saga-tool-ttl', tau_s=2,
            routing='rr', scheduling='fcfs'), [Mock(), Mock()],
            asyncio.get_running_loop())
        try:
            driver._kv_utilization = lambda instance: 0.5
            table = driver.programs
            table.on_turn_release('p', 0, 0)
            table.on_turn_complete('p', 0, now=1, tool_name='tool')
            await driver.turn_ready('p', 1, 9)
            assert driver.retention_exec.policy._history['tool'] == (math.log(8), 0, 1)
            assert table.get('p').gap_n == 1
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(check())
