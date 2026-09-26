"""Tool learning excludes harness delays without altering the release clock."""
import asyncio
import pytest

from policies.program import ProgramTable
from policies.continuum import ContinuumKV
from policies.saga import SagaToolTTL
from bench.core.policy_driver import PolicyDriver, TupleConfig


@pytest.mark.parametrize('duration', [0., .2, 3.])
def test_completed_observation_is_consumed_once(duration):
    table = ProgramTable()
    table.on_turn_release('p', 0, 0)
    table.on_turn_complete('p', 0, now=1, tool_name='sed')
    assert table.tool_mean_gap_s('sed') is None  # no future tool information
    table.observe_completed_tool('p', duration)
    table.on_turn_release('p', 1, 101)
    table.on_turn_release('p', 1, 102)  # routing and scheduling may both release
    assert table.tool_mean_gap_s('sed') == duration
    assert table.get('p').gap_n == 1
    assert table.get('p').completed_tool_duration_s is None


def test_elapsed_gap_fallback_and_saga_explicit_observation():
    table = ProgramTable()
    table.on_turn_complete('p', 0, now=1, tool_name='sed')
    table.on_turn_release('p', 1, 4)
    assert table.tool_mean_gap_s('sed') == 3
    table.on_turn_complete('p', 1, now=5, tool_name='sed')
    table.observe_completed_tool('p', .2)
    saga = SagaToolTTL(2.)
    saga.observe_arrival(table.get('p'), 100)
    assert saga.predicted_gap_s('sed') == pytest.approx(.2)


@pytest.mark.parametrize('delay', [0., 10.])
@pytest.mark.parametrize('policy', ['continuum', 'gate'])
def test_replay_driver_delay_cannot_change_tool_mean_or_pin(delay, policy):
    async def run():
        driver = PolicyDriver(TupleConfig(retention=policy,
                                         scheduling=policy),
                              [], asyncio.get_running_loop())
        # No GPU needed: exercise the real serialized worker and arrival path.
        driver._route_sync = lambda *args: 0
        driver._kv_utilization = lambda instance: .5
        try:
            driver.programs.on_turn_release('p', 0, 0)
            driver.programs.on_turn_complete('p', 0, now=1, tool_name='sed')
            await driver.turn_ready('p', 1, 1.2 + delay,
                                    completed_tool_duration_s=.2)
            assert driver.scheduling_exec.stamps[-1].ts == 1.2 + delay
            assert driver.programs.tool_mean_gap_s('sed') == pytest.approx(.2)
            if policy == 'gate':
                assert driver.retention_exec.policy.gap_ema_by_tool['sed'] == pytest.approx(.2)
            pcb = driver.programs.on_turn_complete('p', 1, now=20, tool_name='sed')
            decision = driver.retention_exec.policy.on_turn_complete(pcb, 'p:1', 20)
            assert decision[:2] == ('protect', 22.)  # deadline not shifted
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(run())


@pytest.mark.parametrize('duration', [-1., float('inf'), float('nan')])
def test_invalid_observation_rejected(duration):
    with pytest.raises(ValueError):
        ProgramTable().observe_completed_tool('p', duration)


@pytest.mark.parametrize('routing', [None, 'session-affinity'])
@pytest.mark.parametrize('policy', ['continuum', 'gate'])
def test_simulator_explicit_sample_survives_routing_and_stamping(routing, policy):
    from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
    adapter = UnifiedPolicyAdapter(retention_value=policy,
                                   scheduling_value=policy,
                                   routing_value=routing, num_instances=1,
                                   block_size=16)
    table = adapter.programs
    table.on_turn_release('p', 0, 0)
    table.on_turn_complete('p', 0, now=1, tool_name='sed')
    row = dict(session_id='p', sub_request_index=1, index=1,
               arrival_time_ns=11_200_000_000, completed_tool_duration_s=.2)
    adapter.select_instance(row, lambda: 0, row['arrival_time_ns'])
    adapter.on_turn_routed(row)
    assert table.tool_mean_gap_s('sed') == pytest.approx(.2)
    assert table.get('p').gap_n == 1
    if policy == 'gate':
        assert adapter.retention_exec.policy.gap_ema_by_tool['sed'] == pytest.approx(.2)
