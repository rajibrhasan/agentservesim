import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from bench.core.tool_timing import wait_for_tool
from bench.core.policy_driver import PolicyDriver
from bench.core.runner import _submit_all_chain


@pytest.mark.parametrize('now,expected_sleep', [(10., .2), (10.15, .05), (12., None)])
def test_tool_sleep_uses_completion_deadline(now, expected_sleep):
    slept = []
    async def sleep(seconds):
        slept.append(seconds)
    deadline = asyncio.run(wait_for_tool(10., 200_000_000, clock=lambda: now, sleep=sleep))
    assert deadline == pytest.approx(10.2)
    if expected_sleep is None:
        assert slept == []
    else:
        assert slept == pytest.approx([expected_sleep])


def test_continuum_avoids_unused_utilization_rpc_but_other_policies_keep_it():
    driver = PolicyDriver.__new__(PolicyDriver)
    driver.cfg = NS(retention='continuum')
    driver._pool_view = Mock(return_value=(.8, 10, 16))
    assert driver._kv_utilization(0) is None
    driver._pool_view.assert_not_called()
    driver.cfg.retention = 'saga-tool-ttl'
    assert driver._kv_utilization(0) == .8
    driver._pool_view.assert_called_once_with(0)


@pytest.mark.parametrize('snapshots', [False, True])
def test_chain_keeps_callback_order_and_real_lateness_in_jct(monkeypatch, snapshots):
    monkeypatch.setenv('BENCH_COMPLETION_KV_SNAPSHOT', '1' if snapshots else '0')
    snapshot = dict(schema_version=1, event='completion_after_free',
                    num_gpu_blocks=101, free_queue_blocks=25,
                    protected_blocks=10, monotonic_ns=123)
    class Engine:
        async def generate(self, prompt, params, request_id, priority):
            now = asyncio.get_running_loop().time()
            yield NS(metrics=NS(arrival_time=now, queued_ts=now, scheduled_ts=now,
                                first_token_ts=now, last_token_ts=now),
                     num_cached_tokens=0, outputs=[NS(token_ids=[1])],
                     policy_kv_snapshot=snapshot if snapshots else None)
    class Driver:
        cfg = NS(engine_policy=None)
        has_admit = False
        def __init__(self): self.events = []
        async def turn_ready(self, sid, turn, now, **kwargs):
            self.events.append(('ready', turn))
            return 0, None
        async def turn_complete(self, sid, turn, *args, **kwargs):
            assert kwargs['kv_snapshot'] == (snapshot if snapshots else None)
            await asyncio.sleep(.02)
            self.events.append(('complete', turn))
    async def run():
        driver = Driver()
        turns, programs = await _submit_all_chain([Engine()], [dict(session_id='p', arrival_time_ns=0,
            sub_requests=[dict(input_toks=1, input_tok_ids=[1], output_toks=1,
                               tool_duration_ns=1_000_000, tool='sed')] * 2)], NS, dict, driver)
        assert driver.events == [('ready', 0), ('complete', 0), ('ready', 1), ('complete', 1)]
        assert turns[0]['tool_ready_target_ts'] == pytest.approx(turns[0]['last_token_ts'] + .001)
        assert turns[0]['tool_wait_finished_ts'] >= turns[0]['completion_callback_finished_ts']
        assert turns[1]['policy_ready_ts'] >= turns[0]['completion_callback_finished_ts']
        assert programs[0]['jct_ns'] >= 20_000_000
        if snapshots:
            assert turns[0]['completion_kv_snapshot'] == dict(snapshot, instance=0)
    asyncio.run(run())
