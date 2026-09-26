"""Opt-in completion observation transport and retention semantics."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import msgspec
import pytest
from bench.core.policy_driver import PolicyDriver, TupleConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.output_processor import RequestState
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind


def snapshot():
    return dict(schema_version=1, event='completion_after_free',
                monotonic_ns=123, num_gpu_blocks=101, free_queue_blocks=25,
                protected_blocks=10)


def test_snapshot_counter_capture_and_serialization():
    pool = NS(num_gpu_blocks=101, free_block_queue=NS(num_free_blocks=25),
              _protected={i: 9 for i in range(10)})
    scheduler = NS(kv_cache_manager=NS(block_pool=pool))
    snap = Scheduler._completion_kv_snapshot(scheduler)
    assert snap['free_queue_blocks'] == 25
    assert snap['protected_blocks'] == 10
    assert pool.free_block_queue.num_free_blocks == 25
    output = EngineCoreOutput(request_id='r', new_token_ids=[2], policy_kv_snapshot=snap)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(output), type=EngineCoreOutput)
    assert decoded.policy_kv_snapshot == snap
    assert EngineCoreOutput(request_id='r', new_token_ids=[]).policy_kv_snapshot is None


def test_request_output_keeps_snapshot_when_aggregating():
    state = NS(prompt_token_ids=[1], prompt_embeds=None, prompt=None,
               logprobs_processor=NS(prompt_logprobs=None),
               output_kind=RequestOutputKind.CUMULATIVE, lora_request=None,
               num_cached_tokens=0, stats=None, policy_kv_snapshot=snapshot())
    completion = CompletionOutput(index=0, text='', token_ids=[2],
                                  cumulative_logprob=None, logprobs=None)
    output = RequestState._new_request_output(state, 'r', [completion], True)
    assert output.policy_kv_snapshot == snapshot()
    state.policy_kv_snapshot = None
    first = RequestState._new_request_output(state, 'r', [completion], False)
    first.add(output, aggregate=False)
    assert first.policy_kv_snapshot == snapshot()


def test_completion_uses_snapshot_and_preserves_ttl_and_ack():
    async def check():
        driver = PolicyDriver(TupleConfig(retention='search-seed', scheduling='search-seed'),
                              [], asyncio.get_running_loop())
        try:
            driver.programs.on_turn_release('p', 0, 1)
            driver._kv_utilization = Mock(side_effect=AssertionError('extra stats RPC'))
            control = Mock()
            control.protect.return_value = 3
            driver.kv._controls = [control]
            await driver.turn_complete('p', 0, 'p:0', 1, 32, 0, 10,
                                       tool_name='tool', kv_snapshot=snapshot())
            control.protect.assert_called_once_with('p:0', 12)
            assert driver.retention_exec.policy.signals.kv_utilization == .75
            assert driver.programs.get('p').kv_protected
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(check())


def test_invalid_snapshot_rejected():
    bad = snapshot()
    bad['free_queue_blocks'] = 200
    with pytest.raises(ValueError):
        PolicyDriver._completion_utilization(bad)
