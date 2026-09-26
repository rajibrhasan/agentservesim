"""The streaming session driver keeps one request per program across tool gaps."""
import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from vllm import SamplingParams
from vllm.engine.protocol import StreamingInput

from bench.core.session_driver import infercept_engine_kwargs, submit_all_sessions


class FakeEngine:
    """Streams one token per step for each StreamingInput it receives."""
    def __init__(self):
        self.turns = []

    async def generate(self, inputs, sampling_params, request_id):
        async for update in inputs:
            assert isinstance(update, StreamingInput)
            assert update.sampling_params.extra_args['infercept_session'] is True
            n_out = update.sampling_params.max_tokens
            self.turns.append((request_id, len(update.prompt['prompt_token_ids']), n_out))
            for k in range(n_out):
                await asyncio.sleep(0)
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[k])], finished=False)
        yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[])], finished=True)


class RecordingDriver:
    def __init__(self):
        self.ready, self.complete = [], []

    async def turn_ready(self, sid, turn, now, prompt_tokens=None):
        self.ready.append((sid, turn, prompt_tokens))
        return 0, None

    async def turn_complete(self, sid, turn, tag, service_s, context, instance, now, tool_name=None):
        self.complete.append((sid, turn, tag, context, tool_name))


def sessions():
    return [
        {'session_id': 'a', 'input_mode': 'streaming-deltas', 'arrival_time_ns': 0, 'sub_requests': [
            {'input_toks': 5, 'output_toks': 3, 'input_tok_ids': [1] * 5, 'tool_duration_ns': 20_000_000, 'tool': 'bash'},
            {'input_toks': 2, 'output_toks': 2, 'input_tok_ids': [2] * 2, 'tool_duration_ns': 10_000_000},
            {'input_toks': 1, 'output_toks': 1, 'input_tok_ids': [3]},
        ]},
        {'session_id': 'b', 'input_mode': 'streaming-deltas', 'arrival_time_ns': 5_000_000, 'sub_requests': [
            {'input_toks': 4, 'output_toks': 2, 'input_tok_ids': [7] * 4},
        ]},
    ]


def test_each_program_is_one_streaming_request_with_turns_in_order():
    engine, driver = FakeEngine(), RecordingDriver()
    turns, programs = asyncio.run(submit_all_sessions([engine], sessions(), SamplingParams, driver=driver))
    assert engine.turns == [('a', 5, 3), ('b', 4, 2), ('a', 2, 2), ('a', 1, 1)] or \
        sorted(engine.turns) == sorted([('a', 5, 3), ('a', 2, 2), ('a', 1, 1), ('b', 4, 2)])
    assert [t for t in engine.turns if t[0] == 'a'] == [('a', 5, 3), ('a', 2, 2), ('a', 1, 1)]
    a = [r for r in turns if r['program_id'] == 'a']
    assert [r['turn_idx'] for r in a] == [0, 1, 2]
    assert [r['input_toks'] for r in a] == [5, 10, 13]
    assert [r['input_delta_toks'] for r in a] == [5, 2, 1]
    assert all(r['streaming_session'] and r['queued_ts'] is None for r in turns)
    assert all(r['arrival_time'] <= r['first_token_ts'] <= r['last_token_ts'] for r in turns)
    for record in turns:
        timing = record['session_timing']
        phases = [timing[name] for name in (
            'arrival_ts', 'task_started_ts', 'driver_ready_started_ts',
            'driver_ready_finished_ts', 'initial_params_started_ts',
            'initial_params_finished_ts', 'first_input_ts')]
        assert phases == sorted(phases)
    # The tool gaps separate consecutive turns of the same program.
    assert a[1]['arrival_time'] - a[0]['last_token_ts'] >= 0.02
    assert a[2]['arrival_time'] - a[1]['last_token_ts'] >= 0.01
    assert [p['program_id'] for p in programs] == ['a', 'b']
    assert programs[0]['num_turns'] == 3 and programs[0]['jct_ns'] >= 30_000_000
    assert driver.ready == [('a', 0, 5), ('b', 0, 4), ('a', 1, 2), ('a', 2, 1)] or \
        sorted(driver.ready) == sorted([('a', 0, 5), ('a', 1, 2), ('a', 2, 1), ('b', 0, 4)])
    assert [c for c in driver.complete if c[0] == 'a'] == [
        ('a', 0, 'a:0', 8, 'bash'), ('a', 1, 'a:1', 12, None), ('a', 2, 'a:2', 14, None)]


def test_full_prompt_trace_is_rejected_before_engine_submission():
    trace = sessions()
    del trace[0]['input_mode']
    engine = FakeEngine()
    with pytest.raises(ValueError, match='streaming-deltas'):
        asyncio.run(submit_all_sessions([engine], trace, SamplingParams))
    assert not engine.turns


def test_engine_submission_delay_is_recorded_and_remains_in_program_jct():
    class SlowSubmission(FakeEngine):
        async def generate(self, inputs, sampling_params, request_id):
            await asyncio.sleep(0.03)
            async for output in super().generate(inputs, sampling_params, request_id):
                yield output

    trace = sessions()[:1]
    trace[0]['sub_requests'] = trace[0]['sub_requests'][:1]
    turns, programs = asyncio.run(submit_all_sessions([SlowSubmission()], trace, SamplingParams))
    timing = turns[0]['session_timing']
    assert timing['first_input_ts'] - timing['initial_params_finished_ts'] >= 0.03
    assert programs[0]['jct_ns'] >= 30_000_000
    assert abs(programs[0]['jct_ns'] / 1e9 -
               (turns[0]['last_token_ts'] - timing['arrival_ts'])) < 1e-6


def test_a_session_the_engine_ends_early_is_an_error():
    class Truncating(FakeEngine):
        async def generate(self, inputs, sampling_params, request_id):
            update = await inputs.__anext__()
            for k in range(update.sampling_params.max_tokens):
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[k])], finished=False)
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[])], finished=True)

    with pytest.raises(RuntimeError, match='after 1 of 3 turns'):
        asyncio.run(submit_all_sessions([Truncating()], sessions()[:1], SamplingParams))


def test_infercept_engine_kwargs_are_explicit_measured_inputs(tmp_path):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'a': 1.0, 'c': 1.0, 'S': 64}))
    good = {'name': 'infercept', 'profile': str(profile), 'bandwidth_bytes_s': 5e10,
            'cpu_bytes_per_rank': 1 << 30}
    kwargs = infercept_engine_kwargs(good, num_instances=1, observations_scheduler=False)
    assert kwargs['scheduler_cls'].endswith('InferceptPolicyScheduler')
    assert kwargs['async_scheduling'] is False
    assert kwargs['kv_transfer_config'].kv_connector == 'InferceptConnector'
    assert kwargs['kv_transfer_config'].kv_connector_extra_config == {
        'cpu_bytes_per_rank': 1 << 30, 'scratch_blocks': 2}
    assert kwargs['additional_config']['infercept_policy']['bandwidth_bytes_s'] == 5e10
    zero = infercept_engine_kwargs(
        {**good, 'cpu_bytes_per_rank': 0, 'scratch_blocks': 0},
        num_instances=1, observations_scheduler=False)
    assert zero['kv_transfer_config'].kv_connector_extra_config == {
        'cpu_bytes_per_rank': 0, 'scratch_blocks': 0}
    for bad, error in [
        ({**good, 'cpu_bytes_per_rank': 0}, ValueError),
        ({**good, 'scratch_blocks': 0}, ValueError),
        ({k: v for k, v in good.items() if k != 'bandwidth_bytes_s'}, ValueError),
        ({**good, 'bandwidth_bytes_s': 0}, ValueError),
        ({**good, 'profile': str(tmp_path / 'missing.json')}, FileNotFoundError),
    ]:
        with pytest.raises(error):
            infercept_engine_kwargs(bad, num_instances=1, observations_scheduler=False)
    with pytest.raises(ValueError, match='single-engine'):
        infercept_engine_kwargs(good, num_instances=2, observations_scheduler=False)
    with pytest.raises(ValueError, match='observations'):
        infercept_engine_kwargs(good, num_instances=1, observations_scheduler=True)


def test_explicit_full_prompts_are_not_counted_as_deltas():
    trace = sessions()[:1]
    trace[0]['input_mode'] = 'full-prompts'
    engine = FakeEngine()
    turns, programs = asyncio.run(submit_all_sessions([engine], trace, SamplingParams))
    assert [r['input_toks'] for r in turns] == [5, 2, 1]
    assert all(r['input_delta_toks'] is None for r in turns)
    assert programs[0]['num_turns'] == 3
