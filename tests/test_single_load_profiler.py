"""CPU checks for the profiling harness; actual batch shapes checked on GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_internal_ids_match_output_ids_without_changing_steps():
    from experiments.e2b_step_overhead.validation import externalize_trace_ids, verify_pure_window
    steps = [[dict(id='1-a8ff5566', tokens=16, computed=0, prompt=16)]]
    steps += [[dict(id='1-a8ff5566', tokens=1, computed=i, prompt=16)]
              for i in range(16, 303)]
    mapped = externalize_trace_ids(steps, ['1'])
    assert len(verify_pure_window(mapped, ['1'], 288)['decode_contexts']) == 287
    assert steps[0][0]['id'] == '1-a8ff5566'
    with pytest.raises(ValueError, match='Unmatched'):
        externalize_trace_ids(steps, ['2'])
    with pytest.raises(ValueError, match='Ambiguous'):
        externalize_trace_ids(steps + [[dict(steps[0][0], id='1-deadbeef')]], ['1'])


def test_load_once_for_eight_cases(monkeypatch, tmp_path):
    from experiments.e2b_step_overhead import run_single_load as mod
    boots, cells = [], []
    engine = object()

    def llm(**kwargs):
        boots.append(kwargs)
        return engine

    def measure(args, instance, trace, control):
        assert instance is engine
        assert Path(control).exists()
        cells.append((args.n_decode, args.pc, args.decoder_prompt))

    monkeypatch.setitem(sys.modules, 'vllm', types.SimpleNamespace(LLM=llm))
    monkeypatch.setattr(mod, 'measure_cell', measure)
    monkeypatch.setattr(sys, 'argv', ['profile', '--out', str(tmp_path)])
    mod.main()
    assert len(boots) == 1
    assert len(set(cells)) == 8
    assert boots[0]['async_scheduling'] is False
    assert boots[0]['max_num_seqs'] == 128
    assert boots[0]['max_num_batched_tokens'] == 16384


def test_limits_change_only_when_drained(monkeypatch, tmp_path):
    class Base:
        def add_request(self, request):
            self.requests[request] = request

        def schedule(self):
            return self.max_num_running_reqs

    parent = 'experiments.e2b_step_overhead.scheduler_trace'
    monkeypatch.setitem(sys.modules, parent,
                        types.SimpleNamespace(SynchronousProfilingScheduler=Base))
    name = 'experiments.e2b_step_overhead._test_single_load_scheduler'
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'experiments/e2b_step_overhead/single_load_scheduler.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scheduler = mod.SingleLoadScheduler()
    scheduler.requests = {}
    scheduler.scheduler_config = types.SimpleNamespace(
        max_num_seqs=128, max_num_batched_tokens=16384)
    control = tmp_path / 'limits.json'
    monkeypatch.setenv('E2B_SCHEDULER_LIMITS', str(control))
    control.write_text(json.dumps(dict(sequences=2, tokens=2048)))
    scheduler.add_request('first')
    assert (scheduler.max_num_running_reqs, scheduler.max_num_scheduled_tokens) == (2, 2048)
    control.write_text(json.dumps(dict(sequences=9, tokens=1032)))
    scheduler.add_request('second')
    assert scheduler.max_num_running_reqs == 2
    scheduler.requests.clear()
    scheduler.add_request('next-window')
    assert (scheduler.max_num_running_reqs, scheduler.max_num_scheduled_tokens) == (9, 1032)
    scheduler.running = []
    assert scheduler.schedule() == 8
    scheduler.running = [types.SimpleNamespace(
        num_computed_tokens=0, num_prompt_tokens=16) for _ in range(8)]
    assert scheduler.schedule() == 8
    for request in scheduler.running:
        request.num_computed_tokens = 16
    assert scheduler.schedule() == 9
    scheduler.requests.clear()
    control.write_text(json.dumps(dict(sequences=129, tokens=1032)))
    with pytest.raises(ValueError, match='buffer sizes'):
        scheduler.add_request('invalid')
