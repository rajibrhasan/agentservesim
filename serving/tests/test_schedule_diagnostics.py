import json
from types import SimpleNamespace as NS

from bench.core.schedule_diagnostics import attach


def test_real_trace_preserves_selected_scheduler_and_output(tmp_path):
    request = NS(request_id='p', num_computed_tokens=0, num_cached_tokens=0, status='WAITING',
                 num_prompt_tokens=16, num_tokens=16, num_output_tokens=0, num_preemptions=0)
    class Engine:
        def __init__(self):
            self.requests = {'p': request}
            self.running, self.waiting = [], [request]
            self.calls = 0
            self.kv_cache_manager = NS(block_pool=NS(
                free_block_queue=NS(num_free_blocks=4), num_gpu_blocks=8,
                _protected={2: 100}, protection_stats={'reclaimed_forced': 0}))

        def schedule(self):
            self.calls += 1
            assert self.kv_cache_manager.allocate_slots(request) is allocation
            self.running, self.waiting = self.waiting, []
            request.num_computed_tokens = 16
            return result

    result = NS(num_scheduled_tokens={'p': 16})
    allocation = []
    engine = Engine()
    original = lambda request: allocation
    engine.kv_cache_manager.allocate_slots = original
    engine = attach(engine, tmp_path)
    assert engine.schedule() is result
    assert engine.calls == 1 and isinstance(engine, Engine)
    row = json.loads(next(tmp_path.glob('real-*.jsonl')).read_text())
    assert row['before']['waiting'] == ['p']
    assert row['after']['running'] == ['p']
    assert row['before']['requests']['p']['computed'] == 0
    assert row['after']['requests']['p']['computed'] == 16
    assert row['scheduled_tokens'] == {'p': 16}
    assert row['allocation_attempts'][0]['success'] is True
    assert engine.kv_cache_manager.allocate_slots is original


def test_sim_trace_preserves_batch_and_reservations(monkeypatch, tmp_path):
    from serving.tests.test_request_cache_recovery import scheduler, MODEL
    def step():
        engine = scheduler()
        engine.add_request([1, MODEL, 16, 18, 0, 0])
        batch = engine.schedule(0, 0)
        return batch.scheduled_tokens, engine.memory.npu_reserved
    baseline = step()
    monkeypatch.setenv('SIM_SCHEDULE_TRACE_DIR', str(tmp_path))
    assert step() == baseline
    row = json.loads(next(tmp_path.glob('sim-*.jsonl')).read_text())
    assert row['scheduled_tokens'] == {'1': 16}
    assert row['before']['requests'][0]['admitted'] is None
    assert row['after']['kv_reserved_bytes'] == baseline[1]


def test_compressed_trace_round_trip(monkeypatch, tmp_path):
    import gzip
    from runtime.schedule_trace import ScheduleTrace
    monkeypatch.setenv('SCHEDULE_TRACE_GZIP', '1')
    trace = ScheduleTrace(tmp_path, 'diagnostic')
    trace.write(event='sample', scheduled_tokens={'p': 16})
    trace.stream.close()
    with gzip.open(next(tmp_path.glob('*.gz')), 'rt') as f:
        assert json.loads(f.readline())['scheduled_tokens'] == {'p': 16}


def test_emitted_execution_trace_records_work_and_transfers(tmp_path):
    from runtime.schedule_trace import trace_execution
    batch = NS(batch_id=1, batch_time=50, total_len=17, prefill_q_list=[16],
               prefill_k_list=[32], decode_k_list=[100], load=64, host_store_bytes=32,
               host_link_bytes_s=1e9, requests=[])
    trace_execution(str(tmp_path), 0, batch, [
        ['qkv', '100', 'LOCAL', '0', 'LOCAL', '0', 'LOCAL', '0', 'NONE', '0', 'NONE'],
        ['attention', '200', 'LOCAL', '0', 'LOCAL', '0', 'LOCAL', '0', 'NONE', '0', 'NONE']])
    row = json.loads(next(tmp_path.glob('execution-*.jsonl')).read_text())
    assert row['emitted_compute_ns'] == 300
    assert row['prefill_queries'] == [16] and row['decode_contexts'] == [100]
    assert row['host_store_bytes'] == 32 and row['host_load_bytes'] == 64
