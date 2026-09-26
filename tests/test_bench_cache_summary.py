from bench.core.runner import _write_program_summary


def summary(tmp_path, records):
    _write_program_summary(tmp_path, [{'jct_ns': 2_000_000_000}], records)
    return (tmp_path / 'summary.txt').read_text()


def test_missing_streaming_cache_measurements_are_not_zero(tmp_path):
    text = summary(tmp_path, [dict(input_toks=100, cached_tokens=None, turn_idx=0),
                              dict(input_toks=200, turn_idx=1)])
    assert 'hit rate         : not measured' in text
    assert 'cached tokens    : not measured' in text
    assert 'cache measurements: 0/2 requests' in text
    assert '0.00%' not in text
    assert 'mean JCT (s)     : 2.000' in text


def test_measured_zero_is_still_reported_as_zero(tmp_path):
    text = summary(tmp_path, [dict(input_toks=100, cached_tokens=0, turn_idx=0)])
    assert 'hit rate         : 0.00%' in text
    assert 'cached tokens    : 0\n' in text
    assert 'not measured' not in text


def test_partial_coverage_does_not_create_a_full_workload_rate(tmp_path):
    text = summary(tmp_path, [dict(input_toks=100, cached_tokens=50, turn_idx=0),
                              dict(input_toks=200, cached_tokens=None, turn_idx=1)])
    assert 'hit rate         : not measured' in text
    assert 'cache measurements: 1/2 requests' in text
    assert 'turn 0 (cold)  : 50.00%  (50/100)' in text
    assert 'turns 1+       : not measured' in text


def test_streaming_summary_uses_input_reuse_not_prefix_lookup_hits(tmp_path):
    text = summary(tmp_path, [dict(input_toks=100, turn_idx=1, cached_tokens=0,
        cache_measurement='input_kv_reuse', kv_reused_tokens=80,
        gpu_reused_tokens=50, cpu_restored_tokens=30,
        prefill_computed_tokens=20, recomputed_context_tokens=10)])
    assert 'Input KV reuse' in text
    assert 'hit rate         : 80.00%' in text
    assert 'cpu_restored_tokens: 30' in text
