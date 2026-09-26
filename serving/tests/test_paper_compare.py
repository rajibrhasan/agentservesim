import json

import pytest

from experiments.paper.compare import compare
from experiments.paper.matrix import sha256


@pytest.fixture
def pair(tmp_path):
    def write(name, obj):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
    trace = tmp_path / 'trace.jsonl'
    trace.write_text(json.dumps(dict(session_id='p', arrival_time_ns=1_000_000_000,
        sub_requests=[dict(input_toks=16, output_toks=2, tool_duration_ns=0)])) + '\n')
    cluster = {'nodes': [{'instances': [{}]}]}
    spec = dict(cell='test', smoke=True, policy='autellix-proxy', model='test-model',
        tp=1, dtype='bfloat16', instances=1, max_model_len=64, max_num_seqs=2,
        max_num_batched_tokens=32, block_size=16, cluster=cluster,
        actual_dataset=str(trace), actual_dataset_sha256=sha256(trace),
        simulator_dataset=str(trace), simulator_dataset_sha256=sha256(trace))
    write('run-spec.json', spec)
    write('real/meta.json', dict(model='test-model', dataset_hash=sha256(trace),
        num_requests=1, policy={}, engine_kwargs=dict(tensor_parallel_size=1,
        dtype='bfloat16', effective_instances=[dict(instance=0, kv_cache_tokens=64, kv_block_size=16,
        max_model_len=64, max_num_seqs=2, max_num_batched_tokens=32,
        scheduler_reserve_full_isl=True, async_scheduling=False)])))
    cluster['nodes'][0]['instances'][0]['kv_pool_tokens'] = 64
    write('matched-cluster.json', cluster)
    write('real/requests.jsonl', dict(program_id='p', turn_idx=0, input_toks=16, output_toks=2))
    (tmp_path / 'real/per_program.csv').write_text(
        'program_id,arrival_s,jct_s,num_turns\np,1,2,1\n')
    (tmp_path / 'run.csv').write_text(
        'program_id,turn_idx,input,output,end_time\np,0,16,2,2500000000\n')
    write('sim-card.json', dict(error=None, n=1, jct_mean=1.5, outdir=str(tmp_path)))
    return tmp_path


def test_jct_comparison_uses_program_arrival_and_checks_completed_tokens(pair):
    result = compare(pair)
    assert result['signed_error_pct'] == -25
    assert result['comparison_checks_passed']
    assert result['smoke']


def test_rejects_missing_turn_even_if_scorecard_claims_completion(pair):
    (pair / 'run.csv').write_text('program_id,turn_idx,input,output,end_time\n')
    with pytest.raises(ValueError, match='incomplete'):
        compare(pair)


def test_rejects_wrong_completed_output_length(pair):
    path = pair / 'real/requests.jsonl'
    row = json.loads(path.read_text())
    row['output_toks'] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='output_toks mismatch'):
        compare(pair)


def test_rejects_modified_workload(pair):
    with (pair / 'trace.jsonl').open('a') as out:
        out.write('\n')
    with pytest.raises(ValueError, match='Workload changed'):
        compare(pair)


def test_rejects_duplicate_completion(pair):
    path = pair / 'real/requests.jsonl'
    path.write_text(path.read_text() + '\n' + path.read_text())
    with pytest.raises(ValueError, match='Duplicate'):
        compare(pair)
