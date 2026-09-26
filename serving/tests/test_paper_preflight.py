import json
from types import SimpleNamespace as NS

import pytest

from experiments.paper import preflight
from experiments.paper.gate import REAL_ENV, SIM_ENV, prepare, validate_environment
from experiments.paper.matrix import matched_cluster, sha256


def test_frozen_profiles_reject_changed_latency(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, 'ROOT', tmp_path)
    for name in ('dense', 'per_sequence', 'attention'):
        path = tmp_path / f'profile/tp1/{name}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('tokens,time_us\n1,2\n')
    (tmp_path / 'profile/meta.yaml').write_text('version: 1\n')
    (tmp_path / 'configs/model').mkdir(parents=True)
    (tmp_path / 'configs/model/test.json').write_text('{}')
    dataset = tmp_path / 'trace.jsonl'
    dataset.write_text('{}\n')
    spec = dict(model='test', tp=1, latency_profile_source='profile', profile=None,
                dataset='trace.jsonl', dataset_sha256=sha256(dataset))
    spec['artifact_hashes'] = preflight.verify_spec(spec)
    (tmp_path / 'profile/tp1/attention.csv').write_text('tokens,time_us\n1,3\n')
    with pytest.raises(ValueError, match='artifacts differ'):
        preflight.verify_spec(spec)


def test_streaming_context_limit_includes_previous_generated_tokens(tmp_path):
    path = tmp_path / 'trace.jsonl'
    row = dict(session_id='p', arrival_time_ns=0, input_mode='streaming-deltas',
               sub_requests=[dict(input_toks=2, input_tok_ids=[1, 2], output_toks=3)] * 2)
    path.write_text(json.dumps(row) + '\n')
    assert preflight.validate_workload(path, 10)['max_context_tokens'] == 10
    with pytest.raises(ValueError, match='exceeds'):
        preflight.validate_workload(path, 9)


def test_two_equal_instances_use_their_own_measured_capacity():
    spec = dict(instances=2, cluster={'nodes': [{'instances': [{}, {}]}]},
                max_model_len=64, max_num_seqs=2, max_num_batched_tokens=32, block_size=16)
    rows = [dict(instance=i, kv_cache_tokens=64 - 16*i, kv_block_size=16,
                 scheduler_reserve_full_isl=True, max_model_len=64, max_num_seqs=2,
                 max_num_batched_tokens=32) for i in range(2)]
    meta = {'engine_kwargs': {'effective_instances': rows}}
    assert [r['kv_pool_tokens'] for r in matched_cluster(spec, meta)['nodes'][0]['instances']] == [64, 48]
    rows[1]['kv_cache_tokens'] = 47
    with pytest.raises(ValueError, match='whole number of blocks'):
        matched_cluster(spec, meta)


def test_gate_engine_bridge_uses_named_policy(tmp_path):
    from policies.gate import EvolvedScheduling
    contract = prepare(tmp_path)
    namespace = {}
    exec((tmp_path / 'harness/evolved_scheduling.py').read_text(), namespace)
    assert namespace['EvolvedScheduling'] is EvolvedScheduling
    assert contract['flags'] == ['--retention', 'gate', '--scheduling', 'gate']
    validate_environment(REAL_ENV)
    validate_environment(SIM_ENV, simulator=True)
    with pytest.raises(ValueError, match='GATE_SITE'):
        validate_environment(dict(REAL_ENV, GATE_SITE='gateway'))
