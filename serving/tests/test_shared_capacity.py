import json

import pytest
from experiments.paper import run


def test_shared_capacity_does_not_require_rate_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'verify_spec', lambda spec: None)
    spec = dict(policy='autellix-proxy', hardware='B200', model='model', tp=1,
                instances=1, dtype='bfloat16', gpu_memory_utilization=0.9,
                max_model_len=64, max_num_seqs=2, max_num_batched_tokens=32,
                block_size=16, cluster={'nodes': [{'instances': [{}]}]}, dataset='trace.jsonl')
    (tmp_path/'trace.jsonl').write_text('{}\n')
    probe = tmp_path/'probe'; (probe/'real').mkdir(parents=True)
    (probe/'run-spec.json').write_text(json.dumps(spec))
    meta = dict(model='model', dataset_hash='different-rate-is-allowed', engine_kwargs=dict(
        tensor_parallel_size=1, dtype='bfloat16', effective_instances=[dict(
            instance=0, kv_cache_tokens=160, kv_block_size=16, max_model_len=64,
            max_num_seqs=2, max_num_batched_tokens=32, scheduler_reserve_full_isl=True)]))
    capacity = probe/'real/engine-startup.json'; capacity.write_text(json.dumps(meta))
    for rate in ('0.02', '0.1'):
        output=tmp_path/rate; output.mkdir()
        result, measured=run.shared_capacity_inputs(dict(spec,rate=rate),output,capacity)
        assert result['rate'] == rate
        assert measured['engine_kwargs']['effective_instances'][0]['kv_cache_tokens'] == 160
        assert not (output/'real').exists()
        assert not (output/'run-spec.json').exists()
    with pytest.raises(ValueError, match='hardware'):
        run.shared_capacity_inputs(dict(spec,hardware='RTXPRO6000'),output,capacity)
    with pytest.raises(ValueError, match='max_model_len'):
        run.shared_capacity_inputs(dict(spec,max_model_len=128),output,capacity)
