import json
from types import SimpleNamespace
from experiments.paper import release_shared_simulations as release


def test_one_running_replay_releases_five_rates_once(tmp_path, monkeypatch):
    source=tmp_path/'source/results/final'; source.mkdir(parents=True)
    spec=dict(cell='b200',policy='autellix-proxy',hardware='B200',model='m',tp=1,
              instances=1,dtype='bfloat16',gpu_memory_utilization=.9,block_size=16,
              max_model_len=64,max_num_seqs=2,max_num_batched_tokens=32,
              cluster={'nodes':[{'instances':[{}]}]})
    specs=[dict(spec,rate=r) for r in ['0.02','0.04','0.06','0.08','0.1']]
    (source/'manifest.json').write_text(json.dumps({'pairs':specs}))
    output=tmp_path/'real_run'; (output/'real').mkdir(parents=True)
    (output/'run-spec.json').write_text(json.dumps(spec))
    meta={'engine_kwargs':{'effective_instances':[dict(instance=0,kv_cache_tokens=160,
        kv_block_size=16,max_model_len=64,max_num_seqs=2,max_num_batched_tokens=32,
        scheduler_reserve_full_isl=True)]}}
    (output/'real/engine-startup.json').write_text(json.dumps(meta))
    calls=[]
    def submit(argv, **kw):
        calls.append(argv); return SimpleNamespace(stdout=str(100+len(calls)))
    monkeypatch.setattr(release.subprocess,'run',submit)
    release.release(tmp_path,'b200',output)
    release.release(tmp_path,'b200',output)
    assert len(calls)==5
    assert all(not any(a.startswith('--dependency') for a in argv) for argv in calls)
    ledger=json.loads((tmp_path/'simulation-jobs.json').read_text())
    assert len(ledger)==5
    assert len({row['capacity_source'] for row in ledger.values()})==1
