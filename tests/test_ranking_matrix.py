from types import SimpleNamespace

from experiments.paper.ranking import CELLS, POLICIES, FLAGS, spec_for, CPU_BYTES_PER_RANK
from policies.search_seed import EvolvedScheduling
from policies.generic import ProgramFCFSScheduling
from bench.core.policy_driver import engine_flags


def test_infercept_no_swap_has_no_host_kv_capacity():
    for cell in CELLS:
        if cell in ('b200_70b_swebench_j0.06', 'b200_phi_bfcl_j0.8', 'b200_phi_bfcl_j1.2'):
            continue
        no_swap = spec_for(cell, 'infercept-no-swap')
        swap = spec_for(cell, 'infercept-swap')
        assert no_swap['cpu_bytes_per_rank'] == 0
        assert no_swap['cluster']['nodes'][0]['cpu_mem']['mem_size'] == 0
        for key in ('dataset_sha256', 'kv_pool_tokens', 'profile', 'tp'):
            assert no_swap[key] == swap[key]


def test_ranking_policies_share_constraints():
    assert len(POLICIES) == 7
    for cell in CELLS:
        policies = FLAGS if cell in ('b200_70b_swebench_j0.06', 'b200_phi_bfcl_j0.8', 'b200_phi_bfcl_j1.2') else POLICIES
        specs = [spec_for(cell, p) for p in policies]
        for key in ('dataset_sha256', 'kv_pool_tokens', 'max_model_len', 'async_scheduling', 'tp'):
            assert len({s[key] for s in specs}) == 1
        for s in specs:
            if cell in ('b200_phi_bfcl_j0.8', 'b200_phi_bfcl_j1.2'):
                assert s['model'] == 'microsoft/Phi-3.5-MoE-instruct'
                assert s['kv_pool_tokens'] == 657040
                assert s['max_model_len'] == 131072
                assert s['tp'] == s['instances'] == 1
                assert s['workload_audit']['turns'] == 1370
                continue
            assert s['cpu_bytes_per_rank'] == CPU_BYTES_PER_RANK == 8 << 30
            assert s['cluster']['nodes'][0]['cpu_mem']['mem_size'] == 8 * s['tp']
            assert s['input_mode'] == 'full-prompts'


def test_restored_joint_seed_is_not_the_old_program_fcfs_label():
    protected = SimpleNamespace(arrival_ts=1, kv_protected=True, context_tokens=12000)
    new = SimpleNamespace(arrival_ts=0, kv_protected=False, context_tokens=100)
    seed = EvolvedScheduling()
    assert seed.priority(protected, 2) < seed.priority(new, 2)
    plain = ProgramFCFSScheduling()
    assert plain.priority(protected, 2) > plain.priority(new, 2)
    assert engine_flags('search-seed', 'search-seed')['kv_protection']
    assert engine_flags('search-seed', 'search-seed')['scheduling_policy'] == 'priority'


def test_seed_constructs_on_both_request_plane_adapters():
    import asyncio
    from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
    from bench.core.policy_driver import PolicyDriver, TupleConfig
    from experiments.paper.matrix import ROOT
    sim = UnifiedPolicyAdapter(retention_value='search-seed', scheduling_value='search-seed',
        routing_value=None, num_instances=1, block_size=16, harness_root=str(ROOT))
    assert isinstance(sim.scheduling_exec.policy, EvolvedScheduling)
    async def check():
        driver = PolicyDriver(TupleConfig(retention='search-seed', scheduling='search-seed',
            harness_root=str(ROOT)), [], asyncio.get_running_loop())
        try:
            assert isinstance(driver.scheduling_exec.policy, EvolvedScheduling)
            driver._pool_view = lambda instance: (0.1, 10000, 16)
            driver.programs.on_turn_release('seed-test', 0, 0, instance=0)
            assert await driver.admit('seed-test', 0, 128, 0, 1.0)
            assert driver.scheduling_exec.admission_admits == 1
        finally:
            driver.retention_exec.finish()
            driver._worker.shutdown(wait=True)
    asyncio.run(check())


def test_real_submission_selects_seven_policies_with_no_swap(tmp_path, monkeypatch):
    import json
    import sys
    from experiments.paper import submit_ranking
    calls = []
    def submit(command, **kwargs):
        calls.append((command, kwargs['env']))
        return SimpleNamespace(stdout=str(100 + len(calls)))
    monkeypatch.setattr(submit_ranking.subprocess, 'run', submit)
    ledger = tmp_path / 'jobs.json'
    monkeypatch.setattr(sys, 'argv', ['submit_ranking', '--cell', 'b200_8b_swebench_j0.02',
        '--leg', 'real', '--infercept-no-swap', '--ledger', str(ledger)])
    submit_ranking.main()
    records = json.loads(ledger.read_text())
    assert {r['policy'] for r in records} == {
        'stock', 'autellix-proxy', 'infercept-no-swap', 'continuum', 'saga', 'seed', 'gate'}
    for command, env in calls:
        assert '--mem=64G' in command and '--gres=gpu:b200:1' in command
        assert env['VLLM_ENV'].endswith('/engine')
        assert env['RANK_CELL'] == 'b200_8b_swebench_j0.02'
