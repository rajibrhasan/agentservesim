from experiments.paper.ablation import BASELINES, CASES, ROOT, RandomRouting, build
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter


def test_cases_and_fixed_routing_resources():
    assert len(CASES) == 19
    assert sum(case.startswith('modules/') for case in CASES) == 7
    for case in CASES:
        spec, dataset, flags = build(case)
        assert dataset.is_file()
        if case.startswith('routing/'):
            instances = spec['cluster']['nodes'][0]['instances']
            b200 = '/b200_' in case
            expected_count = 4 if b200 or 'tp2x4' in case else 2
            assert len(instances) == expected_count
            assert spec['gpus'] == expected_count * (1 if b200 else 2)
            assert instances[0] == instances[1]
            assert instances[0] is not instances[1]
            assert all(i['tp_size'] == (1 if b200 else 2) and i['kv_pool_tokens'] ==
                       (106672 if b200 else 115648) for i in instances)
            if b200:
                assert spec['gpus'] == 4
                assert spec['cpu_bytes_per_rank'] == 0
                assert str(dataset).endswith('swebench50/swebench_jps0.1_n50.jsonl')
                profile = ROOT / spec['latency_profile_source'] / 'tp1'
                assert (profile / 'attention.csv').is_file()
            assert flags[:4] == ['--retention', 'continuum', '--scheduling', 'fcfs']
            assert '--no-saga-stealing' in flags and '--no-saga-prefetch' in flags
        else:
            assert spec['instances'] == 1
            if 'b200_70b_swebench_j0.06' in case:
                assert spec['tp'] == 1
                assert spec['cluster']['nodes'][0]['instances'][0]['kv_pool_tokens'] == 106672
                assert str(dataset).endswith('swebench_jps0.06_n50.jsonl')


def test_continuum_retention_lifecycle_survives_fcfs():
    adapter = UnifiedPolicyAdapter('continuum', 'fcfs', None, 1, 16,
                                   tau_s=2, harness_root=str(ROOT))
    assert adapter._release_at_scheduled
    adapter = UnifiedPolicyAdapter('cache-lru', 'continuum', None, 1, 16,
                                   harness_root=str(ROOT))
    assert adapter.scheduling_value == 'continuum'
    assert not adapter._release_at_scheduled


def test_random_is_reproducible_and_custom_policy_loads():
    a, b = RandomRouting(2), RandomRouting(2)
    choices = [a.route(None, 0)[0] for _ in range(20)]
    assert choices == [b.route(None, 0)[0] for _ in range(20)]
    assert set(choices) == {0, 1}
    adapter = UnifiedPolicyAdapter('continuum', 'fcfs',
        'experiments.paper.ablation:RandomRouting', 2, 16,
        tau_s=2, harness_root=str(ROOT))
    assert adapter.routing_exec is not None
