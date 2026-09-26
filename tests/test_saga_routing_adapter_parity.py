"""Compare actual routing adapters, including their load projections.

No GPU execution or vLLM import is required.
"""
from types import SimpleNamespace

import pytest

from bench.core.saga_engine import SagaConfig, SagaGateway
from policies.saga_runtime import SagaPlacement
from serving.core.routing_drivers import SagaRouter
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter


def paired(used, reserved=(0, 0), cached=(True, False)):
    # 100 usable physical blocks; native pool also contains its null block.
    # Each unit below is one block's bytes, with identical capacity on both sides.
    schedulers, stats = [], {}
    for i in range(2):
        leaf = SimpleNamespace(children={}, owners={'p'} if cached[i] else set())
        memory = SimpleNamespace(
            npu_mem=100, weight=0, npu_used=used[i],
            npu_reserved=reserved[i], evictable_size=lambda device: 0,
            npu_prefix_cache=SimpleNamespace(root_node=SimpleNamespace(
                children={'leaf': leaf}, owners=set())))
        schedulers.append(SimpleNamespace(memory=memory, request=[], inflight=[]))
        stats[i] = dict(num_gpu_blocks=101,
                        free_queue_blocks=100-used[i]-reserved[i],
                        policy_observation=dict(
                            cached_blocks_by_program={'p': 1} if cached[i] else {}))
    gateway = SagaGateway(2, SagaConfig(routing_only=True), SagaPlacement(),
                          None, call=None, stats=lambda i: stats[i])
    router = SagaRouter(2)
    gateway.placement.home['p'] = router.placement.home['p'] = 0
    adapter = UnifiedPolicyAdapter.__new__(UnifiedPolicyAdapter)
    return gateway, router, schedulers, adapter


@pytest.mark.parametrize('used,cached,expected', [
    ((60, 10), (True, False), 0),
    ((80, 10), (True, False), 1),
    ((90, 10), (True, False), 1),
    ((60, 10), (False, False), 1),
    ((10, 10), (False, False), 0),
])
def test_equivalent_published_state_routes_identically(used, cached, expected):
    real, sim, schedulers, adapter = paired(used, cached=cached)
    actual_sim = sim.route('p', schedulers, 1.0, adapter._instance_load)
    workers, resident = real._observe_workers(1.0)
    assert [w.load for w in workers] == pytest.approx(
        [w.load for w in sim._routing_workers])
    assert resident == sim._routing_cached
    assert real.route('p', 1.0) == actual_sim == expected


def test_inflight_reservation_counts_toward_routing_load():
    real, sim, schedulers, adapter = paired((79, 10), reserved=(3, 0))
    actual_sim = sim.route('p', schedulers, 1.0, adapter._instance_load)
    workers, _ = real._observe_workers(1.0)
    assert workers[0].load == pytest.approx(0.82)
    assert sim._routing_workers[0].load == pytest.approx(0.82)
    assert real.route('p', 1.0) == 1
    assert actual_sim == 1


def test_publication_transfers_reservation_without_changing_load():
    real, sim, schedulers, adapter = paired((79, 10), reserved=(3, 0))
    before = adapter._instance_load(schedulers[0])
    schedulers[0].memory.npu_used += 3
    schedulers[0].memory.npu_reserved = 0
    assert adapter._instance_load(schedulers[0]) == before


def test_route_trace_records_actual_inputs_without_changing_decision(monkeypatch, tmp_path):
    import json
    monkeypatch.setenv('SAGA_ROUTING_TRACE_DIR', str(tmp_path))
    real, sim, schedulers, adapter = paired((79, 10), reserved=(3, 0))
    assert sim.route('p', schedulers, 1., adapter._instance_load) == 1
    assert real.route('p', 1.) == 1
    records = {}
    for path in tmp_path.glob('*.jsonl'):
        row = json.loads(path.read_text())
        records[row['plane']] = row
    for plane in ('sim', 'real'):
        row = records[plane]
        assert row['previous_home'] == 0
        assert row['cached_workers'] == [0]
        assert row['destination'] == 1
        assert [x['load'] for x in row['workers']] == pytest.approx([.82, .1])
