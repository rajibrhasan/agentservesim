import asyncio

import pytest

from bench.core.policy_driver import PolicyDriver, TupleConfig
from policies.autellix_runtime import AutellixRuntime, QueueConfig


def test_saga_callbacks_read_shared_observations_without_engine_queries():
    from types import SimpleNamespace
    driver = PolicyDriver.__new__(PolicyDriver)
    driver.cfg = SimpleNamespace(engine_policy='saga')
    driver._block_size = 16
    observation = {'schema_version': 1, 'protected_blocks_by_tag': {},
                   'cached_blocks_by_program': {'p': 2}}
    driver.saga = SimpleNamespace(latest_stats={0: {
        'num_gpu_blocks': 101, 'free_queue_blocks': 40,
        'policy_observation': observation}})
    def forbidden(*args):
        raise AssertionError('callback issued a redundant engine query')
    driver.kv = SimpleNamespace(stats=forbidden,
                                _controls=[SimpleNamespace(stats=forbidden)])
    assert driver._engine_observation(0) is observation
    assert driver._pool_view(0) == (0.6, 40, 16)


def test_autellix_routes_current_prompt_and_preserves_long_call_home():
    async def check():
        driver = PolicyDriver(TupleConfig(engine_policy='autellix'),
                              [object(), object()], asyncio.get_running_loop())
        try:
            assert driver._route_sync('long', 0, 0, 2049) == 0
            assert driver._route_sync('other', 0, 0, 2048) == 1
            # Simulate outstanding load on engine 0. A short turn can leave,
            # but it must not overwrite the long-turn affinity record.
            driver._engine_inflight[:] = [3, 0]
            assert driver._route_sync('long', 1, 1, 10) == 1
            assert driver._route_sync('long', 2, 2, 4096) == 0
            with pytest.raises(ValueError, match='prompt length'):
                driver._route_sync('missing', 0, 0)
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(check())


def test_worker_measured_execution_excludes_host_delay():
    r = AutellixRuntime(QueueConfig((1,), (1, 2), 100))
    r.arrive('r', 'p', 0)
    r.start_batch(['r'], 2)
    # Ten seconds of host/transfer delay must not become ten GPU seconds.
    r.finish_batch(12, ['r'], execution_s=0.25)
    assert r.processes['p'].service_s == 0.25
    assert r.processes['p'].wait_s == 2


def test_cancel_removes_active_queue_entry_without_erasing_consumed_work():
    r = AutellixRuntime(QueueConfig((1,), (1, 2), 100))
    r.arrive('r', 'p', 0)
    r.start_batch(['r'], 0)
    r.finish_batch(1, execution_s=0.25)
    r.cancel('r', 2)
    assert not r.calls
    assert not any(r.queues)
    assert r.processes['p'].service_s == 0.25


def test_engine_change_preserves_global_starvation_history():
    first = AutellixRuntime(QueueConfig((1,), (1, 2), 2))
    second = AutellixRuntime(first.config)
    first.arrive('r0', 'p', 0)
    first.start_batch(['r0'], 6)
    first.finish_batch(8, ['r0'], execution_s=2)
    history = first.processes['p']
    second.arrive('r1', 'p', 10, inherited_service_s=history.service_s,
                  inherited_wait_s=history.wait_s)
    assert second.calls['r1'].queue == 1
    second.plan(10, (), lambda *_: True)
    assert second.calls['r1'].queue == 0
    assert second.processes['p'].service_s == 2
    assert second.processes['p'].wait_s == 6
