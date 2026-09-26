import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.core.policy_scheduler import PolicyObservations
from bench.core.policy_driver import PolicyDriver, TupleConfig


class BaseScheduler:
    def schedule(self):
        return SimpleNamespace(num_scheduled_tokens={'running': 32, 'finished': 16})

    def kv_protection_stats(self):
        return {'protected': 123}


class Observed(PolicyObservations, BaseScheduler):
    pass


def test_snapshot_counts_blocks_not_stale_request_handles(monkeypatch):
    monkeypatch.setattr('bench.core.policy_scheduler.time.time', lambda: 100)
    scheduler = Observed()
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(_protected={1: 101, 2: 99, 3: 1e18}),
        _protected_requests={'a': [1, 1, 2, 4], 'stale': [4], 'held': [3]})
    scheduler.running = [SimpleNamespace(num_computed_tokens=64)]
    scheduler.requests = {'running': object()}
    scheduler.schedule()
    stats = scheduler.kv_protection_stats()
    assert stats['protected'] == 123
    assert stats['policy_observation'] == {
        'schema_version': 1, 'protected_blocks_by_tag': {'a': 1, 'held': 1},
        'scheduled_query_tokens': 32, 'running_context_tokens': 64}


def test_continuum_priority_refresh_keeps_release_handle(monkeypatch):
    monkeypatch.setenv('VLLM_KV_RELEASE_AT_ARRIVAL', '1')

    async def check():
        driver = PolicyDriver(TupleConfig(retention='continuum',
            scheduling='continuum', engine_observations=True), [],
            asyncio.get_running_loop())
        try:
            driver.programs.on_turn_release('p', 0, 0)
            driver.programs.note_retention('p', 'protect', 10, 'p:0')
            driver.kv._where['p:0'] = 0
            control = Mock()
            control.stats.return_value = {'policy_observation': {
                'schema_version': 1, 'protected_blocks_by_tag': {},
                'scheduled_query_tokens': 32, 'running_context_tokens': 64}}
            control.release.return_value = 0
            driver.kv._controls = [control]
            driver._kv_utilization = lambda instance: 0.5
            priority = driver._arrival_sync('p', 1, 1)
            assert priority >= driver.scheduling_exec.policy._CLASS - 1000
            control.release.assert_called_once_with('p:0')
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(check())


def test_requested_observations_fail_instead_of_assuming_idle():
    async def check():
        driver = PolicyDriver(TupleConfig(retention='continuum',
            scheduling='continuum', engine_observations=True), [],
            asyncio.get_running_loop())
        try:
            control = Mock()
            control.stats.return_value = {}
            driver.kv._controls = [control]
            with pytest.raises(RuntimeError, match='unavailable'):
                driver._observed_load()
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(check())


def test_infercept_uses_observed_tool_duration_and_load(tmp_path):
    import json
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'a': 1, 'c': 1, 'S': 16}))

    async def check():
        driver = PolicyDriver(TupleConfig(retention='min-waste', scheduling='fcfs',
            min_waste_profile=str(profile), engine_observations=True), [],
            asyncio.get_running_loop())
        try:
            control = Mock()
            control.stats.return_value = {'policy_observation': {
                'schema_version': 1, 'protected_blocks_by_tag': {},
                'scheduled_query_tokens': 5, 'running_context_tokens': 100}}
            driver.kv._controls = [control]
            table = driver.programs
            table.on_turn_release('p', 0, 0)
            table.on_turn_complete('p', 0, now=1, tool_name='tool')
            table.on_turn_release('p', 1, 1.125)
            pcb = table.on_turn_complete('p', 1, now=2, tool_name='tool',
                                         context_tokens=32)
            action, deadline, info = driver.retention_exec.policy.on_turn_complete(
                pcb, 'p:1', 2)
            assert info['gap_pred_s'] == 0.125
            assert info['inflight_tokens'] == 5
            assert info['running_ctx_tokens'] == 100
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(check())
