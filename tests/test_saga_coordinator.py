import asyncio
from types import SimpleNamespace

import pytest

from bench.core import saga_coordinator as module


def test_steal_reserves_pending_session_until_real_transfer_ack(monkeypatch):
    async def run():
        started, finish = asyncio.Event(), asyncio.Event()
        engines = [object(), object()]

        async def transfer(source, destination, tokens, tag):
            assert (source, destination) == tuple(engines)
            assert tokens == [1] * 33 and tag == 'p:1'
            started.set()
            await finish.wait()
            return {'handle': 'receipt'}

        monkeypatch.setattr(module, 'migrate_prefix', transfer)
        c = module.SagaCoordinator(engines)
        c.enqueue('p', 'p:1', [1] * 33, 0, 0, {'ttl_deadline': 5, 'tenant': 't'})
        task = asyncio.create_task(c.steal(1, [0.9, 0], [None, 0], 1))
        await started.wait()
        assert c.placement.home['p'] == 0
        with pytest.raises(RuntimeError, match='acknowledged'):
            c.take('p', 0)
        with pytest.raises(RuntimeError, match='unacknowledged'):
            await c.cancel('p')
        finish.set()
        await task
        assert c.placement.home['p'] == 1
        with pytest.raises(ValueError, match='stale'):
            c.take('p', 0)
        record = c.take('p', 1)
        assert record.receipt == {'handle': 'receipt'}
        assert record.metadata == {'ttl_deadline': 5, 'tenant': 't'}
        assert not c.pending
    asyncio.run(run())


def test_migrated_queued_cancellation_releases_destination_receipt(monkeypatch):
    async def run():
        calls = []

        async def utility(*args):
            calls.append(args)

        async def transfer(*args):
            return {'handle': 'receipt'}

        monkeypatch.setattr(module, 'migrate_prefix', transfer)
        c = module.SagaCoordinator([object(), SimpleNamespace(
            engine_core=SimpleNamespace(call_utility_async=utility))])
        c.enqueue('p', 'p:1', [1] * 33, 0, 0)
        await c.steal(1, [0.9, 0], [None, 0], 1)
        await c.cancel('p')
        assert calls == [('policy_migration', 'release_import', ['receipt'])]
        assert not c.pending
    asyncio.run(run())


def test_failed_transfer_never_dispatches_or_changes_affinity(monkeypatch):
    async def run():
        async def transfer(*args):
            raise RuntimeError('missing destination acknowledgement')

        monkeypatch.setattr(module, 'migrate_prefix', transfer)
        c = module.SagaCoordinator([object(), object()])
        c.enqueue('p', 'p:1', [1] * 33, 0, 0)
        with pytest.raises(RuntimeError, match='acknowledgement'):
            await c.steal(1, [0.9, 0], [None, 0], 1)
        assert c.placement.home['p'] == 0
        assert c.stats['failed_migrations'] == 1
        with pytest.raises(RuntimeError, match='acknowledged'):
            c.take('p', 0)
    asyncio.run(run())
