import asyncio
from types import SimpleNamespace

import pytest

from bench.core.policy_migration import migrate_prefix


class Endpoint:
    def __init__(self, side, events, fail=None):
        self.side, self.events, self.fail = side, events, fail
        self.held = set()
        self.engine_core = SimpleNamespace(call_utility_async=self.call)

    async def call(self, name, op, args):
        assert name == 'policy_migration'
        assert isinstance(args, list)
        self.events.append((self.side, op))
        if op.startswith('begin_'):
            handle = args[-1]
            self.held.add(handle)
            if self.fail == op:
                # Reservation committed, but the coordinator lost the reply.
                raise asyncio.CancelledError()
            return {'handle': handle, 'tokens': 32, 'fingerprint': 'same', 'chunk_blocks': 1}
        if op.startswith('release_'):
            self.held.discard(args[0])
        if op == 'block_size':
            return 16
        if op == 'read':
            return [{'rank': 0, 'data': {'kv': b'bytes'}}]
        if self.fail == op:
            raise RuntimeError('transfer failed')


def test_migration_ack_precedes_source_release():
    async def run():
        events = []
        source, dest = Endpoint('source', events), Endpoint('destination', events)
        receipt = await migrate_prefix(source, dest, list(range(33)), 'consumer')
        assert events.index(('destination', 'publish')) < events.index(('source', 'release_export'))
        assert not source.held
        assert dest.held == {receipt['handle']}
        assert events.count(('destination', 'write')) == 2
    asyncio.run(run())


@pytest.mark.parametrize('side', ['source', 'destination'])
def test_cancelled_reservation_reply_is_reclaimed_by_known_handle(side):
    async def run():
        events = []
        source = Endpoint('source', events, 'begin_export' if side == 'source' else None)
        dest = Endpoint('destination', events, 'begin_import' if side == 'destination' else None)
        with pytest.raises(asyncio.CancelledError):
            await migrate_prefix(source, dest, list(range(33)), 'consumer')
        assert not source.held
        assert not dest.held
    asyncio.run(run())


def test_failed_destination_cleanup_does_not_release_source():
    async def run():
        events = []
        source, dest = Endpoint('source', events), Endpoint('destination', events, 'write')
        original = dest.call

        async def call(name, op, *args):
            if op == 'release_import':
                raise RuntimeError('unacknowledged DMA')
            return await original(name, op, *args)

        dest.engine_core.call_utility_async = call
        with pytest.raises(RuntimeError, match='unacknowledged'):
            await migrate_prefix(source, dest, list(range(33)), 'consumer')
        assert source.held and dest.held
        assert ('source', 'release_export') not in events
    asyncio.run(run())
