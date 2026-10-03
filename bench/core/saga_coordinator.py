
from dataclasses import dataclass, field

from policies.saga_runtime import SagaPlacement, WorkerObservation
from .policy_migration import migrate_prefix


@dataclass
class PendingSession:
    session: str
    consumer_tag: str
    tokens: tuple
    ready_at: float
    worker: int
    metadata: dict = field(default_factory=dict)
    receipt: object = None
    transferring: bool = False


class SagaCoordinator:
    def __init__(self, engines, placement=None):
        self.engines = tuple(engines)
        if not self.engines:
            raise ValueError('SAGA coordinator needs at least one engine')
        self.placement = placement or SagaPlacement()
        self.pending = {}
        self.stats = {'migrations': 0, 'cache_miss_moves': 0, 'failed_migrations': 0}

    def enqueue(self, session, consumer_tag, tokens, ready_at, worker, metadata=None):
        if session in self.pending:
            raise ValueError('session already has a queued successor')
        if not 0 <= worker < len(self.engines):
            raise ValueError('unknown destination worker')
        record = PendingSession(session, consumer_tag, tuple(tokens), ready_at,
                                worker, dict(metadata or {}))
        self.pending[session] = record
        self.placement.home.setdefault(session, worker)
        return record

    def observations(self, loads, empty_since):
        if len(loads) != len(self.engines) or len(empty_since) != len(self.engines):
            raise ValueError('one current observation is required per worker')
        return tuple(WorkerObservation(i, load, tuple(
            (r.session, r.ready_at) for r in self.pending.values()
            if r.worker == i and not r.transferring), empty_since[i])
            for i, load in enumerate(loads))

    async def steal(self, destination, loads, empty_since, now):
        proposal = self.placement.propose_steal(
            destination, self.observations(loads, empty_since), now)
        if proposal is None:
            return None
        record = self.pending[proposal.session]
        if record.receipt is not None:
            # A queued imported prefix is already reserved. Do not leak its
            # destination ownership by moving it again before consumption.
            self.placement.complete_steal(proposal, published=False)
            return None
        record.transferring = True
        try:
            receipt = await migrate_prefix(self.engines[proposal.source],
                                           self.engines[proposal.destination],
                                           list(record.tokens), record.consumer_tag)
        except BaseException:
            # Failed/uncertain DMA remains owned by the migration protocol.
            # Keep this record frozen: retrying or dispatching it could reuse
            # buffers without an acknowledgement. The caller must surface the
            # failure, not silently fall back to recomputation.
            self.stats['failed_migrations'] += 1
            raise
        record.receipt = receipt
        record.worker = proposal.destination
        record.transferring = False
        self.placement.complete_steal(proposal, published=True)
        self.stats['migrations' if receipt is not None else 'cache_miss_moves'] += 1
        return record

    def take(self, session, worker):
        """Hand ownership to the caller, which must submit or release receipt."""
        record = self.pending[session]
        if record.transferring:
            raise RuntimeError('session migration has not been acknowledged')
        if record.worker != worker:
            raise ValueError('stale worker placement')
        del self.pending[session]
        return record

    async def cancel(self, session):
        record = self.pending[session]
        if record.transferring:
            raise RuntimeError('cannot release an unacknowledged migration')
        if record.receipt is not None:
            # Await the engine acknowledgement before removing local ownership.
            await self.engines[record.worker].engine_core.call_utility_async(
                'policy_migration', 'release_import', [record.receipt['handle']])
        del self.pending[session]
