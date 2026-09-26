"""Transfer ownership protocol shared by simulator and engine adapters.

This coordinates existing allocator reservations; it does not allocate tensors
or invent transfer latencies. A backend must reserve real destination storage,
launch copies and report completion across every participating rank. The source
remains owned until all copies finish. Policy code chooses what to transfer.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Hashable, Protocol, Tuple


class TransferState(str, Enum):
    COPYING = 'copying'
    COMMITTED = 'committed'
    CANCELLED = 'cancelled'


@dataclass(frozen=True)
class KVLocation:
    engine: str
    tier: str
    handle: Hashable
    size_bytes: int


class TransferBackend(Protocol):
    """Methods are invoked at a scheduler boundary with no source writers.

    reserve must be all-or-nothing. start either returns a ticket that owns all
    in-flight work, or raises only after ensuring no copy still references the
    reservation. ready(ticket) becomes true only after *all* ranks finish.
    release is an idempotent allocator operation. These are hard requirements,
    including for failure paths; a mocked ticket is not a physical transfer.
    """

    def reserve(self, request_id: str, engine: str, tier: str,
                size_bytes: int) -> KVLocation: ...
    def start(self, source: KVLocation, destination: KVLocation) -> Hashable: ...
    def ready(self, ticket: Hashable) -> bool: ...
    def release(self, location: KVLocation) -> None: ...


@dataclass
class Transfer:
    request_id: str
    source: KVLocation
    destination: KVLocation
    ticket: Hashable
    state: TransferState = TransferState.COPYING
    cancel_requested: bool = False


class KVTransferRuntime:
    """One authoritative location per request; no runnable half-copied KV.

    A failed capacity reservation leaves source ownership unchanged. Cancellation
    waits for completion before freeing the destination, so DMA never targets a
    reallocated block. To free a cancelled request's source, the scheduler must
    first drain its transfer and then call remove().
    """

    def __init__(self, backend: TransferBackend):
        self.backend = backend
        self.locations: Dict[str, KVLocation] = {}
        self.transfers: Dict[str, Transfer] = {}

    def register(self, request_id: str, location: KVLocation):
        if request_id in self.locations:
            raise ValueError(f'KV already registered for {request_id}')
        if location.tier not in ('gpu', 'cpu') or location.size_bytes <= 0:
            raise ValueError('KV location must have a valid tier and positive size')
        self.locations[request_id] = location

    def begin(self, request_id: str, engine: str, tier: str) -> Transfer:
        if request_id in self.transfers:
            raise RuntimeError('request already has an in-flight transfer')
        source = self.locations[request_id]
        if tier not in ('gpu', 'cpu'):
            raise ValueError('unsupported KV tier')
        if (source.engine, source.tier) == (engine, tier):
            raise ValueError('source and destination are identical')
        destination = self.backend.reserve(request_id, engine, tier, source.size_bytes)
        if (destination.engine != engine or destination.tier != tier
                or destination.size_bytes != source.size_bytes):
            self.backend.release(destination)
            raise ValueError('backend returned a mismatched KV reservation')
        try:
            ticket = self.backend.start(source, destination)
        except Exception:
            self.backend.release(destination)
            raise
        transfer = Transfer(request_id, source, destination, ticket)
        self.transfers[request_id] = transfer
        return transfer

    def cancel(self, request_id: str):
        self.transfers[request_id].cancel_requested = True

    def poll(self) -> Tuple[Transfer, ...]:
        completed = []
        for rid, transfer in tuple(self.transfers.items()):
            if not self.backend.ready(transfer.ticket):
                continue
            if transfer.cancel_requested:
                self.backend.release(transfer.destination)
                transfer.state = TransferState.CANCELLED
            else:
                # Source is retained until readiness covers every rank. A
                # backend release failure leaves the transfer retryable.
                self.backend.release(transfer.source)
                self.locations[rid] = transfer.destination
                transfer.state = TransferState.COMMITTED
            del self.transfers[rid]
            completed.append(transfer)
        return tuple(completed)

    def runnable(self, request_id: str, engine: str) -> bool:
        if request_id in self.transfers:
            return False
        location = self.locations[request_id]
        return location.tier == 'gpu' and location.engine == engine

    def remove(self, request_id: str):
        if request_id in self.transfers:
            raise RuntimeError('drain or cancel the transfer before freeing KV')
        self.backend.release(self.locations[request_id])
        del self.locations[request_id]
