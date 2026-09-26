import pytest

from policies.kv_transfer_runtime import KVLocation, KVTransferRuntime, TransferState


class Allocator:
    """Test backend with explicit rank acknowledgments and finite capacity."""
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.allocated = {}
        self.next_id = 0
        self.acks = set()

    def reserve(self, request_id, engine, tier, size_bytes):
        used = sum(x.size_bytes for x in self.allocated.values()
                   if (x.engine, x.tier) == (engine, tier))
        if used + size_bytes > self.capacity:
            raise MemoryError('destination full')
        self.next_id += 1
        location = KVLocation(engine, tier, self.next_id, size_bytes)
        self.allocated[location.handle] = location
        return location

    def start(self, source, destination):
        return destination.handle

    def ready(self, ticket):
        return (ticket, 0) in self.acks and (ticket, 1) in self.acks

    def release(self, location):
        self.allocated.pop(location.handle, None)

    def ack(self, ticket):
        self.acks.update(((ticket, 0), (ticket, 1)))


def setup():
    backend = Allocator()
    runtime = KVTransferRuntime(backend)
    source = backend.reserve('r', 'a', 'gpu', 32)
    runtime.register('r', source)
    return backend, runtime, source


def test_swap_waits_for_all_ranks_and_retains_source_until_completion():
    backend, runtime, source = setup()
    move = runtime.begin('r', 'a', 'cpu')
    backend.acks.add((move.ticket, 0))
    assert runtime.poll() == ()
    assert source.handle in backend.allocated
    assert not runtime.runnable('r', 'a')
    backend.ack(move.ticket)
    assert runtime.poll() == (move,)
    assert move.state is TransferState.COMMITTED
    assert source.handle not in backend.allocated
    assert not runtime.runnable('r', 'a')
    assert runtime.poll() == ()


def test_cpu_to_other_engine_migration_becomes_runnable_only_at_destination():
    backend, runtime, source = setup()
    store = runtime.begin('r', 'a', 'cpu')
    backend.ack(store.ticket)
    runtime.poll()
    load = runtime.begin('r', 'b', 'gpu')
    assert not runtime.runnable('r', 'b')
    backend.ack(load.ticket)
    runtime.poll()
    assert runtime.runnable('r', 'b')
    assert not runtime.runnable('r', 'a')
    assert len(backend.allocated) == 1


def test_failed_allocation_keeps_source_and_creates_no_transfer():
    backend, runtime, source = setup()
    backend.reserve('other', 'b', 'gpu', 32)
    with pytest.raises(MemoryError):
        runtime.begin('r', 'b', 'gpu')
    assert runtime.locations['r'] == source
    assert not runtime.transfers
    assert runtime.runnable('r', 'a')


def test_cancel_waits_for_dma_before_releasing_destination():
    backend, runtime, source = setup()
    move = runtime.begin('r', 'b', 'gpu')
    runtime.cancel('r')
    assert runtime.poll() == ()
    assert len(backend.allocated) == 2
    with pytest.raises(RuntimeError):
        runtime.remove('r')
    backend.ack(move.ticket)
    runtime.poll()
    assert move.state is TransferState.CANCELLED
    assert runtime.runnable('r', 'a')
    assert len(backend.allocated) == 1
    runtime.remove('r')
    assert not backend.allocated


def test_failed_launch_returns_destination_reservation():
    backend, runtime, source = setup()
    def fail(source, destination):
        raise RuntimeError('launch failed before DMA')
    backend.start = fail
    with pytest.raises(RuntimeError):
        runtime.begin('r', 'a', 'cpu')
    assert list(backend.allocated.values()) == [source]
    assert runtime.runnable('r', 'a')


def test_second_transfer_cannot_overwrite_an_inflight_copy():
    backend, runtime, source = setup()
    runtime.begin('r', 'a', 'cpu')
    with pytest.raises(RuntimeError):
        runtime.begin('r', 'b', 'gpu')
    assert len(backend.allocated) == 2
