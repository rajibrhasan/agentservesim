from types import SimpleNamespace

import pytest

from bench.core.policy_block_backend import BlockTransferBackend
from policies.kv_transfer_runtime import KVTransferRuntime


class Pool:
    """Bounded reference-counted pool; no fake successful over-allocation."""

    def __init__(self, n):
        self.blocks = [SimpleNamespace(block_id=i, ref_cnt=0, is_null=False)
                       for i in range(n)]
        self.allocations = 0

    def get_num_free_blocks(self):
        return sum(b.ref_cnt == 0 for b in self.blocks)

    def get_new_blocks(self, n):
        assert n <= self.get_num_free_blocks()
        self.allocations += 1
        result = [b for b in self.blocks if b.ref_cnt == 0][:n]
        self.touch(result)
        return result

    def touch(self, blocks):
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        for block in blocks:
            assert block.ref_cnt > 0
            block.ref_cnt -= 1


def setup(cpu_blocks=4):
    gpu, cpu = Pool(4), Pool(cpu_blocks)
    submitted = []
    backend = BlockTransferBackend('e', gpu, cpu, 128, 2,
                                   lambda *args: submitted.append(args))
    return backend, gpu, cpu, submitted


def test_physical_swap_round_trip_includes_tail_and_all_rank_ack():
    backend, gpu, cpu, copies = setup()
    original = gpu.get_new_blocks(2)  # Full block + unfinished block.
    source = backend.capture_gpu([b.block_id for b in original])
    runtime = KVTransferRuntime(backend)
    runtime.register('r', source)
    transfer = runtime.begin('r', 'e', 'cpu')
    gpu.free_blocks(original)  # Request detaches; transfer retains both refs.
    assert source.size_bytes == 256
    assert gpu.get_num_free_blocks() == 2
    assert copies == [(transfer.ticket, 'gpu', (0, 1), (0, 1))]
    backend.acknowledge(transfer.ticket, 0)
    backend.acknowledge(transfer.ticket, 0)  # Duplicate cannot stand for rank 1.
    assert runtime.poll() == ()
    backend.acknowledge(transfer.ticket, 1)
    runtime.poll()
    assert gpu.get_num_free_blocks() == 4
    assert cpu.get_num_free_blocks() == 2
    back = runtime.begin('r', 'e', 'gpu')
    backend.acknowledge(back.ticket, 1)
    backend.acknowledge(back.ticket, 0)
    runtime.poll()
    assert runtime.runnable('r', 'e')
    assert cpu.get_num_free_blocks() == 4
    runtime.remove('r')
    assert gpu.get_num_free_blocks() == 4


def test_failed_reservation_does_not_call_allocator_or_release_source():
    backend, gpu, cpu, copies = setup(cpu_blocks=1)
    original = gpu.get_new_blocks(2)
    runtime = KVTransferRuntime(backend)
    source = backend.capture_gpu([b.block_id for b in original])
    runtime.register('r', source)
    with pytest.raises(MemoryError):
        runtime.begin('r', 'e', 'cpu')
    assert cpu.allocations == 0
    assert runtime.locations['r'] == source
    assert all(b.ref_cnt == 2 for b in original)
    assert copies == []


def test_shared_source_refs_and_cancel_wait_for_dma():
    backend, gpu, cpu, copies = setup()
    original = gpu.get_new_blocks(1)
    first = backend.capture_gpu([0])
    second = backend.capture_gpu([0])
    gpu.free_blocks(original)
    runtime = KVTransferRuntime(backend)
    runtime.register('a', first)
    runtime.register('b', second)
    transfer = runtime.begin('a', 'e', 'cpu')
    with pytest.raises(RuntimeError, match='DMA'):
        backend.release(first)
    runtime.remove('b')
    assert gpu.blocks[0].ref_cnt == 1
    runtime.cancel('a')
    assert runtime.poll() == ()
    for rank in (0, 1):
        backend.acknowledge(transfer.ticket, rank)
    runtime.poll()
    assert cpu.get_num_free_blocks() == 4
    assert runtime.locations['a'] == first
    runtime.remove('a')
    assert gpu.get_num_free_blocks() == 4


def test_launch_failure_frees_only_destination():
    backend, gpu, cpu, _ = setup()
    original = gpu.get_new_blocks(1)
    source = backend.capture_gpu([0])
    runtime = KVTransferRuntime(backend)
    runtime.register('r', source)

    def fail(*args):
        raise RuntimeError('not enqueued')

    backend.submit = fail
    with pytest.raises(RuntimeError, match='not enqueued'):
        runtime.begin('r', 'e', 'cpu')
    assert cpu.get_num_free_blocks() == 4
    assert gpu.blocks[0].ref_cnt == 2
    assert not backend._copies


def test_cannot_capture_free_blocks_or_silently_migrate():
    backend, gpu, cpu, _ = setup()
    with pytest.raises(ValueError, match='owned'):
        backend.capture_gpu([0])
    with pytest.raises(ValueError, match='migration transport'):
        backend.reserve('r', 'another', 'gpu', 128)
    assert gpu.allocations == cpu.allocations == 0
