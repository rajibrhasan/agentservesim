"""Measure the host link that InferCept's swap budget is sized from.

The budget converts a forward-pass duration into blocks that can move behind
it, so the rate must come from the node that runs, not from a device table.
"""


def measure_host_bandwidth_bytes_s(device=0, nbytes=64 << 20, iters=10, barrier=None):
    """Slowest of pinned device-to-host and host-to-device copy rates."""
    import time

    import torch

    gpu = torch.empty(nbytes, dtype=torch.uint8, device=f'cuda:{device}')
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    rates = []
    for source, target in ((gpu, host), (host, gpu)):
        target.copy_(source)
        torch.cuda.synchronize(device)
        if barrier is not None:
            barrier.wait(timeout=60)
        start = time.perf_counter()
        for _ in range(iters):
            target.copy_(source, non_blocking=True)
        torch.cuda.synchronize(device)
        rates.append(nbytes * iters / (time.perf_counter() - start))
    return min(rates)


def measure_all_ranks(nbytes=64 << 20, iters=10):
    """Measure both directions while all allocated GPUs compete for host bandwidth."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError('host bandwidth profiling needs an allocated GPU')
    barrier = Barrier(count)
    def measure(rank):
        with torch.cuda.device(rank):
            return measure_host_bandwidth_bytes_s(rank, nbytes, iters, barrier)
    with ThreadPoolExecutor(max_workers=count) as pool:
        rates = list(pool.map(measure, range(count)))
    return {'bytes_per_copy': nbytes, 'iterations': iters,
            'per_rank_bytes_s': rates, 'bandwidth_bytes_s': min(rates),
            'devices': [torch.cuda.get_device_name(i) for i in range(count)]}
