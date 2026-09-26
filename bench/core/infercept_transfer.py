"""Layer-pipelined physical KV exchange for the InferCept connector.

The caller owns all tensors and reservations, including pinned host scratch.
Raw views have shape [physical_blocks, bytes_per_block] for each layer segment.
Scratch is separate from the allocatable host pool and must be budgeted at
initialization. No source or destination may be recycled until finish() ACKs.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSwapPlan:
    store_gpu: tuple = ()
    store_cpu: tuple = ()
    load_cpu: tuple = ()
    load_gpu: tuple = ()


class LayerSwapWorker:
    def __init__(self, layers, gpu, cpu, scratch):
        import torch
        from vllm.v1.simple_kv_offload.cuda_mem_ops import build_params

        self.layers = {name: tuple(segments) for name, segments in layers.items()}
        segments = [s for names in self.layers.values() for s in names]
        if not segments or len(segments) != len(set(segments)):
            raise ValueError('each physical KV segment must belong to exactly one layer')
        if set(segments) != set(gpu) or set(gpu) != set(cpu) or set(cpu) != set(scratch):
            raise ValueError('GPU, CPU, scratch and layer layouts must match')
        self.gpu, self.cpu, self.scratch = gpu, cpu, scratch
        self.gpu_blocks = min(t.shape[0] for t in gpu.values())
        self.cpu_blocks = min(t.shape[0] for t in cpu.values())
        self.scratch_blocks = min(t.shape[0] for t in scratch.values())
        devices = {t.device for t in gpu.values()}
        if len(devices) != 1 or next(iter(devices)).type != 'cuda':
            raise ValueError('all KV tensors must be on one CUDA device')
        self.device = next(iter(devices))
        for name in segments:
            g, c, s = gpu[name], cpu[name], scratch[name]
            if (g.ndim != 2 or c.ndim != 2 or s.ndim != 2
                    or g.dtype != torch.int8 or c.dtype != g.dtype or s.dtype != g.dtype
                    or g.shape[1:] != c.shape[1:] or g.shape[1:] != s.shape[1:]
                    or not all(t.is_contiguous() for t in (g, c, s))
                    or c.device.type != 'cpu' or s.device.type != 'cpu'
                    or not c.is_pinned() or not s.is_pinned()):
                raise ValueError('transfer requires contiguous raw GPU views and matching pinned host buffers')
        # Never let host scratch alias live source data. Full-pool exchanges
        # depend on preserving both directions until DMA has finished.
        spans = sorted((t.data_ptr(), t.data_ptr() + t.numel())
                       for t in (*cpu.values(), *scratch.values()))
        if any(end > next_start for (_, end), (next_start, _) in zip(spans, spans[1:])):
            raise ValueError('host pool and staging segments must not overlap')
        self.stream = torch.cuda.Stream(device=self.device, priority=1)
        self.params = {}
        for layer, names in self.layers.items():
            take = lambda mapping: {name: mapping[name] for name in names}
            self.params[layer] = (
                build_params(take(gpu), take(scratch), self.stream),
                build_params(take(cpu), take(gpu), self.stream))
        self.events = {}
        self.plan = None
        self.failed = False

    @staticmethod
    def _ids(values, capacity, label, *, shared_source=False):
        if (not shared_source and len(values) != len(set(values))) or any(
                not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < capacity
                for v in values):
            raise ValueError(f'invalid or duplicate {label} block IDs')

    def start(self, plan):
        import torch
        from vllm.v1.simple_kv_offload.cuda_mem_ops import copy_blocks

        if self.plan is not None or self.failed:
            raise RuntimeError('previous transfer must be acknowledged before starting another')
        if (len(plan.store_gpu) != len(plan.store_cpu)
                or len(plan.load_cpu) != len(plan.load_gpu)
                or len(plan.store_gpu) > self.scratch_blocks):
            raise ValueError('transfer lengths differ or exceed reserved host staging')
        # Multiple paused requests can own the same cached prefix block.
        # Repeated reads are safe; destinations must remain unique.
        self._ids(plan.store_gpu, self.gpu_blocks, 'GPU store', shared_source=True)
        self._ids(plan.load_gpu, self.gpu_blocks, 'GPU load')
        self._ids(plan.store_cpu, self.cpu_blocks, 'CPU store')
        self._ids(plan.load_cpu, self.cpu_blocks, 'CPU load')
        self.plan = plan
        self.events = {}
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            for layer, (store_params, load_params) in self.params.items():
                # Preserve outgoing data before any same-page incoming write.
                # Host source blocks remain untouched until every load ends.
                copy_blocks(list(plan.store_gpu), list(range(len(plan.store_gpu))), store_params)
                copy_blocks(list(plan.load_cpu), list(plan.load_gpu), load_params)
                ready = torch.cuda.Event()
                ready.record(self.stream)
                self.events[layer] = ready
        except Exception:
            # The engine must fail closed: some DMA may already be queued.
            self.failed = True
            raise

    def wait_for_layer(self, layer):
        import torch

        if self.failed:
            raise RuntimeError('layer transfer failed; storage remains reserved')
        if self.plan is not None:
            torch.cuda.current_stream(self.device).wait_event(self.events[layer])

    def finish(self, *, wait=False):
        """Return true only when DMA and host publication have completed.

        wait=False lets an engine poll; wait=True drains at shutdown/cancellation.
        Cancellation cannot release these buffers until this method returns true.
        """
        import torch

        if self.failed:
            raise RuntimeError('failed transfer cannot acknowledge reusable storage')
        if self.plan is None:
            return True
        if wait:
            self.stream.synchronize()
        elif not all(event.query() for event in self.events.values()):
            return False
        plan = self.plan
        # CPU -> CPU publication is deliberately after all H2D reads: store_cpu
        # is allowed to name the same slots as load_cpu during an exchange.
        if plan.store_cpu:
            indices = torch.tensor(plan.store_cpu, dtype=torch.long)
            for name, target in self.cpu.items():
                target.index_copy_(0, indices, self.scratch[name][:len(indices)])
        self.plan = None
        self.events.clear()
        return True
