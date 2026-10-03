


class PolicyWorkerExtension:
    def policy_kv_init(self, cpu_bytes_per_rank, transport='batched'):
        from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

        if cpu_bytes_per_rank <= 0:
            raise ValueError('CPU swap capacity must be positive')
        self._policy_model_events = []
        runner = self.model_runner
        bytes_per_block = (sum(t.size for t in runner.kv_cache_config.kv_cache_tensors)
                           // runner.kv_cache_config.num_blocks)
        if cpu_bytes_per_rank < bytes_per_block:
            raise ValueError('CPU swap budget cannot hold one physical KV block')
        handler = SimpleCPUOffloadWorker(
            self.vllm_config, runner.kv_cache_config, cpu_bytes_per_rank)
        handler.register_kv_caches({str(i): t for i, t in enumerate(runner.kv_caches)})
        # This synchronous adapter calls the CUDA batch-copy operation directly.
        # No background thread may still write when the scheduler releases KV.
        handler._backend.shutdown()
        self._policy_kv = handler
        if transport not in ('batched', 'contiguous'):
            raise ValueError('unknown swap transport')
        self._policy_transport = transport
        self._policy_staging = {}
        return {'rank': self.rank, 'cpu_blocks': handler.num_cpu_blocks,
                'gpu_blocks': runner.kv_cache_config.num_blocks,
                'layout': [(k, tuple(v.shape[1:]), str(v.dtype))
                           for k, v in handler.gpu_kv_caches.items()]}

    def policy_kv_copy(self, gpu_ids, cpu_ids, to_cpu):
        """Move whole blocks between the GPU KV cache and the host pool.

        'batched' issues vLLM's single cuMemcpyBatchAsync over every block and
        layer. 'contiguous' is Autellix's transport: gather all blocks of all
        layers into one contiguous buffer and move it in one copy, then
        scatter. Both report device time so a run can compare them.
        """
        import torch

        h = self._policy_kv
        if len(gpu_ids) != len(cpu_ids) or not gpu_ids:
            raise ValueError('a copy needs equally sized nonempty block lists')
        if (any(not 0 <= i < h.kv_cache_config.num_blocks for i in gpu_ids)
                or any(not 0 <= i < h.num_cpu_blocks for i in cpu_ids)):
            raise ValueError('block ID outside the allocated pool')
        stream = h.store_stream if to_cpu else h.load_stream
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            if self._policy_transport == 'contiguous':
                self._contiguous_copy(gpu_ids, cpu_ids, to_cpu, stream)
            else:
                from vllm.v1.simple_kv_offload.cuda_mem_ops import copy_blocks
                if to_cpu:
                    copy_blocks(gpu_ids, cpu_ids, h._backend._store_params)
                else:
                    copy_blocks(cpu_ids, gpu_ids, h._backend._load_params)
            end.record()
        stream.synchronize()
        return {'rank': self.rank, 'blocks': len(gpu_ids),
                'seconds': start.elapsed_time(end) / 1000.0}

    def _staging(self, key, nbytes, device):
        import torch

        buffer = self._policy_staging.get((key, device))
        if buffer is None or buffer.numel() < nbytes:
            buffer = (torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) if device == 'cpu'
                      else torch.empty(nbytes, dtype=torch.uint8, device='cuda'))
            self._policy_staging[(key, device)] = buffer
        return buffer[:nbytes]

    def _contiguous_copy(self, gpu_ids, cpu_ids, to_cpu, stream):
        import torch

        h = self._policy_kv
        gpu_index = torch.tensor(gpu_ids, dtype=torch.long, device='cuda')
        cpu_index = torch.tensor(cpu_ids, dtype=torch.long)
        layers = list(h.gpu_kv_caches.items())
        sizes = [len(gpu_ids) * t[0].numel() * t.element_size() for _, t in layers]
        total = sum(sizes)
        device_buffer = self._staging('device', total, 'cuda')
        host_buffer = self._staging('host', total, 'cpu')
        offset = 0
        if to_cpu:
            for (name, tensor), size in zip(layers, sizes):
                gathered = tensor.index_select(0, gpu_index)
                device_buffer[offset:offset + size].view(gathered.dtype).copy_(gathered.flatten())
                offset += size
            host_buffer.copy_(device_buffer, non_blocking=True)
            stream.synchronize()
            offset = 0
            for (name, tensor), size in zip(layers, sizes):
                target = h.cpu_kv_caches[name]
                chunk = host_buffer[offset:offset + size].view(target.dtype)
                target.index_copy_(0, cpu_index, chunk.view(len(cpu_ids), *target.shape[1:]))
                offset += size
        else:
            for (name, tensor), size in zip(layers, sizes):
                source = h.cpu_kv_caches[name].index_select(0, cpu_index)
                host_buffer[offset:offset + size].view(source.dtype).copy_(source.flatten())
                offset += size
            device_buffer.copy_(host_buffer, non_blocking=True)
            offset = 0
            for (name, tensor), size in zip(layers, sizes):
                chunk = device_buffer[offset:offset + size].view(tensor.dtype)
                tensor.index_copy_(0, gpu_index, chunk.view(len(gpu_ids), *tensor.shape[1:]))
                offset += size

    def policy_model_elapsed(self):
        elapsed = 0.0
        for start, end in self._policy_model_events:
            end.synchronize()
            elapsed += start.elapsed_time(end) / 1000.0
        self._policy_model_events.clear()
        return {'rank': self.rank, 'seconds': elapsed}

    def policy_kv_read_cpu(self, cpu_ids):
        import torch

        h = self._policy_kv
        if not cpu_ids or any(not 0 <= i < h.num_cpu_blocks for i in cpu_ids):
            raise ValueError('invalid CPU export blocks')
        indices = torch.tensor(cpu_ids, dtype=torch.long)
        data = {name: tensor.index_select(0, indices).numpy().tobytes()
                for name, tensor in h.cpu_kv_caches.items()}
        return {'rank': self.rank, 'data': data}

    def policy_kv_write_cpu(self, cpu_ids, rank_payloads):
        import torch

        h = self._policy_kv
        if not cpu_ids or any(not 0 <= i < h.num_cpu_blocks for i in cpu_ids):
            raise ValueError('invalid CPU import blocks')
        payload = rank_payloads[self.rank]
        if payload['rank'] != self.rank or set(payload['data']) != set(h.cpu_kv_caches):
            raise ValueError('migration rank or KV layout mismatch')
        indices = torch.tensor(cpu_ids, dtype=torch.long)
        for name, target in h.cpu_kv_caches.items():
            data = payload['data'][name]
            expected = len(cpu_ids) * target[0].numel() * target.element_size()
            if len(data) != expected:
                raise ValueError('migration payload has the wrong byte length')
        for name, target in h.cpu_kv_caches.items():
            source = torch.frombuffer(bytearray(payload['data'][name]), dtype=target.dtype)
            target.index_copy_(0, indices, source.reshape(len(cpu_ids), *target.shape[1:]))
        return {'rank': self.rank, 'blocks': len(cpu_ids)}
