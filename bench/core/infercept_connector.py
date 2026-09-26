"""Native vLLM layer hooks for explicitly reserved InferCept swap plans.

The scheduler must reserve physical blocks before queue_plan(), and retain
ownership until take_acknowledgement() succeeds across every TP rank. This
connector transports plans; the interception policy supplies those plans.
"""
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole,
    KVConnectorWorkerMetadata,
)

from .infercept_transfer import LayerSwapPlan, LayerSwapWorker


@dataclass
class SwapMetadata(KVConnectorMetadata):
    ticket: int = -1
    plan: LayerSwapPlan = LayerSwapPlan()


@dataclass
class SwapAcknowledgement(KVConnectorWorkerMetadata):
    ticket: int
    ranks: frozenset

    def aggregate(self, other):
        if not isinstance(other, SwapAcknowledgement) or other.ticket != self.ticket:
            raise RuntimeError('workers acknowledged different transfer epochs')
        if self.ranks & other.ranks:
            raise RuntimeError('duplicate worker transfer acknowledgement')
        return SwapAcknowledgement(self.ticket, self.ranks | other.ranks)


class InferceptConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        p = vllm_config.parallel_config
        if (vllm_config.scheduler_config.async_scheduling
                or vllm_config.speculative_config is not None
                or p.pipeline_parallel_size != 1 or p.data_parallel_size != 1):
            raise ValueError('InferCept transfers require synchronous TP-only execution')
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec):
            raise ValueError('InferCept transfers require one full-attention KV group')
        cfg = self._kv_transfer_config.kv_connector_extra_config
        self.host_bytes = int(cfg['cpu_bytes_per_rank'])
        self.scratch_blocks = int(cfg['scratch_blocks'])
        if (self.host_bytes < 0 or self.scratch_blocks < 0
                or (self.host_bytes == 0) != (self.scratch_blocks == 0)):
            raise ValueError('CPU capacity and staging must both be positive, or both zero for no swap')
        self.world_size = p.world_size
        self.pending = None
        self.inflight = None
        self.acknowledged = None
        self.next_ticket = 0
        self.worker = None
        self.host_handler = None
        self.worker_ticket = -1

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config):
        # Full CUDA graph replay skips the per-layer Python synchronization.
        return True

    def queue_plan(self, plan):
        if self.host_bytes == 0:
            raise ValueError('transfers are disabled with zero CPU capacity')
        if self.role != KVConnectorRole.SCHEDULER:
            raise RuntimeError('only the scheduler can submit a transfer')
        if self.pending is not None or self.inflight is not None:
            raise RuntimeError('previous transfer has not been acknowledged')
        if len(plan.store_gpu) > self.scratch_blocks:
            raise ValueError('swap-out plan exceeds reserved host staging')
        ticket = self.next_ticket
        self.next_ticket += 1
        self.pending = SwapMetadata(ticket, plan)
        return ticket

    def build_connector_meta(self, scheduler_output):
        if self.pending is None:
            if self.inflight is not None:
                raise RuntimeError('cannot advance an unacknowledged synchronous transfer')
            return SwapMetadata()
        metadata = self.pending
        self.pending = None
        self.inflight = metadata.ticket
        return metadata

    def update_connector_output(self, connector_output):
        ack = connector_output.kv_connector_worker_meta
        if ack is None:
            return
        if (not isinstance(ack, SwapAcknowledgement) or ack.ticket != self.inflight
                or ack.ranks != frozenset(range(self.world_size))):
            raise RuntimeError('incomplete or mismatched TP transfer acknowledgement')
        self.acknowledged = ack.ticket

    def take_acknowledgement(self, ticket):
        if self.pending is not None and self.pending.ticket == ticket:
            return False
        if ticket != self.inflight:
            raise ValueError('unknown transfer ticket')
        if self.acknowledged != ticket:
            return False
        self.inflight = self.acknowledged = None
        return True

    def register_kv_caches(self, kv_caches):
        if self.host_bytes == 0:
            return  # GPU-only ablation: allocate neither host cache nor staging.
        from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

        config = self._kv_cache_config
        block_bytes = sum(t.size for t in config.kv_cache_tensors) // config.num_blocks
        if self.host_bytes < (self.scratch_blocks + 1) * block_bytes:
            raise ValueError('CPU budget must cover staging plus at least one allocatable block')
        handler = SimpleCPUOffloadWorker(self._vllm_config, config, self.host_bytes)
        handler.register_kv_caches(kv_caches)
        handler._backend.shutdown()
        count = handler.num_cpu_blocks - self.scratch_blocks
        cpu = {name: tensor[:count] for name, tensor in handler.cpu_kv_caches.items()}
        scratch = {name: tensor[count:] for name, tensor in handler.cpu_kv_caches.items()}
        layers = {layer: tuple(name for name in handler.gpu_kv_caches
                              if name == layer or name.startswith(layer + '.'))
                  for layer in kv_caches}
        self.worker = LayerSwapWorker(layers, handler.gpu_kv_caches, cpu, scratch)
        self.host_handler = handler

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SwapMetadata):
            raise TypeError('unexpected InferCept connector metadata')
        self.worker_ticket = metadata.ticket
        if metadata.ticket >= 0:
            if self.worker is None:
                raise RuntimeError('KV storage is not registered')
            self.worker.start(metadata.plan)

    def wait_for_layer_load(self, layer_name):
        if self.worker_ticket >= 0:
            self.worker.wait_for_layer(layer_name)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        # Outgoing contexts are paused before this forward, so their copies
        # started before computation rather than after an active layer write.
        pass

    def wait_for_save(self):
        if self.worker_ticket >= 0:
            self.worker.finish(wait=True)

    def build_connector_worker_meta(self):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        if self.worker_ticket < 0:
            return None
        # Also drains transfer-only iterations, which skip wait_for_save().
        if not self.worker.finish(wait=True):
            raise RuntimeError('transfer was not fully published')
        return SwapAcknowledgement(self.worker_ticket,
                                   frozenset((get_tensor_model_parallel_rank(),)))

    def shutdown(self):
        if self.worker is not None:
            self.worker.finish(wait=True)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # Policy-owned CPU state is explicitly restored before admission. It
        # is not an external prefix cache discoverable by arbitrary requests.
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            raise RuntimeError('InferCept connector does not supply external prefix hits')
