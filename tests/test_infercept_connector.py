"""Native connector protocol tests; CUDA bytes/events are tested on Slurm."""
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import FullAttentionSpec

from bench.core.infercept_connector import InferceptConnector, SwapAcknowledgement
from bench.core.infercept_transfer import LayerSwapPlan


def connector(host_bytes=4096, scratch_blocks=2):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
            'cpu_bytes_per_rank': host_bytes, 'scratch_blocks': scratch_blocks}),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1,
            data_parallel_size=1, world_size=2),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None)
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=
        FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32))])
    return InferceptConnector(config, KVConnectorRole.SCHEDULER, cache)


def test_no_reuse_until_every_rank_acknowledges():
    c = connector()
    plan = LayerSwapPlan((1,), (0,), (0,), (1,))
    ticket = c.queue_plan(plan)
    assert not c.take_acknowledgement(ticket)  # queued but not dispatched
    metadata = c.build_connector_meta(None)
    assert metadata.ticket == ticket and metadata.plan == plan
    assert not c.take_acknowledgement(ticket)
    with pytest.raises(RuntimeError, match='acknowledged'):
        c.queue_plan(plan)
    with pytest.raises(RuntimeError, match='unacknowledged'):
        c.build_connector_meta(None)
    partial = SwapAcknowledgement(ticket, frozenset((0,)))
    with pytest.raises(RuntimeError, match='incomplete'):
        c.update_connector_output(SimpleNamespace(kv_connector_worker_meta=partial))
    assert not c.take_acknowledgement(ticket)
    complete = partial.aggregate(SwapAcknowledgement(ticket, frozenset((1,))))
    c.update_connector_output(SimpleNamespace(kv_connector_worker_meta=complete))
    assert c.take_acknowledgement(ticket)
    assert c.build_connector_meta(None).ticket == -1
    assert c.queue_plan(plan) > ticket


def test_acknowledgements_cannot_mix_epochs_or_duplicate_ranks():
    ack = SwapAcknowledgement(1, frozenset((0,)))
    with pytest.raises(RuntimeError, match='epochs'):
        ack.aggregate(SwapAcknowledgement(2, frozenset((1,))))
    with pytest.raises(RuntimeError, match='duplicate'):
        ack.aggregate(ack)


def test_staging_capacity_is_enforced_before_worker_submission():
    c = connector()
    with pytest.raises(ValueError, match='staging'):
        c.queue_plan(LayerSwapPlan((1, 2, 3), (0, 1, 2)))
    assert c.pending is None and c.inflight is None
    assert c.requires_piecewise_for_cudagraph({})
    assert c.get_num_new_matched_tokens(None, 0) == (0, False)


def test_native_output_aggregator_preserves_all_rank_acknowledgements():
    import copy
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput

    c = connector()
    ticket = c.queue_plan(LayerSwapPlan((1,), (0,)))
    c.build_connector_meta(None)
    outputs = []
    for rank in range(2):
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = KVConnectorOutput(kv_connector_worker_meta=
            SwapAcknowledgement(ticket, frozenset((rank,))))
        outputs.append(output)
    combined = KVOutputAggregator(2).aggregate(outputs)
    c.update_connector_output(combined.kv_connector_output)
    assert c.take_acknowledgement(ticket)


def test_zero_host_capacity_allocates_no_storage_and_forbids_transfers():
    c = connector(0, 0)
    c.register_kv_caches({})
    assert c.worker is None and c.host_handler is None
    assert c.build_connector_meta(None).ticket == -1
    with pytest.raises(ValueError, match='disabled'):
        c.queue_plan(LayerSwapPlan((1,), (0,)))
    for capacity, staging in ((0, 2), (4096, 0), (-1, 0), (0, -1)):
        with pytest.raises(ValueError, match='both'):
            connector(capacity, staging)
