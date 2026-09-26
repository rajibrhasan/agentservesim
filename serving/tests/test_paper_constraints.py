from types import SimpleNamespace as NS

import pytest

from bench.core.effective_config import effective_engine_configs
from serving.tests.test_request_cache_recovery import scheduler, memory, MODEL


def test_measured_capacity_is_exact_and_cannot_be_changed_after_allocation():
    m = memory(64)
    m.set_kv_capacity_tokens(48)
    assert m.npu_mem - m.weight == m.get_kv(48)
    assert m.npu_prefix_cache.capacity == m.get_kv(48)
    with pytest.raises(ValueError):
        m.set_kv_capacity_tokens(47)
    m.npu_reserved = m.get_kv(16)
    with pytest.raises(RuntimeError):
        m.set_kv_capacity_tokens(64)


def test_request_context_limit_checks_prompt_plus_generation():
    s = scheduler()
    s.max_model_len = 32
    with pytest.raises(ValueError, match='exceeds'):
        s.add_request([1, MODEL, 16, 33, 0, 0])
    s.add_request([1, MODEL, 16, 32, 0, 0])
    assert len(s.request) == 1


def test_resolved_metadata_keeps_per_instance_capacity_and_async_default():
    def engine(blocks):
        return NS(vllm_config=NS(
            cache_config=NS(num_gpu_blocks=blocks, block_size=16),
            model_config=NS(max_model_len=65536),
            scheduler_config=NS(max_num_seqs=128, max_num_batched_tokens=16384,
                                async_scheduling=True, scheduler_reserve_full_isl=True)))
    rows = effective_engine_configs([engine(100), engine(98)])
    assert [r['kv_cache_tokens'] for r in rows] == [1600, 1568]
    assert all(r['async_scheduling'] for r in rows)
