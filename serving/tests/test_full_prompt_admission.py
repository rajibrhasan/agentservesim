"""Waiting feasibility and running order match the vLLM V1 scheduler."""
from serving.tests.test_request_cache_recovery import scheduler, memory, request
from serving.tests.test_impossible_allocation_spares_pins import setup_shared_pins, snapshot
from serving.core.memory_model import Device


def test_full_prompt_must_fit_even_when_first_chunk_fits():
    s = scheduler()
    s.memory = memory(48)
    s.max_num_batched_tokens = 16
    r = request(1, 1000, 64)
    r.num_computed_tokens = 0
    s.request.append(r)
    assert s.schedule(0, 0) is None
    assert r.admit_seq is None
    assert s.memory.npu_reserved == 0


def test_failed_full_prompt_check_preserves_pins():
    s = scheduler()
    s.memory, adapter = setup_shared_pins()
    s.max_num_batched_tokens = 16
    r = request(1, 1000, 64)
    r.num_computed_tokens = 0
    s.request.append(r)
    before = snapshot(s.memory, adapter)
    assert s.schedule(0, 0) is None
    assert snapshot(s.memory, adapter) == before


def test_feasible_full_prompt_allocates_only_current_chunk():
    s = scheduler()
    s.memory, adapter = setup_shared_pins()
    s.max_num_batched_tokens = 16
    r = request(1, 1000, 48)
    r.num_computed_tokens = 0
    s.request.append(r)
    batch = s.schedule(0, 0)
    assert batch.total_len == 16
    assert r.kv_reserved == s.memory.get_kv(16)
    assert adapter.stats['reclaimed_forced'] == 1
    assert s.memory.npu_prefix_cache.protected_size() == 32


def test_running_prefill_precedes_later_decode_in_token_budget():
    s = scheduler()
    s.max_num_batched_tokens = 16
    prefill = request(1, 0, 64)
    prefill.num_computed_tokens = 16
    prefill.admit_seq = 0
    decode = request(2, 100, 16)
    decode.admit_seq = 1
    s.request.extend([decode, prefill])
    batch = s.schedule(0, 0)
    assert [r.id for r in batch.requests] == [prefill.id]
    assert batch.total_len == 16


def test_full_prompt_check_credits_cached_prefix():
    s = scheduler()
    s.memory = memory(64)
    cached = request(99, 0, 32)
    s.memory.cache_unfinished_req(cached, Device.NPU)
    s.memory.unlock_prefix(cached, Device.NPU)
    s.max_num_batched_tokens = 16
    r = request(1, 0, 64)
    r.num_computed_tokens = 0
    s.request.append(r)
    batch = s.schedule(0, 0)
    assert batch is not None
    assert r.npu_cache_hit == 32
    assert batch.total_len == 16


def test_full_prompt_check_includes_earlier_candidates_chunk():
    s = scheduler()
    s.memory = memory(64)
    s.max_num_batched_tokens = 32
    s.long_prefill_token_threshold = 16
    first = request(1, 0, 16)
    second = request(2, 1000, 64)
    first.num_computed_tokens = second.num_computed_tokens = 0
    s.request.extend([first, second])
    batch = s.schedule(0, 0)
    assert [r.id for r in batch.requests] == [first.id]
    assert second.admit_seq is None
