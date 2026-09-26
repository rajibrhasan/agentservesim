"""Admission reconstruction from SAGA real step 56760 (2026-09-21).

The trace records four running contexts, not just the two highlighted programs.
Synthetic disjoint IDs preserve lengths and the waiting request's measured hit;
they do not reconstruct the native block-sharing graph or the 74-second duration.
"""
import pytest

from serving.core.memory_model import Device
from serving.tests.test_request_cache_recovery import memory, request, scheduler


@pytest.mark.parametrize('contexts,shared,admitted', [
    ([59498, 31020, 3512, 698], False, False),
    ([59498, 31020, 3512, 698], True, False),
    ([59498], False, False),
    ([59498], True, True),
    ([31020, 3512, 698], False, True),
    ([], False, True),
])
def test_recorded_saga_full_prompt_fit(contexts, shared, admitted):
    s = scheduler()
    # 6,667 physical blocks, one null block, 16 tokens/block in the real trace.
    s.memory = memory((6667 - 1) * 16)
    s.max_num_batched_tokens = 16384
    m = s.memory
    for index, length in enumerate(contexts):
        r = request(index + 1, start=(index + 1) * 100000, length=length)
        r.admit_seq = index
        r.is_init = False
        m.cache_unfinished_req(r, Device.NPU)
        s.request.append(r)

    cached = request(90, start=900000, length=8560)
    m.cache_finished_req(cached, Device.NPU)
    waiting = request(91, start=900000, length=47291)
    if shared:
        waiting.input_hash_ids[:8560] = s.request[0].input_hash_ids[:8560]
    waiting.num_computed_tokens = 0
    s.request.append(waiting)
    m.prefix_match(waiting)
    assert waiting.npu_cache_hit == 8560
    batch = s.schedule(0, 0)
    assert batch is not None
    assert (waiting in batch.requests) is admitted
    assert s.num_preemptions == 0
    assert m.npu_used + m.npu_reserved <= m.npu_mem
    if not admitted:
        assert len(batch.requests) == len(contexts)
        assert all(batch.scheduled_tokens[r.id] == 1 for r in batch.requests)
        assert waiting.admit_seq is None
