"""A failed admission must return speculative locks, not session ownership."""
from serving.core.memory_model import Device
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
from serving.tests.test_request_cache_recovery import scheduler, memory, request
import pytest


@pytest.mark.parametrize('infercept', [True, False])
def test_running_order_survives_older_sessions_readmission(infercept):
    s = scheduler()
    s.memory = memory(128)
    s.max_num_batched_tokens = 1
    older = request(1, 0, 32)
    younger = request(2, 100, 32)
    for r, arrival, admission in [(older, 0, 2), (younger, 1, 1)]:
        r.infercept_session = infercept
        r.queue_arrival = arrival
        r.admit_seq = admission
        s.memory.cache_unfinished_req(r, Device.NPU)
    s.request.extend([younger, older])
    b = s.schedule(0, 0)
    assert b is not None
    # InferceptSessionScheduler sorts its running list by original arrival
    # every iteration. Stock vLLM keeps admission order instead.
    expected = older if infercept else younger
    assert [r.id for r in b.requests] == [expected.id]


@pytest.mark.parametrize('infercept', [True, False])
@pytest.mark.parametrize('token_budget', [1, 64])
@pytest.mark.parametrize('recovering_prefill', [True, False])
def test_pressure_victim_uses_the_native_running_order(
        infercept, token_budget, recovering_prefill):
    s = scheduler()
    s.memory = memory(48 if recovering_prefill else 64)
    s.max_num_batched_tokens = token_budget
    older = request(1, 0, 32)
    younger = request(2, 100, 32)
    survivor = older if infercept else younger
    if recovering_prefill:
        survivor.num_computed_tokens = 16
    for r, arrival, admission in [(older, 0, 2), (younger, 1, 1)]:
        r.infercept_session = infercept
        r.queue_arrival = arrival
        r.admit_seq = admission
        s.memory.cache_unfinished_req(r, Device.NPU)
    s.request.extend([younger, older])
    b = s.schedule(0, 0)
    assert b is not None
    victim = younger if infercept else older
    assert victim.n_preempted == 1
    assert survivor.n_preempted == 0
    assert [r.id for r in b.requests] == [survivor.id]
    assert sum(b.scheduled_tokens.values()) <= token_budget
    assert s.memory.npu_used + s.memory.npu_reserved <= s.memory.npu_mem


def test_failed_infercept_candidate_returns_its_speculative_prefix_lock():
    s = scheduler()
    s.memory = memory(128)
    running = request(1, 0, 32)
    running.infercept_session = True
    running.admit_seq = 1
    s.memory.cache_unfinished_req(running, Device.NPU)
    cached = request(99, 1000, 96)
    s.memory.cache_finished_req(cached, Device.NPU)
    waiting = request(2, 1000, 128)
    waiting.infercept_session = True
    waiting.num_computed_tokens = 0
    s.request.extend([running, waiting])
    # The running decode fits by reclaiming one page from the later request's
    # unowned prefix. Trying that later request must not strand this capacity.
    b = s.schedule(0, 0)
    assert b is not None
    assert [r.id for r in b.requests] == [running.id]
    assert not waiting._prefix_locked
    assert waiting.num_computed_tokens == 0
    assert s.memory.npu_used + s.memory.npu_reserved <= s.memory.npu_mem


def idle_scheduler(capacity):
    s = scheduler()
    s.memory = memory(capacity)
    s.max_num_seqs = 1
    adapter = UnifiedPolicyAdapter(
        'min-waste', 'fcfs', 'session-affinity', 1, 16,
        min_waste_profile='policies/profiles/infercept_profile_rtx6000_70B_tp2.json',
        min_waste_swap=True, min_waste_fcfs_restore=True)
    adapter.configure_host_swap(s.memory, 25)
    s.policy_hooks = adapter
    return s


@pytest.mark.parametrize('capacity,younger_start,preemptions', [
    (128, 1000, 1),  # Distinct prefixes: free 16, head needs 32.
    (80, 0, 1),     # Shared prefix must survive releasing the younger owner.
    (160, 1000, 0), # Enough free space: do not preempt unnecessarily.
])
def test_idle_recovery_excludes_heads_own_prefix(capacity, younger_start, preemptions):
    s = idle_scheduler(capacity)
    s.memory.cache_finished_req(request(99, 0, 48), Device.NPU)
    head = request(1, 0, 80)
    head.input_hash_ids[48:] = list(range(2000, 2032))
    head.num_computed_tokens = 0
    head.infercept_session = True
    younger = request(2, younger_start, 80)
    younger.num_computed_tokens = 64
    younger.infercept_session = True
    younger.infercept_retained_prefix = True
    younger.queue_arrival = 1
    s.memory.cache_unfinished_req(younger, Device.NPU)
    s.request.extend([head, younger])

    b = s.schedule(0, 0)
    assert b is not None, s._none_reason
    assert [q.id for q in b.requests] == [head.id]
    assert head.npu_cache_hit == 48
    assert younger.n_preempted == preemptions
    assert s.memory.npu_used + s.memory.npu_reserved <= s.memory.npu_mem
    # Admission's temporary reference becomes the running reference, then
    # releases normally on completion (no extra reference from recovery).
    head.num_computed_tokens = head.output - 1
    s.memory.cache_finished_req(head, Device.NPU)
    if preemptions:
        assert s.memory.npu_prefix_cache.protected_size() == 0
    else:
        assert s.memory.npu_prefix_cache.protected_size() == 64


def test_failed_idle_admission_releases_head_probe_without_lock_leak():
    s = idle_scheduler(128)
    s.memory.cache_finished_req(request(99, 0, 48), Device.NPU)
    external_owner = request(100, 1000, 64)
    s.memory.cache_unfinished_req(external_owner, Device.NPU)
    head = request(1, 0, 80)
    head.num_computed_tokens = 0
    head.infercept_session = True
    s.request.append(head)
    for now in range(3):
        assert s.schedule(now, 0) is None
        assert not head._prefix_locked
        assert not head.infercept_retained_prefix
        assert s.memory.npu_prefix_cache.protected_size() == 64
        assert s.memory.npu_prefix_cache.evictable_size() == 48
        assert s.memory.npu_reserved == 0
