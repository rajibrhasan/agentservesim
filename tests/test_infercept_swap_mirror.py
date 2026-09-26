"""Causal host-copy regression tests; not native InferCept parity claims."""
import json
import tempfile
import math
from types import SimpleNamespace

import pytest
from serving.core.memory_model import Device, MemoryModel
from serving.core.request import Request, Batch
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
from serving.core.trace_generator import _host_transfer_rows
from policies.utils.waste_model import WasteProfile, t_fwd_s

MODEL = 'meta-llama/Llama-3.1-8B'
BLOCK = 16
PROFILE = {"a": 0.0279, "c": 15.4, "S": 384}


def profile_path():
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(PROFILE, f)
    f.close()
    return f.name


def memory(link_gbs=25.0, tp=1):
    m = MemoryModel(MODEL, 0, 0, tp, tp, 80, 80, BLOCK, 16, True, False, None, None)
    m.enable_swap_tier()
    m.cpu_mem_bw_gbs = link_gbs
    return m


def finished_request(req_id, tokens):
    """A request that has run to completion and been cached on the NPU:
    the state the router sees when it fires on_turn_complete."""
    r = Request(req_id, MODEL, len(tokens), len(tokens) + 1, 0, 0,
                input_hash_ids=list(tokens), output_hash_ids=[10 ** 9])
    r.num_computed_tokens = len(tokens)
    r.latency = 2_000_000_000
    r.queuing_delay = 0
    return r


def adapter(swap=True, gap_s=10.0, scheduling='plas'):
    return UnifiedPolicyAdapter(retention_value="min-waste", scheduling_value=scheduling,
                                routing_value="session-affinity", num_instances=1,
                                block_size=BLOCK, min_waste_profile=profile_path(),
                                default_gap_s=gap_s, min_waste_swap=swap,
                                min_waste_fcfs_restore=scheduling == 'fcfs')


def row(session, idx, arrival_ns, req_index):
    return {"index": req_index, "session_id": session,
            "sub_request_index": idx, "arrival_time_ns": arrival_ns}


def setup_copy(tokens=64, tp=1, link=25.0, session='s0', scheduling='plas'):
    ad, mem = adapter(scheduling=scheduling), memory(tp=tp, link_gbs=link)
    ad.configure_host_swap(mem, link)
    req = finished_request(100, list(range(1, tokens + 1)))
    mem.cache_finished_req(req, Device.NPU)
    ad.select_instance(row(session, 0, 0, 100), lambda: 0, 0)
    ad.on_turn_routed(row(session, 0, 0, 100))
    ad.on_turn_complete(req, session, 0, 'tool', tokens, mem, 2_000_000_000)
    # Copy/ACK tests need memory pressure now that ample GPU space suppresses
    # outgoing transfers. Keep room for a second small copy source.
    mem.npu_mem = mem.weight + mem.get_kv(tokens + 64)
    mem.npu_prefix_cache.capacity = mem.get_kv(tokens + 64)
    return ad, mem, req


def batch(batch_id=0, tokens=128, now=2_000_000_000, load=0):
    return Batch(batch_id, MODEL, tokens, 0, [], [], 0, 0, [], [], [],
                 now, 0, load=load)


def test_callback_cannot_backdate_copy_or_release_gpu():
    ad, mem, _ = setup_copy()
    assert ad.retention_exec.decisions[-1].blocks == 0
    assert ad.retention_exec.decisions[-1].info['transfer_pending']
    assert mem.npu_prefix_cache.total_size() == 64
    assert mem.npu_prefix_cache.protected_size() == 64
    assert mem.cpu_used == 0
    assert ad.stats['swapped_tokens'] == 0


def test_copy_publishes_only_after_transfer_batch_completes():
    ad, mem, req = setup_copy()
    b = batch()
    mem.host_swap.plan(b)
    assert b.host_store_bytes == mem.get_kv(64)
    assert mem.cpu_swap_reserved == mem.get_kv(64)
    assert mem.cpu_used == 0
    assert mem.npu_prefix_cache.total_size() == 64
    mem.host_swap.complete(b)
    assert mem.cpu_swap_reserved == 0
    assert mem.cpu_used == mem.get_kv(64)
    assert mem.npu_prefix_cache.total_size() == 0
    assert ad.stats['swapped_tokens'] == 64
    successor = finished_request(101, req.input_hash_ids + [1000] * 16)
    successor.num_computed_tokens = 0
    mem.prefix_match(successor)
    assert successor.storage_cache_hit == 64
    assert successor.npu_cache_hit == 0
    # Duplicate completion notifications do not double-count transfers.
    mem.host_swap.complete(b)
    assert ad.stats['swapped_tokens'] == 64


def test_ample_gpu_capacity_does_not_start_proactive_copy():
    _, mem, _ = setup_copy()
    mem.npu_mem = mem.weight + mem.get_kv(100000)
    mem.npu_prefix_cache.capacity = mem.get_kv(100000)
    b = batch()
    mem.host_swap.plan(b)
    assert b.host_store_bytes == 0
    assert not mem.host_swap.pending
    assert mem.cpu_swap_reserved == 0


def test_zero_host_capacity_keeps_session_controller_without_transfers():
    ad = adapter(scheduling='fcfs')
    mem = MemoryModel(MODEL, 0, 0, 1, 1, 80, 0, BLOCK, 16,
                      True, False, None, None)
    ad.configure_host_swap(mem, 25)
    old = finished_request(100, list(range(64)))
    mem.cache_finished_req(old, Device.NPU)
    ad.on_turn_routed(row('s0', 0, 0, 100))
    ad.on_turn_complete(old, 's0', 0, 'tool', 64, mem, 2_000_000_000)
    b = batch()
    mem.host_swap.plan(b)
    mem.host_swap.complete(b)
    assert mem.cpu_mem == 0
    assert mem.cpu_used == mem.cpu_swap_reserved == 0
    assert b.host_store_bytes == b.load == 0
    ad.on_turn_routed(row('s0', 1, 2_000_000_001, 101))
    successor = finished_request(101, list(range(80)))
    successor.session_id = 's0'
    successor.num_computed_tokens = 0
    ad.bind_infercept_continuation(successor, mem)
    assert successor.infercept_session
    assert successor.storage_cache_hit == 0


def test_host_match_reuses_only_complete_blocks():
    _, mem, req = setup_copy()
    b = batch()
    mem.host_swap.plan(b)
    mem.host_swap.complete(b)
    successor = finished_request(101, req.input_hash_ids[:17] + [1000] * 47)
    successor.num_computed_tokens = 0
    mem.prefix_match(successor)
    assert successor.storage_cache_hit == BLOCK


def test_zero_forward_batch_has_zero_store_budget():
    _, mem, _ = setup_copy()
    mem.last_batch_tokens = 2048  # stale history must have no effect
    b = batch(tokens=0)
    mem.host_swap.plan(b)
    assert b.host_store_bytes == 0
    assert not mem.host_swap.pending
    assert mem.cpu_swap_reserved == 0


def test_context_larger_than_budget_copies_in_chunks():
    ad, mem, _ = setup_copy()
    forward = t_fwd_s(WasteProfile(**PROFILE), 128)
    mem.cpu_mem_bw_gbs = mem.get_kv(BLOCK) * 1.01 / forward / 1e9
    for i in range(4):
        b = batch(i)
        mem.host_swap.plan(b)
        assert b.host_store_bytes == mem.get_kv(BLOCK)
        assert mem.npu_prefix_cache.total_size() == 64
        mem.host_swap.complete(b)
        assert mem.cpu_used == mem.get_kv((i + 1) * BLOCK)
    assert ad.stats['swapped_tokens'] == 64
    assert mem.npu_prefix_cache.total_size() == 0


def test_restores_consume_the_same_transfer_budget():
    _, mem, _ = setup_copy()
    forward = t_fwd_s(WasteProfile(**PROFILE), 128)
    mem.cpu_mem_bw_gbs = mem.get_kv(BLOCK) * 1.01 / forward / 1e9
    b = batch(load=mem.get_kv(BLOCK))
    mem.host_swap.plan(b)
    assert b.host_store_bytes == 0
    assert mem.cpu_swap_reserved == 0


def test_host_capacity_is_reserved_across_pending_copies():
    ad, mem, _ = setup_copy()
    mem.second_tier_prefix_cache.capacity = mem.get_kv(64)
    first = batch(0)
    mem.host_swap.plan(first)
    other = finished_request(102, list(range(1000, 1064)))
    mem.cache_finished_req(other, Device.NPU)
    mem.host_swap.enqueue('other', other, 2_000_000_000)
    second = batch(1)
    mem.host_swap.plan(second)
    assert second.host_store_bytes == 0
    assert mem.avail_size(Device.CPU) == 0
    mem.host_swap.complete(first)
    assert mem.cpu_used == mem.second_tier_prefix_cache.capacity
    assert mem.cpu_swap_reserved == 0


@pytest.mark.parametrize('inflight', [False, True])
def test_arrival_cancels_pending_work_without_recycling_live_source(inflight):
    ad, mem, _ = setup_copy()
    b = batch()
    if inflight:
        mem.host_swap.plan(b)
    ad.select_instance(row('s0', 1, 2_000_000_001, 101), lambda: 0, 2_000_000_001)
    ad.on_turn_routed(row('s0', 1, 2_000_000_001, 101))
    assert not mem.host_swap.queued
    assert mem.npu_prefix_cache.protected_size() == (64 if inflight else 0)
    if inflight:
        mem.host_swap.complete(b)
    assert mem.npu_prefix_cache.total_size() == 64
    assert mem.npu_prefix_cache.protected_size() == 0
    assert mem.cpu_swap_reserved == 0


def test_tp_budget_and_duration_use_per_rank_bytes():
    _, mem, _ = setup_copy(tp=2)
    b = batch()
    mem.host_swap.plan(b)
    assert b.host_store_bytes == mem.get_kv(64)
    assert mem.cpu_swap_reserved == mem.get_kv(64) * 2
    rows = [['embedding', '10', 'REMOTE:0', '32', 'LOCAL', '0',
             'LOCAL', '0', 'NONE', '0', 'NONE']]
    slow = _host_transfer_rows(b, rows)
    assert slow[0][2:4] == ['REMOTE:0', '32']
    assert int(slow[0][1]) == math.ceil(mem.get_kv(64) / 25.0)
    b.host_link_bytes_s *= 2
    fast = _host_transfer_rows(b, rows)
    assert int(slow[0][1]) in (2 * int(fast[0][1]), 2 * int(fast[0][1]) - 1)


@pytest.mark.parametrize('rate', [None, 0, -1, float('nan'), float('inf')])
def test_unmeasured_or_invalid_link_fails_before_mutation(rate):
    ad, mem = adapter(), memory()
    with pytest.raises(ValueError, match='measured per-rank'):
        ad.configure_host_swap(mem, rate)
    assert mem.host_swap is None


def test_reconsider_uses_elapsed_gap_and_releases_obsolete_source():
    ad, mem, _ = setup_copy()
    mem.host_swap.reconsider(2_000_000_000, [])
    assert mem.npu_prefix_cache.protected_size() == 64
    mem.host_swap.reconsider(102_000_000_000, [])
    assert not mem.host_swap.queued
    assert mem.npu_prefix_cache.total_size() == 0
    assert ad.stats['swapped_tokens'] == 0


def test_idle_swap_wakeup_releases_source_after_policy_threshold():
    _, mem, _ = setup_copy(tokens=4096)
    swap = mem.host_swap
    now = 2_000_000_000
    wake = swap.next_reconsider_ns(now, [])
    assert wake > now + 1
    assert (wake - now) % 1_000_000 == 0
    assert swap.reconsider(wake - 1_000_000, []) == 0
    assert mem.npu_prefix_cache.protected_size() == 4096
    assert swap.reconsider(wake, []) == 1
    assert mem.npu_prefix_cache.protected_size() == 0
    assert swap.next_reconsider_ns(wake, []) is None


@pytest.mark.parametrize('threshold_ns', [0, 500_000, 1_000_000, 1_500_000, 250_000_000])
def test_idle_jump_matches_explicit_one_ms_polling(threshold_ns):
    from serving.core.host_swap import Copy, MinWasteSwapPolicy
    from unittest.mock import patch
    policy = MinWasteSwapPolicy(WasteProfile(**PROFILE))
    now = 10_000_123_456  # Polls are relative to the failed check, not time zero.
    copy = Copy('test', [1], None, now)
    discard = threshold_ns / 1e9
    with patch.object(policy, 'score', return_value=(0, 0, discard)):
        wake = policy.next_reconsider_ns(copy, now, [])
    explicit = now + 1_000_000
    while explicit - now <= threshold_ns:
        explicit += 1_000_000
    assert wake == explicit


def test_idle_swap_wakeup_does_not_release_inflight_copy():
    _, mem, _ = setup_copy()
    b = batch()
    mem.host_swap.plan(b)
    assert mem.host_swap.pending
    assert mem.host_swap.next_reconsider_ns(102_000_000_000, []) is None
    assert mem.host_swap.reconsider(102_000_000_000, []) == 0
    assert mem.npu_prefix_cache.protected_size() == 64


def test_waiting_request_can_run_after_idle_swap_deadline():
    from serving.tests.test_request_cache_recovery import scheduler, request
    _, mem, _ = setup_copy(tokens=4096)
    mem.npu_mem = mem.weight + mem.get_kv(4096 + 16)
    mem.mem_for_kv = mem.get_kv(4096 + 16)
    mem.npu_prefix_cache.capacity = mem.mem_for_kv
    s = scheduler()
    s.memory = mem
    s.max_num_batched_tokens = 16
    req = request(200, 10000, 32)
    req.num_computed_tokens = 0
    s.request.append(req)
    now = 2_000_000_000
    assert s.schedule(now, 0) is None
    assert not s.inflight
    wake = mem.host_swap.next_reconsider_ns(now, s.request)
    assert wake > now
    assert s.schedule(wake, 0) is not None


def test_preempt_swap_has_no_idle_discard_deadline():
    from serving.core.host_swap import PreemptSwapPolicy
    _, mem, _ = setup_copy()
    mem.host_swap.policy = PreemptSwapPolicy()
    assert mem.host_swap.next_reconsider_ns(102_000_000_000, []) is None
    assert mem.host_swap.reconsider(102_000_000_000, []) == 0


def test_named_workflow_successor_cancels_copy():
    ad, mem = adapter(), memory()
    ad.configure_host_swap(mem, 25.0)
    req = finished_request(100, list(range(64)))
    mem.cache_finished_req(req, Device.NPU)
    ad.on_turn_complete(req, 'workflow', 'first', 'tool', 64, mem, 2_000_000_000)
    ad.on_turn_routed({'workflow_id': 'workflow', 'node_id': 'second',
                      'arrival_time_ns': 2_000_000_001})
    assert not mem.host_swap.queued
    assert mem.npu_prefix_cache.protected_size() == 0


def test_budgeted_copy_precedes_discard_of_old_context():
    _, mem, _ = setup_copy()
    b = batch(now=102_000_000_000)
    mem.host_swap.plan(b)
    assert b.host_store_bytes == mem.get_kv(64)
    assert mem.npu_prefix_cache.protected_size() == 64
    assert mem.cpu_used == 0
    mem.host_swap.complete(b)
    assert mem.cpu_used == mem.get_kv(64)


def test_teardown_releases_queued_sources_and_completed_cpu_copies():
    _, mem, _ = setup_copy()
    b = batch()
    mem.host_swap.plan(b)
    with pytest.raises(RuntimeError, match='unfinished transfers'):
        mem.free_prefix_cache()
    mem.host_swap.complete(b)
    mem.free_prefix_cache()
    assert mem.cpu_used == 0
    assert mem.npu_used == mem.weight


# 6. restore sizing through the scheduler --------------------------------
def _tight_scheduler(pool_tokens, ad):
    """A one-instance scheduler whose NPU pool holds exactly pool_tokens of
    KV beyond the weights, with the host-copy tier configured."""
    from serving.core.scheduler import Scheduler
    s = Scheduler(MODEL, 0, 0, 128, 64, 1, 1, 1, 80, 80, 0,
                  None, BLOCK, 16, 0, False, True, False, None, None, True)
    m = s.memory
    m.npu_mem = m.weight + m.get_kv(pool_tokens)
    m.mem_for_kv = m.get_kv(pool_tokens)
    m.npu_prefix_cache.capacity = m.mem_for_kv
    m.kv_protection = ad
    ad.configure_host_swap(m, 25.0)
    return s


def _swap_out(s, ad, tokens):
    """A finished request's chain is queued, copied by one transfer batch and
    dropped from the NPU: the successor can only restore it from the host."""
    m = s.memory
    r0 = finished_request(1, tokens)
    m.cache_finished_req(r0, Device.NPU)
    ad.select_instance(row("s0", 0, 0, 1), lambda: 0, 0)
    ad.on_turn_routed(row("s0", 0, 0, 1))
    ad.on_turn_complete(r0, "s0", 0, "tool", len(tokens), m, 2_000_000_000)
    assert ad.retention_exec.decisions[-1].action == "swap"
    b = batch(batch_id=900, tokens=4096)
    m.host_swap.plan(b)
    m.host_swap.complete(b)
    assert m.npu_prefix_cache.total_size() == 0
    assert m.second_tier_prefix_cache.total_size() == len(tokens)


def test_restored_tokens_are_reserved_on_admission():
    """Board job 42522214_5: a successor restored from the CPU tier counted
    its restored tokens as computed, reserved only its chunk, and the insert
    that completed the chunk overflowed a full pool. Pool of 56 tokens: the
    successor (32 restored + 16 new) must reserve 48 and keep a 16-token
    newcomer out; with the old sizing both were admitted and the successor's
    publication raised KVCapacityError."""
    ad = adapter()
    s = _tight_scheduler(56, ad)
    m = s.memory
    tokens = list(range(1, 1 + 2 * BLOCK))
    _swap_out(s, ad, tokens)

    succ_ids = tokens + list(range(500, 500 + BLOCK))
    succ = Request(2, MODEL, len(succ_ids), len(succ_ids) + 8, 0, 0,
                   input_hash_ids=list(succ_ids), output_hash_ids=[])
    other = Request(3, MODEL, BLOCK, BLOCK + 8, 0, 0,
                    input_hash_ids=list(range(900, 900 + BLOCK)), output_hash_ids=[])
    s.request.extend([succ, other])

    batch = s.schedule(0, 0)
    assert [r.id for r in batch.requests] == [succ.id]  # newcomer does not fit
    assert succ.storage_cache_hit == len(tokens) and succ.npu_cache_hit == 0
    assert succ.storage_restored is True
    assert succ.chunk_len == BLOCK
    # Reservation covers the restored pages and the new chunk; the load is
    # charged for the restore.
    assert m.npu_reserved == m.get_kv(len(tokens) + BLOCK)
    assert batch.load == len(tokens) * m.get_kv(1)

    # The chunk completes: the insert of computed (restored + new) tokens fits
    # the reservation exactly, and the accounts agree.
    succ.num_computed_tokens += succ.chunk_len
    m.cache_unfinished_req(succ, Device.NPU)
    assert m.npu_reserved == 0
    assert m.npu_prefix_cache.total_size() == len(succ_ids)
    assert m.npu_used - m.weight == m.npu_prefix_cache.total_size() * m._bytes_per_token
    assert m.npu_used <= m.npu_mem


def test_old_sizing_reproduces_the_board_failure():
    """Same scenario with the restored tokens treated as resident (the
    pre-fix sizing): the newcomer is co-admitted and the successor's
    completing insert does not fit."""
    from serving.core.memory_model import KVCapacityError
    ad = adapter()
    s = _tight_scheduler(56, ad)
    m = s.memory
    tokens = list(range(1, 1 + 2 * BLOCK))
    _swap_out(s, ad, tokens)
    succ_ids = tokens + list(range(500, 500 + BLOCK))
    succ = Request(2, MODEL, len(succ_ids), len(succ_ids) + 8, 0, 0,
                   input_hash_ids=list(succ_ids), output_hash_ids=[])
    other = Request(3, MODEL, BLOCK, BLOCK + 8, 0, 0,
                    input_hash_ids=list(range(900, 900 + BLOCK)), output_hash_ids=[])
    m.prefix_match(succ)
    succ.storage_restored = True  # pretend the restore already happened
    s.request.extend([succ, other])
    batch = s.schedule(0, 0)
    assert [r.id for r in batch.requests] == [succ.id, other.id]
    assert m.npu_reserved == m.get_kv(2 * BLOCK)
    succ.num_computed_tokens += succ.chunk_len
    try:
        m.cache_unfinished_req(succ, Device.NPU)
    except KVCapacityError:
        pass
    else:
        raise AssertionError("old sizing must overflow the pool")


# 7. FCFS chunk restore ---------------------------------------------------
def _restorable(gate_mem, rid, restore_tokens, prompt=256, arrival=0):
    """A waiting prefill whose context really does sit on the host tier.

    The gate probes the tiers rather than reading the request, so the tokens
    have to be inserted for real; setting storage_cache_hit by hand would test
    a field the gate no longer trusts.
    """
    ids = list(range(rid * 10_000, rid * 10_000 + prompt))
    r = Request(rid, MODEL, prompt, prompt + 8, arrival, 0,
                input_hash_ids=ids, output_hash_ids=[])
    r.num_computed_tokens = 0
    if restore_tokens:
        gate_mem.second_tier_prefix_cache.insert(ids[:restore_tokens])
        gate_mem.apply_kv_cache_events()
    return r


def _gate(link_gbs=25.0):
    from serving.core.host_swap import RestoreGate
    m = memory(link_gbs=link_gbs)
    return RestoreGate(m, WasteProfile(**PROFILE), {}), m


def test_restore_gate_admits_in_arrival_order_within_the_window():
    gate, m = _gate()
    reqs = [_restorable(m, i, 512, prompt=1024, arrival=i) for i in (1, 2, 3)]
    kept = gate.admit(reqs, 16384)
    assert [r.id for r in kept] == [r.id for r in kept]        # a prefix
    assert kept == reqs[:len(kept)]                            # never reordered


def test_a_held_restore_holds_everything_behind_it():
    """Otherwise the order stops being first-come."""
    gate, m = _gate(link_gbs=0.001)     # ~1 MB/s: nothing fits the window
    reqs = [_restorable(m, 1, 4096, prompt=8192, arrival=0),
            _restorable(m, 2, 16, prompt=64, arrival=1)]
    kept = gate.admit(reqs, 16384)
    assert kept == reqs[:1]             # oversized head progresses alone
    assert gate.stats["restore_oversized"] == 1


def test_requests_needing_no_restore_are_untouched():
    gate, m = _gate(link_gbs=0.001)
    plain = _restorable(m, 1, 0)          # nothing of it is on the host tier
    assert gate.admit([plain], 16384) == [plain]


def test_an_already_restored_request_does_not_pay_again():
    gate, m = _gate(link_gbs=0.001)
    done = _restorable(m, 1, 4096, prompt=8192)
    done.storage_restored = True
    assert gate.admit([done], 16384) == [done]


def test_a_bigger_batch_hides_more_restore():
    """The window is the batch being assembled, not the last one that ran."""
    # At 2 GB/s a 16-token forward pass hides nothing; a 16k-token one hides
    # 450 blocks, comfortably more than this 2048-token (128-block) restore.
    gate, m = _gate(link_gbs=2.0)
    small = [_restorable(m, 1, 2048, prompt=4096)]
    large = [_restorable(m, 2, 2048, prompt=16384)]
    assert gate.admit(small, 16384) == small
    assert gate.stats["restore_oversized"] == 1
    assert gate.admit(large, 16384) == large
    assert gate.stats["restore_oversized"] == 1


def test_oversized_restore_makes_scheduler_progress_and_charges_full_load():
    ad = UnifiedPolicyAdapter(
        retention_value='min-waste', scheduling_value='fcfs',
        routing_value='session-affinity', num_instances=1, block_size=BLOCK,
        min_waste_profile=profile_path(), default_gap_s=10.,
        min_waste_swap=True, min_waste_fcfs_restore=True)
    s = _tight_scheduler(128, ad)
    s.policy_hooks = ad
    m = s.memory
    m.cpu_mem_bw_gbs = .001
    ids = list(range(1, 49))
    m.second_tier_prefix_cache.insert(ids[:32])
    m.apply_kv_cache_events()
    r = Request(50, MODEL, 48, 56, 0, 0,
                input_hash_ids=ids, output_hash_ids=[])
    s.request.append(r)
    b = s.schedule(0, 0)
    assert b is not None and b.requests == [r]
    assert b.load == m.get_kv(32)
    assert b.host_link_bytes_s == 1e6
    assert m.npu_reserved == m.get_kv(48)
    assert ad.stats['restore_oversized'] == 1


def test_restore_gate_refuses_an_unmeasured_link():
    gate, m = _gate()
    req_ = _restorable(m, 1, 512, prompt=1024)
    m.cpu_mem_bw_gbs = 0
    try:
        gate.admit([req_], 16384)
    except ValueError as e:
        assert "measured per-rank" in str(e)
    else:
        raise AssertionError("accepted an unmeasured link")


def test_the_gate_probes_the_tiers_rather_than_the_request_fields():
    """Job 42596388: the gate ran before the scheduler's prefix_match, so a
    waiting request's storage_cache_hit was still its initial zero and every
    restore looked free. 1.3M blocks swapped, zero restores gated."""
    gate, m = _gate(link_gbs=0.001)
    r = _restorable(m, 1, 4096, prompt=8192)
    # Exactly the state a waiting request is in when the gate sees it.
    assert r.storage_cache_hit == 0 and r.npu_cache_hit == 0
    assert gate._restore_tokens(r) > 0      # the probe sees the host copy
    assert gate.admit([r], 16384) == [r]
    assert gate.stats['restore_oversized'] == 1


def test_a_context_already_on_the_npu_needs_no_restore():
    gate, m = _gate(link_gbs=0.001)
    r = _restorable(m, 1, 4096, prompt=8192)
    # The same prefix is resident on the NPU: nothing has to come back.
    m.npu_prefix_cache.insert(r.input_hash_ids[:4096])
    m.apply_kv_cache_events()
    assert gate._restore_tokens(r) == 0
    assert gate.admit([r], 16384) == [r]


@pytest.mark.parametrize('inflight', [False, True])
@pytest.mark.parametrize('matching_tokens', [16, 64])
def test_infercept_return_transfers_ownership_before_cancel(inflight, matching_tokens):
    from serving.tests.test_request_cache_recovery import scheduler
    ad, mem, old = setup_copy(scheduling='fcfs')
    b = batch()
    if inflight:
        mem.host_swap.plan(b)
    ad.on_turn_routed(row('s0', 1, 3_000_000_000, 101))
    successor = finished_request(101, old.input_hash_ids[:matching_tokens] + [1000] * 16)
    successor.arrival = 3_000_000_000
    successor.session_id = 's0'
    successor.num_computed_tokens = 0
    ad.bind_infercept_continuation(successor, mem)
    assert successor.queue_arrival == 0
    assert successor.arrival == 3_000_000_000
    assert successor._prefix_locked
    if inflight:
        mem.host_swap.complete(b)
    mem.evict_prefix_cache(mem.get_kv(64), Device.NPU)
    assert mem.npu_prefix_cache.protected_size() == matching_tokens
    assert mem.npu_prefix_cache.peek_prefix_length(old.input_hash_ids) == matching_tokens
    s = scheduler()
    s.memory = mem
    s._drop_prefill(successor)
    assert mem.npu_prefix_cache.protected_size() == matching_tokens
    s._preempt_recompute(successor)
    assert mem.npu_prefix_cache.protected_size() == 0
    assert successor.queue_arrival == 0


def test_infercept_swapped_return_owns_host_until_restore_completion():
    ad, mem, old = setup_copy(scheduling='fcfs')
    b = batch()
    mem.host_swap.plan(b)
    mem.host_swap.complete(b)
    ad.on_turn_routed(row('s0', 1, 3_000_000_000, 101))
    successor = finished_request(101, old.input_hash_ids + [1000] * 16)
    successor.session_id = 's0'
    successor.num_computed_tokens = 0
    ad.bind_infercept_continuation(successor, mem)
    assert successor.npu_cache_hit == 0
    assert successor.storage_cache_hit == 64
    mem.second_tier_prefix_cache.evict(64)
    assert mem.second_tier_prefix_cache.protected_size() == 64
    successor.storage_restored = True
    successor.num_computed_tokens = 80
    mem.cache_unfinished_req(successor, Device.NPU)
    assert mem.second_tier_prefix_cache.protected_size() == 0
    assert successor.infercept_cpu_node is None
    mem.cache_finished_req(successor, Device.NPU)
    assert mem.npu_prefix_cache.protected_size() == 0


@pytest.mark.parametrize('now,expected', [(99, 'new'), (100, 'old')])
def test_infercept_session_order_does_not_backdate_release(now, expected):
    from serving.tests.test_request_cache_recovery import scheduler
    ad = adapter(scheduling='fcfs')
    s = scheduler()
    s.policy_hooks = ad
    ad.configure_host_swap(s.memory, 25)
    ad._infercept_arrivals.update(old=0, new=50)
    s.max_num_seqs = 1
    s.add_request([1, MODEL, 16, 17, 100, 0, list(range(16)), [100]], session_id='old')
    s.add_request([2, MODEL, 16, 17, 50, 0, list(range(100, 116)), [200]], session_id='new')
    assert s.request[0].session_id == 'new'  # Readiness list stays release-ordered.
    b = s.schedule(now, 0)
    assert [r.session_id for r in b.requests] == [expected]
    assert b.requests[0].arrival == (100 if expected == 'old' else 50)


def test_paper_shared_budget_and_partial_waiting_order(monkeypatch, tmp_path):
    from serving.tests.test_request_cache_recovery import scheduler
    monkeypatch.setenv('INFERCEPT_PAPER_SCHEDULING', '1')
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(dict(PROFILE, S=16)))
    ad = UnifiedPolicyAdapter('min-waste', 'fcfs', None, 1, BLOCK,
        min_waste_profile=str(profile), min_waste_swap=True, min_waste_fcfs_restore=True)
    s = scheduler()
    s.policy_hooks = ad
    s.max_num_batched_tokens = 64
    ad.configure_host_swap(s.memory, 25)
    ad._infercept_arrivals.update(old=0, young=10)
    s.add_request([1, MODEL, 40, 43, 0, 0, list(range(40)), [200,201,202]], session_id='young')
    young = s.request[0]
    # A resident partial prefill has progress but must not bypass older waiters.
    young.num_computed_tokens = 16
    young.admit_seq = 0
    s.memory.cache_unfinished_req(young, Device.NPU)
    s.add_request([2, MODEL, 40, 43, 1, 0, list(range(100,140)), [300,301,302]], session_id='old')
    b = s.schedule(1, 0)
    assert sum(b.scheduled_tokens.values()) == 16
    assert [r.session_id for r in b.requests] == ['old']
    assert young.num_computed_tokens == 16
    assert young.n_preempted == 0
    assert s.max_num_batched_tokens == 64


def test_infercept_idle_head_reclaims_younger_waiting_owner():
    from serving.tests.test_request_cache_recovery import scheduler, memory as bounded_memory, request
    ad = adapter(scheduling='fcfs')
    s = scheduler()
    s.memory = bounded_memory(64)
    s.policy_hooks = ad
    ad.configure_host_swap(s.memory, 25)
    for rid, start, length in [(90, 0, 16), (91, 100, 48)]:
        s.memory.cache_finished_req(request(rid, start, length), Device.NPU)
    ad._infercept_arrivals.update(old=0, new=1)
    ad._swap_programs.update(old='old:0', new='new:0')
    s.add_request([1, MODEL, 48, 49, 100, 0, list(range(48)), [200]], session_id='old')
    s.add_request([2, MODEL, 64, 65, 90, 0, list(range(100, 164)), [300]], session_id='new')
    assert s.memory.npu_prefix_cache.protected_size() == 64
    younger = next(r for r in s.request if r.session_id == 'new')
    s.max_num_seqs = 1
    b = s.schedule(100, 0)
    assert b is not None
    assert [r.session_id for r in b.requests] == ['old']
    assert younger.n_preempted == 1
    assert not younger._prefix_locked
    assert younger.queue_arrival == 1


def test_infercept_closed_loop_continuations_complete_under_pressure(tmp_path):
    from serving.tests.test_request_cache_recovery import scheduler, memory as bounded_memory
    from serving.core.router import Router
    ad = adapter(scheduling='fcfs')
    s = scheduler()
    s.memory = bounded_memory(128)
    s.policy_hooks = ad
    s.max_num_seqs = 2
    s.max_num_batched_tokens = 16
    ad.configure_host_swap(s.memory, 25)
    router = Router(1, [s], 0, policy_adapter=ad)
    rows = []
    for p in range(3):
        context = list(range(1000 * p, 1000 * p + 32))
        turns = []
        for turn in range(3):
            output = [10000 + p * 100 + turn * 2, 10001 + p * 100 + turn * 2]
            turns.append(dict(input_toks=len(context), input_tok_ids=context[:],
                              output_toks=2, output_tok_ids=output, tool_duration_ns=1_000_000))
            context += output + [20000 + turn] * 14
        rows.append(dict(session_id=str(p), arrival_time_ns=p * 1_000_000,
                         sub_requests=turns))
    path = tmp_path / 'sessions.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    router.load_requests(str(path), enable_prefix_caching=True)
    now = 0
    for _ in range(300):
        router.route_arrived_requests(now)
        b = s.schedule(now, 0)
        now += 1_000_000
        if b is not None:
            _, _, finished = s.add_done(b.batch_id + 1, 0, now)
            for r in finished:
                router.notify_request_completed(r.id, now, req_obj=r, memory=s.memory)
        if len(s.done) == 9:
            break
    assert len(s.done) == 9, s._none_reason
    assert len(router._workflow_metrics) == 3
    assert all(r.arrival >= r.queue_arrival for r in s.done)
    s.memory.host_swap.close()
    assert s.memory.npu_prefix_cache.protected_size() == 0
    assert s.memory.second_tier_prefix_cache.protected_size() == 0
    assert s.memory.cpu_swap_reserved == 0


def test_discarded_continuation_recovers_in_profile_chunks_before_new_input():
    from serving.tests.test_request_cache_recovery import scheduler
    ad, mem, old = setup_copy(tokens=1536, scheduling='fcfs')
    # Drop the intercepted GPU context; no host chunks remain to restore.
    assert mem.host_swap.reconsider(10**15, []) == 1
    mem.npu_mem = mem.weight + mem.get_kv(4096)
    mem.npu_prefix_cache.capacity = mem.get_kv(4096)
    successor = finished_request(101, old.input_hash_ids + [9000] * 512)
    successor.session_id = 's0'
    successor.num_computed_tokens = 0
    ad.bind_infercept_continuation(successor, mem)
    assert successor.infercept_recompute_end == 1536
    assert successor.infercept_recompute_chunk == PROFILE['S']
    s = scheduler()
    s.memory = mem
    s.max_num_batched_tokens = 4096
    s.request.append(successor)
    chunks = []
    for i in range(5):
        b = s.schedule(i * 100, 0, batch_id=i)
        assert b is not None
        chunks.append(b.scheduled_tokens[successor.id])
        s.add_done(i + 1, 0, i * 100 + 50)
    assert chunks == [384, 384, 384, 384, 512]
    assert successor.infercept_recompute_end == 0


def test_preserved_continuation_does_not_cap_new_input():
    ad, mem, old = setup_copy(tokens=1536, scheduling='fcfs')
    successor = finished_request(101, old.input_hash_ids + [9000] * 512)
    successor.session_id = 's0'
    successor.num_computed_tokens = 0
    ad.bind_infercept_continuation(successor, mem)
    assert successor.infercept_recompute_end == 0
