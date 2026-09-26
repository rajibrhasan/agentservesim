"""Capacity failures must preserve the cache events needed for recovery."""
import unittest
from unittest.mock import patch

from serving.core import memory_model
from serving.core.memory_model import Device, KVCapacityError, MemoryModel
from serving.core.request import Request
from serving.core.scheduler import Scheduler

MODEL = 'meta-llama/Llama-3.1-8B'


def memory(tokens):
    m = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, 16, 16,
                    True, False, None, None)
    m.npu_mem = m.weight + m.get_kv(tokens)
    m.mem_for_kv = m.get_kv(tokens)
    m.npu_prefix_cache.capacity = m.mem_for_kv
    return m


def request(id, start=0, length=32):
    r = Request(id, MODEL, length, length + 8, 0, 0,
                list(range(start, start + length)), list(range(1000, 1008)))
    r.num_computed_tokens = length
    return r


def scheduler():
    return Scheduler(MODEL, 0, 0, 128, 64, 1, 1, 1, 80, 80, 0,
                     None, 16, 16, 0, False, True, False, None, None, True)


class CacheRecoveryTests(unittest.TestCase):
    def protect(self, m, r):
        from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
        adapter = UnifiedPolicyAdapter('continuum', 'fcfs', 'session-affinity', 1, 16)
        m.kv_protection = adapter
        node = r.npu_last_node
        m.unlock_prefix(r, Device.NPU)
        adapter._ctx = (m, node, r.num_computed_tokens)
        adapter.protect('pin', 100.0)
        return adapter, node

    def test_publication_rejects_before_inserting_and_keeps_reservation(self):
        for finished in (False, True):
            with self.subTest(finished=finished):
                m = memory(16)
                r = request(1)
                r.kv_reserved = m.get_kv(16)
                m.npu_reserved = r.kv_reserved
                publish = m.cache_finished_req if finished else m.cache_unfinished_req
                with self.assertRaises(KVCapacityError):
                    publish(r, Device.NPU)
                self.assertEqual(m.npu_prefix_cache.total_size(), 0)
                self.assertEqual(m.npu_reserved, m.get_kv(16))
                self.assertFalse(m.npu_prefix_cache.kv_event_queue)
                self.assert_accounted(m)

    def test_bare_publication_breaks_retention_pins_before_insert(self):
        for finished in (False, True):
            with self.subTest(finished=finished):
                m = memory(64)
                old = request(1, 0, 32)
                m.cache_unfinished_req(old, Device.NPU)
                adapter, _ = self.protect(m, old)
                incoming = request(2, 100, 48)
                publish = m.cache_finished_req if finished else m.cache_unfinished_req
                with patch.object(memory_model, '_KV_INVARIANT', True):
                    publish(incoming, Device.NPU)
                self.assertGreater(adapter.stats['reclaimed_forced'], 0)
                self.assert_accounted(m)

    def test_decode_publication_uses_the_same_pressure_boundary(self):
        m = memory(48)
        parked = request(1)
        m.cache_unfinished_req(parked, Device.NPU)
        adapter, _ = self.protect(m, parked)
        decode = Request(2, MODEL, 16, 49, 0, 0,
                         list(range(100, 116)), list(range(1000, 1033)))
        decode.num_computed_tokens = 16
        m.cache_unfinished_req(decode, Device.NPU)
        decode.num_computed_tokens = 32
        self.assertFalse(decode.is_prefill())
        with patch.object(memory_model, '_KV_INVARIANT', True):
            # The formerly bare decode call, with no scheduler recovery wrapper.
            m.cache_unfinished_req(decode, Device.NPU)
        self.assertGreater(adapter.stats['reclaimed_forced'], 0)
        self.assert_accounted(m)

    def test_pin_shared_with_running_request_is_not_free_capacity(self):
        m = memory(32)
        old = request(1)
        m.cache_unfinished_req(old, Device.NPU)
        adapter, node = self.protect(m, old)
        # Another live request holds the same prefix. Breaking its retention
        # pin cannot make the running request's blocks allocatable.
        m.npu_prefix_cache.inc_lock_ref(node)
        new = request(2, 100, 16)
        new.num_computed_tokens = 0
        with self.assertRaises(KVCapacityError):
            m.reserve_kv([new], {new.id: 16})
        self.assertEqual(m.npu_reserved, 0)
        self.assertEqual(new.kv_reserved, 0)
        self.assertEqual(m.npu_prefix_cache.total_size(), 32)
        self.assert_accounted(m)

    def test_admission_measures_space_after_breaking_a_shared_pin(self):
        s = scheduler()
        m = s.memory = memory(64)
        running = request(1)
        running.admit_seq = 1
        m.cache_unfinished_req(running, Device.NPU)
        _, node = self.protect(m, running)
        m.npu_prefix_cache.inc_lock_ref(node)
        running.npu_last_node = node
        running._prefix_locked = True
        waiting = request(2, 100)
        waiting.num_computed_tokens = 0
        s.request.extend([running, waiting])
        with patch.object(memory_model, '_KV_INVARIANT', True):
            batch = s.schedule(0, 0)
        # There are 32 free tokens. The decode reserves 16, so a new 32-token
        # prefill cannot join it by counting the running prefix's pin again.
        self.assertEqual([r.id for r in batch.requests], [running.id])
        self.assertLessEqual(m.npu_used + m.npu_reserved, m.npu_mem)

    def assert_accounted(self, m):
        self.assertEqual(m.npu_used - m.weight,
                         m.npu_prefix_cache.total_size() * m._bytes_per_token)
        self.assertLessEqual(m.npu_used, m.npu_mem)

    def test_failed_insert_is_not_silently_forgotten_on_retry(self):
        m = memory(16)
        r = request(1)
        # Inject a low-level publication to exercise the defensive event
        # transaction independently of the normal pre-insertion check.
        r.npu_last_node = m.npu_prefix_cache.cache_unfinished_req(r)
        m.npu_prefix_cache.inc_lock_ref(r.npu_last_node)
        r._prefix_locked = True
        with self.assertRaises(KVCapacityError):
            m.apply_kv_cache_events()
        self.assertEqual(m.npu_used, m.weight)
        self.assertTrue(m.npu_prefix_cache.kv_event_queue)
        self.assertFalse(m._npu_cache_hashtolen)
        with self.assertRaises(KVCapacityError):
            m.apply_kv_cache_events()
        # Unlocking the failed publication allows eviction and recovery.
        m.unlock_prefix(r, Device.NPU)
        with patch.object(memory_model, '_KV_INVARIANT', True):
            m.apply_kv_cache_events()
        self.assert_accounted(m)
        self.assertFalse(m.npu_prefix_cache.kv_event_queue)
        m.evict_prefix_cache(m.get_kv(32), Device.NPU)
        self.assertEqual(m.npu_used, m.weight)

    def test_partial_reclaim_then_failure_can_be_retried(self):
        m = memory(32)
        old = request(1, 0, 16)
        m.cache_unfinished_req(old, Device.NPU)
        m.unlock_prefix(old, Device.NPU)
        large = request(2, 100, 48)
        large.npu_last_node = m.npu_prefix_cache.cache_unfinished_req(large)
        m.npu_prefix_cache.inc_lock_ref(large.npu_last_node)
        large._prefix_locked = True
        with self.assertRaises(KVCapacityError):
            m.apply_kv_cache_events()
        # The old entry was removed from the tree, but its removal and the
        # failed insertion are both pending; neither was charged halfway.
        self.assertEqual(m.npu_used - m.weight, m.get_kv(16))
        m.unlock_prefix(large, Device.NPU)
        with patch.object(memory_model, '_KV_INVARIANT', True):
            m.apply_kv_cache_events()
        self.assert_accounted(m)
        m.evict_prefix_cache(m.get_kv(64), Device.NPU)
        self.assertEqual(m.npu_used, m.weight)
        self.assertFalse(m._npu_cache_hashtolen)

    def test_failed_publication_charges_once_when_capacity_becomes_available(self):
        m = memory(16)
        r = request(1)
        with self.assertRaises(KVCapacityError):
            m.cache_unfinished_req(r, Device.NPU)
        m.npu_mem += m.get_kv(16)
        with patch.object(memory_model, '_KV_INVARIANT', True):
            m.cache_unfinished_req(r, Device.NPU)
            m.apply_kv_cache_events()
        self.assert_accounted(m)
        self.assertEqual(m.npu_used - m.weight, m.get_kv(32))

    def test_actual_scheduler_recovers_by_preempting_and_reconciling(self):
        from types import SimpleNamespace
        s = scheduler()
        s.memory = memory(48)
        s.scheduling_policy = 'priority'
        high, low = request(1, 0, 16), request(2, 100, 16)
        completing = request(3, 200, 32)
        for seq, r in enumerate((low, high, completing), start=1):
            r.admit_seq = seq
        high.priority, low.priority = 0, 10
        for r in (high, low):
            s.memory.cache_unfinished_req(r, Device.NPU)
        with patch.object(memory_model, '_KV_INVARIANT', True):
            s._cache_unfinished_checked(
                completing, SimpleNamespace(requests=[high, low, completing]))
        self.assertIsNone(low.admit_seq)
        self.assertIsNotNone(high.admit_seq)
        self.assertEqual(s.num_preemptions, 1)
        self.assert_accounted(s.memory)
        self.assertFalse(s.memory.npu_prefix_cache.kv_event_queue)


class PriorityRecoveryTests(unittest.TestCase):
    def test_held_retention_pins_are_reclaimed_under_sustained_pressure(self):
        from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
        s = scheduler()
        s.max_num_batched_tokens = 32
        m = s.memory
        m.mem_for_kv = m.get_kv(128)
        m.npu_mem = m.weight + m.mem_for_kv
        m.npu_prefix_cache.capacity = m.mem_for_kv
        adapter = UnifiedPolicyAdapter('continuum', 'fcfs', 'session-affinity', 1, 16)
        m.kv_protection = adapter
        for i in range(12):
            s.request.append(Request(
                i, MODEL, 32, 49, 0, 0, list(range(i * 100, i * 100 + 32)),
                list(range(10000 + i * 100, 10017 + i * 100))))
        generated = 0
        with patch.object(memory_model, '_KV_INVARIANT', True):
            for now in range(2000):
                adapter._now_ns = now
                batch = s.schedule(now, 0)
                if batch:
                    _, n, ended = s.add_done(batch.batch_id + 1, 0, now + 1)
                    generated += n
                    for r in ended:
                        node = m.npu_prefix_cache.match_prefix(
                            (r.input_hash_ids + r.output_hash_ids)[:48]).last_device_node
                        adapter._ctx = (m, node, 48)
                        # Keep the pin unexpired for this whole reproduction.
                        adapter.protect(str(r.id), 100.0)
                if len(s.done) == 12:
                    break
        self.assertEqual(len(s.done), 12)
        self.assertEqual(generated, 12 * 17)
        self.assertEqual(sum(len(r.itl) for r in s.done), 12 * 16)
        self.assertGreater(adapter.stats['reclaimed_forced'], 0)
        self.assertEqual(m.npu_reserved, 0)

    def test_pressure_workloads_complete_with_accounting_invariant(self):
        for cap in (64, 96, 128):
            for policy in ('fcfs', 'priority'):
                with self.subTest(capacity=cap, policy=policy):
                    s = scheduler()
                    s.scheduling_policy = policy
                    s.max_num_batched_tokens = 32
                    m = s.memory
                    m.mem_for_kv = m.get_kv(cap)
                    m.npu_mem = m.weight + m.mem_for_kv
                    m.npu_prefix_cache.capacity = m.mem_for_kv
                    for i in range(6):
                        r = Request(i, MODEL, 32, 49, 0, 0,
                                    list(range(i * 100, i * 100 + 32)),
                                    list(range(10000 + i * 100, 10017 + i * 100)))
                        r.priority = 6 - i
                        s.request.append(r)
                    counted = 0
                    with patch.object(memory_model, '_KV_INVARIANT', True):
                        for now in range(1000):
                            b = s.schedule(now, 0)
                            if b:
                                _, generated, _ = s.add_done(b.batch_id + 1, 0, now + 1)
                                counted += generated
                            if len(s.done) == 6:
                                break
                    self.assertEqual(len(s.done), 6)
                    self.assertEqual(counted, 6 * 17)
                    self.assertEqual(sum(len(r.itl) for r in s.done), 6 * 16)
                    self.assertEqual(m.npu_reserved, 0)
                    self.assertEqual(m.npu_prefix_cache.protected_size(), 0)

    def test_priority_requeue_does_not_override_priority(self):
        for policy, expected in [('priority', 1), ('fcfs', 2)]:
            with self.subTest(policy=policy):
                s = scheduler()
                s.scheduling_policy = policy
                s.max_num_seqs = 1
                high, low = request(1, 0, 16), request(2, 100, 16)
                for r in (high, low):
                    r.num_computed_tokens = 0
                high.priority, low.priority = 0, 10
                low.preempt_seq = 1
                s.request.extend([low, high])
                b = s.schedule(0, 0)
                self.assertIsNotNone(b)
                self.assertEqual([r.id for r in b.requests], [expected])

    def test_recovery_does_not_swallow_invariant_errors(self):
        s = scheduler()
        r = request(1)
        from types import SimpleNamespace
        with patch.object(s.memory, 'cache_unfinished_req',
                          side_effect=RuntimeError('invariant failed')):
            with patch.object(s, '_preempt_recompute') as preempt:
                with self.assertRaisesRegex(RuntimeError, 'invariant failed'):
                    s._cache_unfinished_checked(r, SimpleNamespace(requests=[r]))
                preempt.assert_not_called()

    def test_publication_recovery_uses_priority_victim(self):
        from types import SimpleNamespace
        s = scheduler()
        s.scheduling_policy = 'priority'
        completing, low, newest = request(1), request(2, 100), request(3, 200)
        for seq, r in enumerate((completing, low, newest)):
            r.admit_seq = seq
        low.priority, newest.priority = 10, 0
        with patch.object(s.memory, 'cache_unfinished_req',
                          side_effect=[KVCapacityError('full'), None]):
            with patch.object(s, '_preempt_recompute') as preempt:
                s._cache_unfinished_checked(
                    completing, SimpleNamespace(requests=[completing, low, newest]))
                preempt.assert_called_once_with(low)


if __name__ == '__main__':
    unittest.main()
