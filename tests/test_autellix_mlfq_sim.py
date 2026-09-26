"""Autellix Algorithm 1 on the simulator, not the attained-service proxy.

`--scheduling autellix-mlfq` drives policies.autellix_runtime.AutellixRuntime
-- the same object the engine adapter runs -- from simulator batch events.
These check the parts the simulator half is responsible for: registering
calls, the cumulative fit callback, the batch lifecycle that accrues service
and demotes on quantum exhaustion, starvation promotion, overprovisioning,
and that the planner's order reaches the waiting queue.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.autellix_driver import AutellixDriver, queue_config
from serving.core.memory_model import MemoryModel
from serving.core.request import Request
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
import policies

MODEL = 'meta-llama/Llama-3.1-8B'
BLOCK = 16
# Two boundaries -> three queues. Short quanta so a single batch demotes.
CFG = dict(service_boundaries_s=[1.0, 4.0], quanta_s=[0.5, 1.0, 2.0],
           starvation_ratio=2.0)


def cfg():
    return queue_config(CFG['service_boundaries_s'], CFG['quanta_s'],
                        CFG['starvation_ratio'])


def req(rid, program, tokens=64, computed=0):
    r = Request(rid, MODEL, tokens, tokens + 8, 0, 0,
                input_hash_ids=list(range(rid * 1000, rid * 1000 + tokens)),
                output_hash_ids=[])
    r.num_computed_tokens = computed
    r.session_id = program
    return r


def pcb_of(r):
    return None if r.session_id is None else types.SimpleNamespace(program_id=r.session_id)


def always_fits(req, selected):
    return True


def test_queue_config_refuses_to_invent_parameters():
    for args in (([], [1.0], 2.0), ([1.0], [], 2.0), ([1.0], [0.5, 1.0], None)):
        try:
            queue_config(*args)
        except ValueError as e:
            assert "--autellix-service-boundaries" in str(e)
        else:
            raise AssertionError(f"accepted {args}")
    # One quantum per interval, enforced by the runtime's own config.
    try:
        queue_config([1.0, 4.0], [0.5, 1.0], 2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a short quanta list")


def test_plan_registers_calls_and_returns_queue_order():
    d = AutellixDriver(cfg())
    a, b = req(1, "pA"), req(2, "pB")
    out = d.plan([a, b], [], pcb_of, 0.0, always_fits)
    assert [r.id for r in out] == [1, 2]
    assert set(d.known) == {"1", "2"}
    assert all(d.runtime.calls[r].queue == 0 for r in ("1", "2"))


def test_flat_requests_bypass_the_planner_and_keep_engine_order():
    d = AutellixDriver(cfg())
    prog, flat = req(1, "pA"), req(2, None)
    out = d.plan([prog, flat], [], pcb_of, 0.0, always_fits)
    assert [r.id for r in out] == [1, 2]
    assert set(d.known) == {"1"}          # the flat call was never admitted


def test_quantum_exhaustion_demotes_and_reorders():
    d = AutellixDriver(cfg())
    a, b = req(1, "pA"), req(2, "pB")
    d.plan([a, b], [], pcb_of, 0.0, always_fits)
    d.batch_started([a, b], 0.0)
    # 0.6 s of service exceeds queue 0's 0.5 s quantum: both demote to queue 1.
    d.batch_finished([], 0.6, 0.6)
    assert d.runtime.calls["1"].queue == 1 and d.runtime.calls["2"].queue == 1
    assert d.stats["demoted"] == 2        # counted where Algorithm 1 demotes
    # A newly arrived call enters queue 0 and therefore plans ahead of them.
    c = req(3, "pC")
    out = d.plan([a, b, c], [], pcb_of, 0.7, always_fits)
    assert [r.id for r in out] == [3, 1, 2]
    assert d.stats["demoted"] == 2


def test_service_is_inherited_across_a_program_s_turns():
    d = AutellixDriver(cfg())
    first = req(1, "pA")
    d.plan([first], [], pcb_of, 0.0, always_fits)
    d.batch_started([first], 0.0)
    d.batch_finished([1], 1.5, 1.5)       # finished: 1.5 s of service recorded
    assert d.runtime.processes["pA"].service_s >= 1.5
    # The program's next turn starts in queue 1, not queue 0: 1.5 s is past
    # the first boundary. A fresh program still starts at queue 0.
    nxt, other = req(2, "pA"), req(3, "pB")
    d.plan([nxt, other], [], pcb_of, 1.6, always_fits)
    assert d.runtime.calls["2"].queue == 1
    assert d.runtime.calls["3"].queue == 0


def test_starvation_promotes_a_waiting_call_back_to_queue_zero():
    d = AutellixDriver(cfg())
    a, b = req(1, "pA"), req(2, "pB")
    d.plan([a, b], [], pcb_of, 0.0, always_fits)
    d.batch_started([a, b], 0.0)
    d.batch_finished([], 0.6, 0.6)        # both demoted to queue 1
    assert d.runtime.calls["1"].queue == 1
    # Only 'a' runs from here; 'b' waits until wait >= 2x service.
    for t in (1.0, 2.0, 3.0, 4.0):
        d.plan([b], [a], pcb_of, t, lambda r, s: r.id == 1)
        d.batch_started([a], t)
        d.batch_finished([], t + 0.1, 0.1)
    d.plan([b], [a], pcb_of, 20.0, always_fits)
    assert d.runtime.calls["2"].queue == 0
    assert d.stats["promoted"] >= 1


def test_fit_callback_stops_the_plan_and_overprovisions():
    d = AutellixDriver(cfg(), overprovision=1)
    reqs = [req(i, f"p{i}") for i in (1, 2, 3, 4)]
    # Only two calls fit at a time.
    out = d.plan(reqs, [], pcb_of, 0.0, lambda r, selected: len(selected) < 2)
    assert [r.id for r in out] == [1, 2, 3]
    assert d.stats["overprovisioned"] == 1


def test_resident_dropped_by_the_plan_is_offered_as_a_victim():
    d = AutellixDriver(cfg())
    a, b = req(1, "pA"), req(2, "pB")
    d.plan([a, b], [], pcb_of, 0.0, always_fits)
    d.batch_started([a, b], 0.0)
    d.batch_finished([], 0.6, 0.6)
    c = req(3, "pC")
    # Room for one: the queue-0 newcomer wins and the residents are dropped.
    d.plan([c], [a, b], pcb_of, 0.7, lambda r, selected: not selected)
    assert {r.id for r in d.last_preempt} == {1, 2}


def test_batch_membership_overrides_the_plan():
    """start_batch is told what actually ran, not what was planned."""
    d = AutellixDriver(cfg())
    a, b = req(1, "pA"), req(2, "pB")
    d.plan([a, b], [], pcb_of, 0.0, always_fits)
    d.batch_started([a], 0.0)             # only 'a' fit in the end
    d.batch_finished([], 0.6, 0.6)
    assert d.runtime.calls["1"].queue == 1    # 'a' served, demoted
    assert d.runtime.calls["2"].queue == 0    # 'b' untouched


def test_dropped_call_preserves_its_consumed_service():
    d = AutellixDriver(cfg())
    a = req(1, "pA")
    d.plan([a], [], pcb_of, 0.0, always_fits)
    d.batch_started([a], 0.0)
    d.batch_finished([], 0.3, 0.3)
    d.drop(1, 0.4)
    assert "1" not in d.known
    assert d.runtime.processes["pA"].service_s >= 0.3


# ---------------------------------------------------------------- wiring
def test_adapter_selects_the_planner_and_turns_on_the_hooks():
    ad = UnifiedPolicyAdapter(
        retention_value="cache-lru", scheduling_value="autellix-mlfq",
        routing_value="session-affinity", num_instances=1, block_size=BLOCK,
        autellix_queues=(CFG['service_boundaries_s'], CFG['quanta_s'],
                         CFG['starvation_ratio']),
        autellix_overprovision=2)
    assert ad.custom_hooks            # filter_waiting must be called
    assert not ad.priority_scheduling  # ordering is the plan's, not a stamp
    mem = types.SimpleNamespace(instance_id=0)
    d = ad._autellix_driver(mem)
    assert d.overprovision == 2
    assert ad._autellix_driver(mem) is d          # one runtime per instance
    assert ad._autellix_driver(types.SimpleNamespace(instance_id=1)) is not d


def test_adapter_requires_the_queue_parameters():
    try:
        UnifiedPolicyAdapter(retention_value="cache-lru",
                             scheduling_value="autellix-mlfq",
                             routing_value="session-affinity",
                             num_instances=1, block_size=BLOCK)
    except ValueError as e:
        assert "--autellix-quanta" in str(e)
    else:
        raise AssertionError("built without queue parameters")


def test_registered_as_a_scheduling_value():
    assert "autellix-mlfq" in policies.choices_for("scheduling")


def test_fit_callback_uses_the_memory_model_and_token_budget():
    """The planner's feasibility test is the scheduler's, read-only."""
    ad = UnifiedPolicyAdapter(
        retention_value="cache-lru", scheduling_value="autellix-mlfq",
        routing_value="session-affinity", num_instances=1, block_size=BLOCK,
        autellix_queues=(CFG['service_boundaries_s'], CFG['quanta_s'],
                         CFG['starvation_ratio']))
    m = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, BLOCK, 16, True, False, None, None)
    m.npu_mem = m.weight + m.get_kv(64)
    m.mem_for_kv = m.get_kv(64)
    m.npu_prefix_cache.capacity = m.mem_for_kv
    sched = types.SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=96,
                                  long_prefill_token_threshold=0)
    fits = ad._autellix_fits(m, sched)
    a, b, c = req(1, "pA", 48), req(2, "pB", 48), req(3, "pC", 48)
    assert fits(a, ()) is True
    assert fits(b, (a,)) is False          # 96 tokens fit, but 64 KV tokens do not
    sched.max_num_seqs = 1
    assert fits(b, (a,)) is False          # and the sequence slot is gone too
    used_before = m.npu_used
    fits(c, (a,))
    assert m.npu_used == used_before       # read-only, as the runtime requires


# ------------------------------------------------- Autellix preemption swap
def test_preempt_swap_policy_never_drops_a_source():
    """Autellix's victim must reach the host or it re-prefills, which is the
    outcome swapping exists to avoid."""
    from serving.core.host_swap import MinWasteSwapPolicy, PreemptSwapPolicy
    from policies.utils.waste_model import WasteProfile
    copy = types.SimpleNamespace(ids=list(range(64)), copied=0, since_ns=0)
    reqs = []
    p = PreemptSwapPolicy()
    assert p.drop(copy, 1e6, reqs) is False       # even after a long wait
    assert p.score(copy, 5.0, reqs)[0] == 5.0     # oldest first
    # InferCept's does drop, which is the difference.
    mw = MinWasteSwapPolicy(WasteProfile(a=0.0279, c=15.4, S=384))
    assert mw.drop(copy, 1e6, reqs) is True


def test_victim_is_copied_to_host_before_the_scheduler_recomputes_it():
    from serving.core.host_swap import PreemptSwapPolicy
    from serving.core.memory_model import Device, MemoryModel
    from policies.utils.waste_model import WasteProfile
    ad = UnifiedPolicyAdapter(
        retention_value="cache-lru", scheduling_value="autellix-mlfq",
        routing_value="session-affinity", num_instances=1, block_size=BLOCK,
        autellix_queues=(CFG['service_boundaries_s'], CFG['quanta_s'],
                         CFG['starvation_ratio']),
        autellix_swap=True)
    m = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, BLOCK, 16, True, False, None, None)
    ad.configure_host_swap(m, 25.0, profile=WasteProfile(a=0.0279, c=15.4, S=384),
                           policy=PreemptSwapPolicy(), flag='--autellix-swap')
    victim = req(7, "pV", 64, computed=64)
    m.cache_unfinished_req(victim, Device.NPU)
    # Unrelated resident context, so locking the victim's chain still leaves
    # the pool something to evict.
    other = req(8, "pO", 512, computed=512)
    m.cache_unfinished_req(other, Device.NPU)
    ad._last_memory = m
    ad._now_ns = 1_000_000_000
    ad._autellix_swap_out(victim, ad._now_ns)
    assert ad.stats["autellix_swapped"] == 1
    assert "7" in m.host_swap.queued           # source owned until it lands
    assert len(m.host_swap.queued["7"].ids) == 64


def test_swap_is_off_unless_asked():
    ad = UnifiedPolicyAdapter(
        retention_value="cache-lru", scheduling_value="autellix-mlfq",
        routing_value="session-affinity", num_instances=1, block_size=BLOCK,
        autellix_queues=(CFG['service_boundaries_s'], CFG['quanta_s'],
                         CFG['starvation_ratio']))
    assert ad.autellix_swap is False
    assert ad._autellix_swap_out(req(1, "p"), 0) is None
    assert ad.stats["autellix_swapped"] == 0


def test_a_prompt_longer_than_the_token_budget_still_fits_by_chunking():
    """Job 42596388: demanding the whole prefill fit made every prompt longer
    than max_num_batched_tokens permanently infeasible, so the planner picked
    nothing and the engine deadlocked on a full-but-evictable pool."""
    from serving.core.memory_model import MemoryModel
    ad = UnifiedPolicyAdapter(
        retention_value="cache-lru", scheduling_value="autellix-mlfq",
        routing_value="session-affinity", num_instances=1, block_size=BLOCK,
        autellix_queues=(CFG['service_boundaries_s'], CFG['quanta_s'],
                         CFG['starvation_ratio']))
    m = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, BLOCK, 16, True, False, None, None)
    sched = types.SimpleNamespace(max_num_seqs=128, max_num_batched_tokens=16384,
                                  long_prefill_token_threshold=0)
    fits = ad._autellix_fits(m, sched)
    long_prompt = req(1, "pA", tokens=16722)
    assert fits(long_prompt, ()) is True
    # The chunk consumes the budget, so a second request gets nothing.
    assert fits(req(2, "pB", tokens=64), (long_prompt,)) is False


def test_the_planner_never_returns_an_empty_batch_for_fittable_work():
    d = AutellixDriver(cfg())
    reqs = [req(i, f"p{i}", tokens=16722) for i in (1, 2)]
    from serving.core.memory_model import MemoryModel
    ad = UnifiedPolicyAdapter(
        retention_value="cache-lru", scheduling_value="autellix-mlfq",
        routing_value="session-affinity", num_instances=1, block_size=BLOCK,
        autellix_queues=(CFG['service_boundaries_s'], CFG['quanta_s'],
                         CFG['starvation_ratio']))
    m = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, BLOCK, 16, True, False, None, None)
    sched = types.SimpleNamespace(max_num_seqs=128, max_num_batched_tokens=16384,
                                  long_prefill_token_threshold=0)
    out = d.plan(reqs, [], pcb_of, 0.0, ad._autellix_fits(m, sched))
    assert [r.id for r in out] == [1]        # one chunked prefill, not nothing


def test_swapping_stops_before_it_locks_the_pool():
    """Job 42610546: a queued copy owns its source until it lands, so an
    unbounded queue walked the pool to fully locked and the engine deadlocked
    with nothing evictable."""
    from serving.core.host_swap import PreemptSwapPolicy
    from serving.core.memory_model import Device, MemoryModel
    from policies.utils.waste_model import WasteProfile
    ad = UnifiedPolicyAdapter(
        retention_value="cache-lru", scheduling_value="autellix-mlfq",
        routing_value="session-affinity", num_instances=1, block_size=BLOCK,
        autellix_queues=(CFG['service_boundaries_s'], CFG['quanta_s'],
                         CFG['starvation_ratio']),
        autellix_swap=True)
    m = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, BLOCK, 16, True, False, None, None)
    ad.configure_host_swap(m, 25.0, profile=WasteProfile(a=0.0279, c=15.4, S=384),
                           policy=PreemptSwapPolicy(), flag='--autellix-swap')
    ad._last_memory = m
    ad._now_ns = 1_000_000_000
    sched = types.SimpleNamespace(max_num_batched_tokens=16384, max_num_seqs=128,
                                  config={'max_position_embeddings': 65536})
    ad.attach_schedulers([sched])
    pool = (m.npu_mem - m.weight) // m._bytes_per_token
    cap = ad._swap_queue_limit(m)
    # Headroom is one request's worst-case footprint, because a chunked
    # prefill accumulates KV across chunks -- not one step's token budget.
    assert pool - cap == 65536 + 128 * 16
    # First victim is small: it is swapped. enqueue returns 0 by design --
    # a queued copy is not a completed one -- so the queue is the evidence.
    small = req(7, "pV", 64, computed=64)
    m.cache_unfinished_req(small, Device.NPU)
    ad._autellix_swap_out(small, ad._now_ns)
    assert ad.stats["autellix_swapped"] == 1
    assert "7" in m.host_swap.queued
    # A victim that would push the queue past a quarter of the pool is not:
    # its tokens would be pinned, unreclaimable, until the copy lands.
    big = req(8, "pB", 64, computed=cap)
    assert ad._autellix_swap_out(big, ad._now_ns) == 0
    assert ad.stats["autellix_swap_declined"] == 1
    assert ad.stats["autellix_swapped"] == 1
    assert "8" not in m.host_swap.queued

    # A mostly copied source still owns its entire chain. The remaining
    # bytes alone would pass this cap and permit a second locked chain.
    copy = m.host_swap.queued['7']
    copy.copied = 48
    ad._swap_queue_limit = lambda memory: 80
    second = req(9, 'pC', 32, computed=32)
    m.cache_unfinished_req(second, Device.NPU)
    ad._autellix_swap_out(second, ad._now_ns)
    assert '9' not in m.host_swap.queued
    assert ad.stats['autellix_swap_declined'] == 2


def test_scheduler_enforces_quantum_yield_with_spare_kv_capacity():
    from serving.core.scheduler import Scheduler
    ad = UnifiedPolicyAdapter(
        retention_value='cache-lru', scheduling_value='autellix-mlfq',
        routing_value='session-affinity', num_instances=1, block_size=BLOCK,
        autellix_queues=([1., 4.], [.5, 1., 2.], 1000))
    ad._pcb_of_req = pcb_of
    s = Scheduler(MODEL, 0, 0, 1, 128, 1, 1, 1, 80, 80, 0,
                  None, 16, BLOCK, 0, False, True, False, None, None, True)
    s.policy_hooks = ad
    a, b = req(1, 'long', 16, computed=16), req(2, 'short', 16)
    a.admit_seq = 0
    s.request = [a, b]
    driver = ad._autellix_driver(s.memory)
    driver.plan([b], [a], pcb_of, 0, lambda r, selected: not selected)
    driver.batch_started([a], 0)
    driver.batch_finished([], .6, .6)
    batch = s.schedule(600_000_000, 0)
    assert [r.id for r in batch.requests] == [b.id]
    assert s.num_preemptions == 1
    assert a.admit_seq is None and a.n_preempted == 1
    assert len(s.preemption_log) == 1
    assert driver._in_flight == ('2',)


def test_a_resident_only_yields_to_a_better_placed_waiter():
    """Job 42631045: preempting every resident the plan dropped livelocked the
    engine at 13,468 preemptions on 50 programs. The simulator's only
    preemption is recomputation, which the runtime says must not substitute
    for swapping, so a yield has to be a real quantum yield: one call stepping
    aside for better-placed queued work, bounded per tick."""
    from serving.core.memory_model import MemoryModel
    ad = UnifiedPolicyAdapter(
        retention_value="cache-lru", scheduling_value="autellix-mlfq",
        routing_value="session-affinity", num_instances=1, block_size=BLOCK,
        autellix_queues=(CFG['service_boundaries_s'], CFG['quanta_s'],
                         CFG['starvation_ratio']))
    m = MemoryModel(MODEL, 0, 0, 1, 1, 80, 80, BLOCK, 16, True, False, None, None)
    driver = ad._autellix_driver(m)
    res_a, res_b = req(1, "pA", computed=32), req(2, "pB", computed=32)
    res_a.admit_seq = res_b.admit_seq = 1
    driver.plan([res_a, res_b], [], pcb_of, 0.0, always_fits)
    driver.batch_started([res_a, res_b], 0.0)
    driver.batch_finished([], 0.6, 0.6)          # both demote to queue 1
    # Nothing is waiting: a dropped resident has nobody to yield to.
    driver.plan([], [res_a, res_b], pcb_of, 0.7, lambda r, s: False)
    sched = types.SimpleNamespace(memory=m, request=[], preemption_log=[],
                                  _admit_counter=0, num_preemptions=0,
                                  _preempt_recompute=lambda r: None)
    assert ad.apply_scheduling_plan(sched, [res_a, res_b], 0) == [res_a, res_b]
    assert sched.num_preemptions == 0

    # A fresh queue-0 waiter appears: exactly one resident yields, not both.
    newcomer = req(3, "pC")
    driver.plan([newcomer], [res_a, res_b], pcb_of, 0.8, lambda r, s: not s)
    sched.request = [newcomer]
    kept = ad.apply_scheduling_plan(sched, [res_a, res_b], 0)
    assert sched.num_preemptions == 1
    assert len(kept) == 1
