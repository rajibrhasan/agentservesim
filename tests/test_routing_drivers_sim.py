"""SAGA placement with stealing, and Autellix prompt-length routing.

The decisions are policies.saga_runtime.SagaPlacement and the paper's
short/long split; these check the observation half the simulator supplies and
that an agreed steal actually moves a queued turn.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.routing_drivers import AutellixRouter, SagaRouter
from serving.core.request import Request
from serving.core.unified_policy_adapter import UnifiedPolicyAdapter
import policies

MODEL = 'meta-llama/Llama-3.1-8B'
NS = 1_000_000_000


def waiting(rid, session, arrival_s=0.0, computed=0, admitted=False):
    r = Request(rid, MODEL, 64, 72, int(arrival_s * NS), 0,
                input_hash_ids=list(range(rid * 100, rid * 100 + 64)),
                output_hash_ids=[])
    r.session_id = session
    r.num_computed_tokens = computed
    r.admit_seq = 1 if admitted else None
    return r


class FakeCache:
    def __init__(self, owners=()):
        leaf = types.SimpleNamespace(children={}, owners=set(owners))
        self.root_node = types.SimpleNamespace(children={"a": leaf}, owners=set())


def sched(requests=(), owners=(), inflight=0):
    s = types.SimpleNamespace()
    s.request = list(requests)
    s.inflight = [types.SimpleNamespace(requests=[None]) for _ in range(inflight)]
    s.memory = types.SimpleNamespace(npu_prefix_cache=FakeCache(owners))
    return s


def load_of(value):
    return lambda s: value


# ------------------------------------------------------------------- SAGA
def test_routing_observations_refresh_together_after_100ms():
    r = SagaRouter(2)
    a, b = sched(owners=['p']), sched()
    r.placement.home['p'] = 0
    loads = {id(a): 0.6, id(b): 0.2}
    assert r.route('p', [a, b], 1.0, lambda s: loads[id(s)]) == 0
    a.memory.npu_prefix_cache.root_node.children.clear()
    assert r.route('p', [a, b], 1.05, lambda s: loads[id(s)]) == 0
    assert r.route('p', [a, b], 1.11, lambda s: loads[id(s)]) == 1


def test_saga_retention_uses_the_routing_snapshot():
    a, b = sched(), sched()
    router = SagaRouter(2)
    router.route('p', [a, b], 1.0, lambda s: 0.6 if s is a else 0.2)
    adapter = UnifiedPolicyAdapter.__new__(UnifiedPolicyAdapter)
    adapter.routing_value = 'saga-placement'
    adapter._live_router = router
    adapter._schedulers = [a, b]
    assert adapter._retention_utilization(a.memory) == 0.6
    assert adapter._retention_utilization(b.memory) == 0.2

def test_observation_never_reports_a_queue_as_both_busy_and_idle():
    r = SagaRouter(2)
    busy, idle = sched([waiting(1, "s0", 1.0)]), sched()
    w = r.observe([busy, idle], 5.0, lambda s: 0.5)
    assert w[0].queued_sessions == (("s0", 1.0),) and w[0].empty_since_s is None
    assert w[1].queued_sessions == () and w[1].empty_since_s == 5.0
    # The idle instance keeps its original empty-since across ticks.
    assert r.observe([busy, idle], 9.0, lambda s: 0.5)[1].empty_since_s == 5.0


def test_affinity_holds_a_session_on_the_instance_that_cached_it():
    r = SagaRouter(2)
    a, b = sched(owners=["s0"]), sched()
    first = r.route("s0", [a, b], 1.0, lambda s: 0.1)
    assert r.route("s0", [a, b], 2.0, lambda s: 0.1) == first
    assert r.stats["affinity_hits"] >= 1


def test_affinity_yields_when_the_home_is_over_the_limit():
    r = SagaRouter(2)
    a, b = sched(owners=["s0"]), sched()
    r.placement.home["s0"] = 0
    # Home at 0.95 utilization is past the 0.8 affinity limit.
    assert r.route("s0", [a, b], 1.0, lambda s: 0.95 if s is a else 0.1) == 1


def test_idle_instance_steals_the_oldest_queued_session():
    r = SagaRouter(2)
    busy = sched([waiting(1, "old", 1.0), waiting(2, "new", 4.0)])
    idle = sched()
    moved = {}

    def move(session, source, destination):
        moved["call"] = (session, source, destination)
        return True

    # Idle long enough, and the busy worker is over the 2x load ratio.
    r.observe([busy, idle], 0.0, lambda s: 0.8 if s is busy else 0.1)
    out = r.steal([busy, idle], 5.0, lambda s: 0.8 if s is busy else 0.1, move)
    assert moved["call"] == ("old", 0, 1)     # oldest queued session
    assert len(out) == 1 and r.stats["steals_completed"] == 1


def test_a_steal_that_cannot_be_carried_out_is_reported_back():
    r = SagaRouter(2)
    busy, idle = sched([waiting(1, "old", 1.0)]), sched()
    r.observe([busy, idle], 0.0, lambda s: 0.8 if s is busy else 0.1)
    r.steal([busy, idle], 5.0, lambda s: 0.8 if s is busy else 0.1,
            lambda *a: False)
    assert r.stats["steals_abandoned"] == 1
    # The session is not left permanently un-stealable.
    assert r.placement.pending == {}


def test_balanced_workers_do_not_steal():
    r = SagaRouter(2)
    busy, idle = sched([waiting(1, "s", 1.0)]), sched()
    r.observe([busy, idle], 0.0, lambda s: 0.4)
    assert r.steal([busy, idle], 5.0, lambda s: 0.4, lambda *a: True) == ()


# --------------------------------------------------------------- Autellix
def test_short_prompts_go_to_the_least_loaded_engine():
    r = AutellixRouter(2, long_prompt_tokens=2048)
    a, b = sched(inflight=3), sched()
    assert r.route("p0", 100, [a, b], lambda s: len(s.inflight)) == 1
    assert r.stats["short_routed"] == 1 and r.stats["long_routed"] == 0


def test_a_long_prompt_establishes_and_then_keeps_its_home():
    r = AutellixRouter(2, long_prompt_tokens=2048)
    a, b = sched(inflight=3), sched()
    first = r.route("p0", 9000, [a, b], lambda s: len(s.inflight))
    assert first == 1
    # Even once that engine is the loaded one, the long call goes home.
    a.inflight, b.inflight = [], [None] * 5
    assert r.route("p0", 9000, [a, b], lambda s: len(s.inflight)) == 1
    assert r.stats["home_hits"] == 1


def test_a_short_prompt_does_not_establish_a_home():
    r = AutellixRouter(2, long_prompt_tokens=2048)
    a, b = sched(inflight=3), sched()
    r.route("p0", 100, [a, b], lambda s: len(s.inflight))
    assert "p0" not in r.home


# ----------------------------------------------------------------- wiring
def test_routing_values_are_registered():
    for v in ("saga-placement", "autellix-route"):
        assert v in policies.choices_for("routing")


def test_adapter_moves_a_waiting_turn_but_never_a_started_one():
    ad = UnifiedPolicyAdapter(retention_value="cache-lru", scheduling_value="fcfs",
                              routing_value="saga-placement", num_instances=2,
                              block_size=16)
    started = waiting(1, "busy", 1.0, computed=32, admitted=True)
    queued = waiting(2, "busy", 2.0)
    busy = sched([started, queued])
    idle = sched()
    ad.attach_schedulers([busy, idle])
    ad._live_router.observe([busy, idle], 0.0, lambda s: 0.9 if s is busy else 0.0)
    ad._instance_load = lambda s: 0.9 if s is busy else 0.0
    ad.steal_tick(5 * NS)
    assert started in busy.request          # generating call never moves
    assert queued in idle.request           # the waiting turn did
    assert ad.routing_stats()["steals_completed"] == 1
