"""SagaGateway: observed inputs in, paper decisions out, no trace lookahead."""
import asyncio
from types import SimpleNamespace

import pytest

from bench.core.saga_engine import SagaConfig, SagaGateway
from policies.saga_runtime import SagaPlacement


class FakeCoordinator:
    def __init__(self, steal_result=None):
        self.enqueued, self.taken, self.steal_calls = [], [], []
        self.steal_result = steal_result

    def enqueue(self, session, tag, tokens, ready_at, worker):
        self.enqueued.append((session, tag, tuple(tokens), worker))

    def take(self, session, worker):
        self.taken.append((session, worker))

    async def steal(self, destination, loads, empty, now):
        self.steal_calls.append((destination, tuple(loads), tuple(empty), now))
        return self.steal_result


def make(n=2, stats=None, config=None, coordinator=None, generate=None, tool_ttl=None):
    stats = stats or {}
    calls = []

    async def call(i, method, *args):
        calls.append((i, method, args))

    gateway = SagaGateway(n, config or SagaConfig(), SagaPlacement(), coordinator or FakeCoordinator(),
                          stats=lambda i: stats.get(i, {}), call=call, generate=generate,
                          block_bytes=lambda i: 1024, tool_ttl_s=tool_ttl)
    return gateway, calls


def engine_stats(total=100, free=50, protected=None):
    return {'num_gpu_blocks': total, 'free_queue_blocks': free,
            'policy_observation': {'protected_blocks_by_tag': protected or {},
                                   'cached_blocks_by_program': {
                                       tag.rsplit(':', 1)[0]: count
                                       for tag, count in (protected or {}).items()}}}


def test_routing_follows_the_engine_that_holds_the_session_until_it_is_loaded():
    stats = {0: engine_stats(free=60, protected={'p:0': 3}), 1: engine_stats(free=90)}
    gateway, _ = make(stats=stats)
    gateway.register('p')
    asyncio.run(gateway.acquire('p', 0, 0, [1], now=0.0))
    gateway.release('p', 0, now=0.5, service_s=0.1, prompt_tokens=[1], output_tokens=[2])
    assert gateway.route('p', now=1.0) == 0, 'affinity: p is cached on 0 and 0 is below the limit'
    stats[0] = engine_stats(free=5, protected={'p:0': 3})   # load 0.96 > 0.8
    assert gateway.route('p', now=2.0) == 1, 'over the affinity limit, least loaded wins'
    gateway.register('q')
    assert gateway.route('q', now=2.0) == 1


def test_routing_only_epoch_does_not_steal_publish_or_migrate():
    coordinator = FakeCoordinator()
    gateway, calls = make(config=SagaConfig(routing_only=True), coordinator=coordinator)
    asyncio.run(gateway.epoch(1.0))
    assert coordinator.steal_calls == []
    assert calls == []


def test_unprotected_resident_context_keeps_affinity():
    stats = {0: engine_stats(free=40), 1: engine_stats(free=80)}
    stats[0]['policy_observation']['cached_blocks_by_program'] = {'p': 3}
    gateway, _ = make(stats=stats)
    gateway.placement.home['p'] = 0
    assert gateway.route('p', 1.0) == 0
    stats[0]['policy_observation']['cached_blocks_by_program'] = {}
    assert gateway.route('p', 1.1) == 1


def test_route_refresh_coalesces_concurrent_arrivals_and_publishes_atomically():
    async def run():
        calls = []
        async def call(i, method):
            calls.append(i)
            await asyncio.sleep(0)
            return engine_stats(free=40 + i * 40)
        gateway = SagaGateway(2, SagaConfig(routing_only=True), SagaPlacement(),
                              FakeCoordinator(), call=call)
        await asyncio.gather(gateway.refresh_for_route(1.0),
                             gateway.refresh_for_route(1.0))
        assert calls == [0, 1]
        await gateway.refresh_for_route(1.05)
        assert calls == [0, 1]
        await gateway.refresh_for_route(1.11)
        assert calls == [0, 1, 0, 1]
        assert set(gateway.latest_stats) == {0, 1}
    asyncio.run(run())


def test_wa_lru_order_is_published_per_engine_from_observed_cache_state():
    stats = {0: engine_stats(protected={'old:0': 2, 'fresh:0': 2, 'done:0': 8})}
    gateway, calls = make(n=1, stats=stats)
    for pid in ('old', 'fresh', 'done'):
        gateway.register(pid)
        gateway._admit(gateway.programs[pid], 0)
    gateway.release('old', 0, now=1.0, service_s=0.5, prompt_tokens=[1] * 20, output_tokens=[2] * 4, tool='bash')
    gateway.release('fresh', 0, now=9.0, service_s=0.5, prompt_tokens=[1] * 20, output_tokens=[2] * 4, tool='bash')
    gateway.release('done', 0, now=9.5, service_s=0.5, prompt_tokens=[1] * 20, output_tokens=[2] * 4, last_turn=True)
    asyncio.run(gateway.publish_orders(now=10.0))
    assert len(calls) == 1 and calls[0][1] == 'kv_reclaim_order'
    order = calls[0][2][0]
    # A finished program has no successor (reuse 0) and the largest footprint:
    # it goes first; the idle one before the recently used one.
    assert order == ['done:0', 'old:0', 'fresh:0']
    assert gateway.stats['orders_published'] == 1


def test_capacity_hold_is_queued_for_stealing_and_admitted_when_room_appears():
    coordinator = FakeCoordinator()
    gateway, _ = make(n=2, stats={0: engine_stats(), 1: engine_stats()},
                      config=SagaConfig(capacity_limit=1), coordinator=coordinator)
    for pid in ('a', 'b'):
        gateway.register(pid)

    async def scenario():
        assert await gateway.acquire('a', 0, 0, [1, 2, 3], now=0.0) == 0
        pending = asyncio.ensure_future(gateway.acquire('b', 0, 0, [4, 5, 6], now=0.1))
        await asyncio.sleep(0)
        assert not pending.done() and coordinator.enqueued == [('b', 'b:0', (4, 5, 6), 0)]
        assert gateway.stats['holds'] == 1
        gateway.release('a', 0, now=1.0, service_s=1.0, prompt_tokens=[1, 2, 3], output_tokens=[9])
        gateway._drain_holds(now=1.0)
        assert await pending == 0 and coordinator.taken == [('b', 0)]
    asyncio.run(scenario())


def test_idle_engine_steals_a_held_call_and_the_call_moves():
    record = SimpleNamespace(session='b')
    coordinator = FakeCoordinator(steal_result=record)
    gateway, _ = make(n=2, stats={0: engine_stats(free=10), 1: engine_stats(free=95)},
                      config=SagaConfig(capacity_limit=1), coordinator=coordinator)
    for pid in ('a', 'b'):
        gateway.register(pid)

    async def scenario():
        await gateway.acquire('a', 0, 0, [1], now=0.0)
        pending = asyncio.ensure_future(gateway.acquire('b', 0, 0, [1, 2], now=0.1))
        await asyncio.sleep(0)
        gateway.empty_since[1] = 0.0
        await gateway.steal(now=1.0)
        assert coordinator.steal_calls and coordinator.steal_calls[0][0] == 1
        assert gateway.stats['steals'] == 1
        gateway._drain_holds(now=1.0)
        assert await pending == 1, 'the stolen call runs on the idle engine'
        assert gateway.inflight == [1, 1]
    asyncio.run(scenario())


def test_afs_shares_bound_a_tenant_in_flight_and_need_an_explicit_slack():
    gateway, _ = make(n=1, stats={0: engine_stats()}, config=SagaConfig(capacity_limit=4))
    with pytest.raises(ValueError, match='overdue_slack_s'):
        gateway.register('x', tenant='t1', deadline_s=5.0)
    gateway, _ = make(n=1, stats={0: engine_stats()},
                      config=SagaConfig(capacity_limit=4, overdue_slack_s=0.5))
    gateway.register('u1', tenant='urgent', deadline_s=1.0)
    gateway.register('r1', tenant='relaxed', deadline_s=100.0)
    gateway.register('r2', tenant='relaxed', deadline_s=100.0)
    gateway.register('r3', tenant='relaxed', deadline_s=100.0)
    gateway.update_shares(now=0.0)
    assert gateway.shares['urgent'] > gateway.shares['relaxed']
    # Three relaxed programs far from their deadline against one urgent program:
    # the relaxed share of the 4 slots rounds up to a single slot.
    import math
    allowed = max(1, math.ceil(gateway.shares['relaxed'] * 4))
    assert allowed == 1

    async def scenario():
        assert await gateway.acquire('r1', 0, 0, [1], now=0.0) == 0
        pending = asyncio.ensure_future(gateway.acquire('r2', 0, 0, [1], now=0.0))
        await asyncio.sleep(0)
        assert not pending.done() and gateway.stats['afs_holds'] == 1
        assert await gateway.acquire('u1', 0, 0, [1], now=0.0) == 0, 'the urgent tenant is not held'
        # The relaxed slot frees: the held relaxed call is admitted next epoch.
        gateway.release('r1', 0, now=1.0, service_s=0.5, prompt_tokens=[1], output_tokens=[2], last_turn=True)
        gateway.update_shares(now=1.0)
        gateway._drain_holds(now=1.0)
        assert await pending == 0
    asyncio.run(scenario())


def test_prefetch_recomputes_an_evicted_context_before_the_tool_returns():
    issued = []

    async def generate(instance, program_id, turn, tokens):
        issued.append((instance, program_id, turn, tuple(tokens)))

    stats = {0: engine_stats(protected={})}
    gateway, _ = make(n=1, stats=stats, config=SagaConfig(prefetch=True, prefetch_margin_s=0.5),
                      generate=generate, tool_ttl=lambda tool: 2.0)
    gateway.register('p')
    gateway._admit(gateway.programs['p'], 0)
    gateway.release('p', 0, now=0.0, service_s=0.3, prompt_tokens=[1, 2], output_tokens=[3], tool='bash')
    asyncio.run(gateway.prefetch(now=1.0))
    assert not issued, 'too early: the tool is expected at 2.0 s'
    stats[0] = engine_stats(protected={'p:0': 2})
    asyncio.run(gateway.prefetch(now=1.6))
    assert not issued, 'still resident: nothing to prefetch'
    stats[0] = engine_stats(protected={})
    asyncio.run(gateway.prefetch(now=1.6))
    assert issued == [(0, 'p', 1, (1, 2, 3))]
    asyncio.run(gateway.prefetch(now=1.7))
    assert len(issued) == 1, 'one prefetch per gap'


def test_a_failed_epoch_stops_held_calls_with_its_cause():
    class Failing(FakeCoordinator):
        async def steal(self, destination, loads, empty, now):
            raise RuntimeError('policy engine is not enabled')

    gateway, _ = make(n=2, stats={0: engine_stats(free=10), 1: engine_stats(free=95)},
                      config=SagaConfig(capacity_limit=1, epoch_s=0.01, idle_s=0.001),
                      coordinator=Failing())
    for pid in ('a', 'b'):
        gateway.register(pid)

    async def scenario():
        await gateway.acquire('a', 0, 0, [1], now=0.0)
        pending = asyncio.ensure_future(gateway.acquire('b', 0, 0, [1, 2], now=0.1))
        await asyncio.sleep(0)
        gateway.empty_since[1] = 0.0
        gateway.start(clock=lambda: 5.0)
        with pytest.raises(RuntimeError, match='failed epoch'):
            await asyncio.wait_for(pending, timeout=2)
        assert isinstance(gateway.failure, RuntimeError)
        await gateway.stop()
        with pytest.raises(RuntimeError, match='failed epoch'):
            await gateway.acquire('a', 1, 0, [1], now=6.0)
    asyncio.run(scenario())
