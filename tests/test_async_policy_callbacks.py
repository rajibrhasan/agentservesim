import asyncio
from types import SimpleNamespace as NS

import pytest
from bench.core.policy_driver import PolicyDriver, TupleConfig


SNAPSHOT = dict(schema_version=1, event='completion_after_free',
                num_gpu_blocks=101, free_queue_blocks=25)


def test_other_program_progresses_while_protect_waits(monkeypatch):
    monkeypatch.setenv('BENCH_ASYNC_POLICY_RPC', '1')
    monkeypatch.setenv('VLLM_KV_RELEASE_AT_ARRIVAL', '1')
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def utility(method, *args):
            calls.append((method, args))
            if method == 'kv_protect' and args[0] == 'a:0':
                entered.set()
                await release.wait()
            return {'num_gpu_blocks': 101, 'free_queue_blocks': 25} if method == 'kv_protection_stats' else 3
        engine = NS(engine_core=NS(call_utility_async=utility))
        d = PolicyDriver(TupleConfig(retention='search-seed', scheduling='search-seed'),
                         [engine], asyncio.get_running_loop())
        try:
            for p in ('a', 'b'):
                d.programs.on_turn_release(p, 0, 0)
            async def complete(p):
                await d.turn_complete(p, 0, p+':0', 1, 32, 0, 10,
                                      tool_name='tool', kv_snapshot=SNAPSHOT)
            a = asyncio.create_task(complete('a'))
            await asyncio.wait_for(entered.wait(), 2)
            assert not d.programs.get('a').kv_protected
            ready = asyncio.create_task(d.turn_ready('a', 1, 11, 32, .1))
            await asyncio.wait_for(complete('b'), 2)
            assert d.programs.get('b').kv_protected
            assert not ready.done()
            release.set()
            await asyncio.wait_for(a, 2)
            assert await asyncio.wait_for(ready, 2) == (0, d.scheduling_exec.stamps[-1].priority)
            assert d.retention_exec.decisions[0].program_id == 'b'
            a_dec = next(x for x in d.retention_exec.decisions if x.request_id=='a:0' and x.action=='protect')
            assert a_dec.deadline_ts == 12
            assert d.retention_exec.policy.signals.kv_utilization == .75
            assert sum(m=='kv_protection_stats' for m,args in calls) == 1
        finally:
            release.set()
            d._worker.shutdown(wait=True)
    asyncio.run(check())


@pytest.mark.parametrize('outcome', [0, 'error'])
def test_failed_protection_never_publishes_ownership(monkeypatch, outcome):
    monkeypatch.setenv('BENCH_ASYNC_POLICY_RPC', '1')
    monkeypatch.setenv('VLLM_KV_RELEASE_AT_ARRIVAL', '1')
    async def check():
        async def utility(method, *args):
            if outcome == 'error':
                raise RuntimeError('engine rejected RPC')
            return outcome
        d = PolicyDriver(TupleConfig(retention='search-seed', scheduling='search-seed'),
            [NS(engine_core=NS(call_utility_async=utility))], asyncio.get_running_loop())
        try:
            d.programs.on_turn_release('a', 0, 0)
            call = d.turn_complete('a', 0, 'a:0', 1, 32, 0, 10, kv_snapshot=SNAPSHOT)
            if outcome == 'error':
                with pytest.raises(RuntimeError, match='engine rejected'):
                    await call
            else:
                await call
            assert not d.programs.get('a').kv_protected
            assert 'a:0' not in d.kv._where
        finally:
            d._worker.shutdown(wait=True)
    asyncio.run(check())


def test_cancelled_completion_finishes_ack_before_unlock(monkeypatch):
    monkeypatch.setenv('BENCH_ASYNC_POLICY_RPC', '1')
    monkeypatch.setenv('VLLM_KV_RELEASE_AT_ARRIVAL', '1')
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        async def utility(method, *args):
            entered.set()
            await release.wait()
            return 2
        d = PolicyDriver(TupleConfig(retention='search-seed', scheduling='search-seed'),
            [NS(engine_core=NS(call_utility_async=utility))], asyncio.get_running_loop())
        try:
            d.programs.on_turn_release('a', 0, 0)
            task = asyncio.create_task(d.turn_complete('a', 0, 'a:0', 1, 32, 0, 10, kv_snapshot=SNAPSHOT))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            await asyncio.sleep(.01)
            assert not task.done()
            assert d._async_callbacks.locks['a'].locked()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert d.programs.get('a').kv_protected
            assert not d._async_callbacks.locks['a'].locked()
        finally:
            release.set()
            d._worker.shutdown(wait=True)
    asyncio.run(check())


def test_arrivals_query_selected_engine_and_preserve_utilization(monkeypatch):
    monkeypatch.setenv('BENCH_ASYNC_POLICY_RPC', '1')
    monkeypatch.setenv('VLLM_KV_RELEASE_AT_ARRIVAL', '1')
    async def check():
        calls = []
        def engine(index):
            async def utility(method, *args):
                assert method == 'kv_protection_stats'
                calls.append(index)
                return dict(num_gpu_blocks=101, free_queue_blocks=25 if index==0 else 75)
            return NS(engine_core=NS(call_utility_async=utility))
        d = PolicyDriver(TupleConfig(retention='search-seed', scheduling='search-seed'),
                         [engine(0), engine(1)], asyncio.get_running_loop())
        try:
            assert (await d.turn_ready('a', 0, 1, 32))[0] == 0
            assert d.retention_exec.policy.signals.kv_utilization == .75
            assert calls == [0]
            assert (await d.turn_ready('b', 0, 2, 32))[0] == 1
            assert d.retention_exec.policy.signals.kv_utilization == .25
            assert calls == [0, 1]
        finally:
            d._worker.shutdown(wait=True)
    asyncio.run(check())


def test_rejected_pin_cancels_provisional_before_next_turn(monkeypatch):
    monkeypatch.setenv('BENCH_ASYNC_POLICY_RPC', '1')
    monkeypatch.setenv('VLLM_KV_RELEASE_AT_ARRIVAL', '1')
    monkeypatch.setenv('VLLM_KV_PIN_AT_FREE_TTL', '2')
    async def check():
        calls = []
        async def utility(method, *args):
            calls.append((method, args))
            return 3
        d = PolicyDriver(TupleConfig(retention='search-seed', scheduling='search-seed'),
                         [NS(engine_core=NS(call_utility_async=utility))], asyncio.get_running_loop())
        try:
            d.programs.on_turn_release('a', 0, 0)
            d.retention_exec.policy.on_turn_complete = lambda *args: None
            await d.turn_complete('a', 0, 'a:0', 1, 32, 0, 10, kv_snapshot=SNAPSHOT)
            assert calls == [('kv_release', ('a:0',))]
            assert not d.programs.get('a').kv_protected
            assert d.retention_exec.decisions[-1].action == 'none'
        finally:
            d._worker.shutdown(wait=True)
    asyncio.run(check())


@pytest.mark.parametrize('policy', ['search-seed', 'gate', 'continuum', 'saga-tool-ttl'])
def test_sync_async_single_program_decisions_match(monkeypatch, policy):
    monkeypatch.setenv('VLLM_KV_RELEASE_AT_ARRIVAL', '1')
    monkeypatch.setenv('VLLM_KV_PIN_AT_FREE_TTL', '2')
    async def run(enabled):
        monkeypatch.setenv('BENCH_ASYNC_POLICY_RPC', '1' if enabled else '0')
        async def utility(method, *args):
            return dict(num_gpu_blocks=101, free_queue_blocks=25) if method=='kv_protection_stats' else 3
        d = PolicyDriver(TupleConfig(retention=policy, scheduling=policy if policy!='saga-tool-ttl' else 'fcfs', tau_s=2.0),
                         [NS(engine_core=NS(call_utility_async=utility))], asyncio.get_running_loop())
        try:
            for turn in range(2):
                await d.turn_ready('p', turn, 1+turn*2, 32, .2 if turn else None)
                await d.turn_complete('p', turn, f'p:{turn}', 1, 32, 0,
                                      2+turn*2, tool_name='tool', kv_snapshot=SNAPSHOT)
            return ([x.to_json() for x in d.retention_exec.decisions],
                    [x.to_json() for x in d.scheduling_exec.stamps])
        finally:
            d._worker.shutdown(wait=True)
    assert asyncio.run(run(False)) == asyncio.run(run(True))


@pytest.mark.parametrize('policy,snapshot', [('search-seed', SNAPSHOT), ('gate', None)])
def test_async_completion_with_callback_and_rpc_tracing(monkeypatch, tmp_path, policy, snapshot):
    import json
    monkeypatch.setenv('BENCH_ASYNC_POLICY_RPC', '1')
    monkeypatch.setenv('VLLM_KV_RELEASE_AT_ARRIVAL', '1')
    monkeypatch.setenv('VLLM_KV_PIN_AT_FREE_TTL', '2')
    callback_path = tmp_path / 'callbacks.jsonl'
    monkeypatch.setenv('BENCH_CALLBACK_TRACE', str(callback_path))
    monkeypatch.setenv('BENCH_KV_RPC_TRACE', str(tmp_path / 'rpcs.jsonl'))
    async def check():
        async def utility(method, *args):
            return dict(num_gpu_blocks=101, free_queue_blocks=25) if method == 'kv_protection_stats' else 3
        d = PolicyDriver(TupleConfig(retention=policy, scheduling=policy),
                        [NS(engine_core=NS(call_utility_async=utility))], asyncio.get_running_loop())
        try:
            await d.turn_ready('p', 0, 1, 32, None)
            await d.turn_complete('p', 0, 'p:0', 1, 32, 0, 2,
                                  tool_name='tool', kv_snapshot=snapshot)
            assert len(d.retention_exec.decisions) == 1
        finally:
            d._worker.shutdown(wait=True)
    asyncio.run(check())
    records = [json.loads(line) for line in callback_path.read_text().splitlines()]
    ack = [row for row in records if row['callback'] == '_ack']
    assert len(ack) == 1
    assert ack[0]['program_id'] == 'p'
