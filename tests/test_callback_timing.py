import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from bench.core.policy_driver import PolicyDriver


def test_callback_trace_preserves_return_and_failure(monkeypatch, tmp_path):
    path = tmp_path/'callbacks.jsonl'
    monkeypatch.setenv('BENCH_CALLBACK_TRACE', str(path))
    async def check():
        d = PolicyDriver.__new__(PolicyDriver)
        d._loop = asyncio.get_running_loop()
        d._worker = ThreadPoolExecutor(max_workers=1)
        def success(program):
            return 7
        def failure(program):
            raise ValueError('expected')
        try:
            assert await d._run(success, 'p') == 7
            try:
                await d._run(failure, 'p')
            except ValueError as error:
                assert str(error) == 'expected'
            else:
                assert False, 'exception swallowed'
        finally:
            d._worker.shutdown(wait=True)
    asyncio.run(check())
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x['callback'] for x in rows] == ['success', 'failure']
    assert all(x['queued_ts'] <= x['started_ts'] <= x['finished_ts'] for x in rows)


def test_rpc_trace_records_protection_deadline_and_ack(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from bench.core.policy_driver import AsyncEngineKVControl
    path = tmp_path/'rpcs.jsonl'
    monkeypatch.setenv('BENCH_KV_RPC_TRACE', str(path))
    async def check():
        async def utility(method, request, deadline):
            assert (method, request, deadline) == ('kv_protect', 'p:0', 112)
            return 3
        engine = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))
        loop = asyncio.get_running_loop()
        control = AsyncEngineKVControl(engine, loop, 100)
        result = await loop.run_in_executor(None, control.protect, 'p:0', 12)
        assert result == 3
    asyncio.run(check())
    row = json.loads(path.read_text())
    assert row['succeeded'] is True
    assert row['affected_blocks'] == 3
    assert row['deadline_ts'] == 12
