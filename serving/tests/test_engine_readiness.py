import asyncio

import pytest

from bench.core.readiness import await_engine_readiness


def test_readiness_waits_for_all_engines_and_records_setup_time():
    class Engine:
        calls = 0

        async def get_supported_tasks(self):
            await asyncio.sleep(0.01)
            self.calls += 1
            return ('generate',)

    engines = [Engine(), Engine()]
    records = asyncio.run(await_engine_readiness(engines))
    assert [e.calls for e in engines] == [1, 1]
    assert [r['instance'] for r in records] == [0, 1]
    assert all(r['elapsed_s'] >= 0.01 for r in records)


def test_non_generation_engine_fails_before_workload_submission():
    class Engine:
        async def get_supported_tasks(self):
            return ('embed',)

    with pytest.raises(ValueError, match='does not support generation'):
        asyncio.run(await_engine_readiness([Engine()]))
