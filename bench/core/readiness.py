"""Resolve engine capabilities before starting the workload arrival clock."""
import asyncio


async def await_engine_readiness(engines):
    records = []
    loop = asyncio.get_running_loop()
    for instance, engine in enumerate(engines):
        started = loop.time()
        tasks = await engine.get_supported_tasks()
        elapsed = loop.time() - started
        if 'generate' not in tasks:
            raise ValueError(f'Engine {instance} does not support generation: {tasks}')
        records.append(dict(instance=instance, supported_tasks=list(tasks), elapsed_s=elapsed))
    return records
