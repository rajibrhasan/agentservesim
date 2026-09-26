"""Wait only for the part of a tool gap not already spent in callbacks."""
import asyncio


async def wait_for_tool(completed_ts, duration_ns, *, clock, sleep=asyncio.sleep):
    deadline = completed_ts + duration_ns / 1e9
    remaining = deadline - clock()
    if remaining > 0:
        await sleep(remaining)
    return deadline
