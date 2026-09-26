"""Opt-in callback phases: serialized decisions, independent engine waits."""
import asyncio
from collections import defaultdict
import os


class AsyncPolicyCallbacks:
    def __init__(self, driver):
        from .policy_driver import _FanoutKVControl
        if driver.cfg.engine_policy or driver.cfg.engine_observations:
            raise ValueError('Async policy RPCs do not yet support native engine-policy observations')
        if driver.cfg.retention == 'min-waste':
            raise ValueError('Async policy RPCs do not yet support inline min-waste load probes')
        self.driver = driver
        self.fanout = isinstance(driver.kv, _FanoutKVControl)
        if self.fanout and not os.environ.get('VLLM_KV_RELEASE_AT_ARRIVAL'):
            raise ValueError('Async policy RPCs require engine-side arrival/scheduled release')
        self.locks = defaultdict(asyncio.Lock)

    async def _rpc(self, instance, method, *args):
        # Only the per-engine transport runs here. Program state, shared policy
        # state and ownership maps are exclusively touched by driver._run.
        control = self.driver.kv._controls[instance]
        return await self.driver._loop.run_in_executor(None, getattr(control, method), *args)

    async def _ordered(self, program, operation):
        async with self.locks[program]:
            task = asyncio.create_task(operation())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Do not abandon an in-flight protect between its engine effect
                # and ownership acknowledgement, or release this program's lock.
                await task
                raise

    async def turn_ready(self, program, turn, now, prompt, gap):
        d = self.driver
        async def operation():
            if gap is not None:
                await d._run(d.programs.observe_completed_tool, program, gap)
            instance = await d._run(d._route_sync, program, turn, now, prompt)
            # Keep fresh arrival queries for every policy. Query only the
            # selected engine and wait outside the shared decision worker.
            stats = await self._rpc(instance, 'stats') if self.fanout else {}
            priority = await d._run(d._arrival_sync, program, turn, now, stats)
            return instance, priority
        return await self._ordered(program, operation)

    async def admit(self, program, turn, prompt, inflight, now):
        d = self.driver
        async def operation():
            pcb = await d._run(d.programs.get, program)
            instance = pcb.kv_instance or 0
            stats = await self._rpc(instance, 'stats') if self.fanout else {}
            return await d._run(d._admit_sync, program, turn, prompt, inflight, now, stats)
        return await self._ordered(program, operation)

    def _prepare(self, program, turn, request, service, context, instance,
                 now, tool, utilization):
        d = self.driver
        if d.routing_exec is not None:
            d.routing_exec.turn_complete(instance)
        d.scheduling_exec.turn_complete(program, service, turn_idx=turn)
        dec = d.retention_exec.prepare_complete(
            program, turn, request, tool, now, context, utilization)
        target = instance
        if self.fanout:
            if dec.action == 'protect':
                stale = d.programs.get(program).kv_request_id
                if stale is not None:
                    d.kv.release(stale)  # deferred engine-side release; no RPC
                    d.programs.note_retention(program, 'release')
            elif dec.action == 'evict':
                target = d.kv._where.get(request, instance)
        return dec, target

    def _ack(self, dec, instance, blocks):
        d = self.driver
        if self.fanout:
            if dec.action == 'protect' and blocks > 0:
                d.kv._where[dec.request_id] = instance
            elif dec.action == 'evict' or (dec.action == 'none'
                    and os.environ.get('VLLM_KV_PIN_AT_FREE_TTL')):
                d.kv._where.pop(dec.request_id, None)
        return d.retention_exec.ack_complete(dec, blocks)

    async def turn_complete(self, program, turn, request, service, context,
                            instance, now, tool, snapshot):
        d = self.driver
        async def operation():
            if snapshot is not None:
                utilization = d._completion_utilization(snapshot)
            elif self.fanout and d.cfg.retention != 'continuum':
                stats = await self._rpc(instance, 'stats')
                utilization = d._pool_view_from_stats(stats)[0]
            else:
                utilization = None
            dec, target = await d._run(self._prepare, program, turn, request,
                                      service, context, instance, now, tool, utilization)
            blocks = None
            if dec.action == 'swap':
                raise ValueError('Async policy RPCs do not support gateway swap actions')
            if dec.action != 'none':
                args = (request, dec.deadline_ts) if dec.action == 'protect' else (request,)
                blocks = await self._rpc(target, dec.action, *args) if self.fanout else 0
            elif self.fanout and os.environ.get('VLLM_KV_PIN_AT_FREE_TTL'):
                await self._rpc(target, 'cancel_provisional', request)
            await d._run(self._ack, dec, target, blocks, trace_program_id=program)
        return await self._ordered(program, operation)
