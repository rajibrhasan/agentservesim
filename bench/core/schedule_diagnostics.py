"""Wrap the selected vLLM scheduler without replacing its scheduling rules."""
import importlib
import time
from types import MethodType

from runtime.schedule_trace import ScheduleTrace


def snapshot(scheduler):
    pool = scheduler.kv_cache_manager.block_pool
    result = dict(running=[r.request_id for r in scheduler.running],
                waiting=[r.request_id for r in scheduler.waiting],
                requests={rid: dict(computed=r.num_computed_tokens,
                                    cached=r.num_cached_tokens, status=str(r.status),
                                    prompt_tokens=r.num_prompt_tokens, known_tokens=r.num_tokens,
                                    generated_tokens=r.num_output_tokens, preemptions=r.num_preemptions)
                          for rid, r in scheduler.requests.items()},
                free_blocks=pool.free_block_queue.num_free_blocks,
                capacity_blocks=pool.num_gpu_blocks,
                protected_blocks=len(pool._protected),
                protection_counters=dict(pool.protection_stats))
    if hasattr(scheduler, '_turn_kv'):
        result['turns'] = {rid: dict(program=m.program, turn=m.turn, complete=m.complete,
                                   computed_total=m.compute_total, recomputed_total=m.recompute_total)
                           for rid, m in scheduler._turn_kv.items()}
        residency = scheduler.residency
        result['residency'] = {rid: dict(cpu_blocks=len(st.cpu), discarded=st.discarded,
                                       computed_tokens=st.computed_tokens)
                               for rid, st in residency.states.items()}
        pending = residency.pending
        result['transfer'] = None if pending is None else dict(ticket=pending.ticket,
            chunks=[dict(request=c.state.request.request_id, storing=c.storing,
                         indices=c.indices) for c in pending.chunks])
        result['infercept_counters'] = dict(scheduler.infercept_stats, **residency.stats)
    return result


class SchedulingTraceMixin:
    __slots__ = ()

    def schedule(self):
        before = snapshot(self)
        start = time.monotonic_ns()
        manager = self.kv_cache_manager
        attempts, originals = [], {}
        def wrap(name, original):
            def checked(*args, **kwargs):
                request = kwargs.get('request', args[0] if args else None)
                pool = manager.block_pool
                row = dict(operation=name, request=request.request_id,
                           free_blocks=pool.free_block_queue.num_free_blocks,
                           protected_blocks=len(pool._protected))
                result = original(*args, **kwargs)
                row['success'] = result if name == 'can_fit_full_sequence' else result is not None
                attempts.append(row)
                return result
            return checked
        try:
            for name in ('allocate_slots', 'can_fit_full_sequence'):
                if hasattr(manager, name):
                    original = getattr(manager, name)
                    originals[name] = (original, name in manager.__dict__)
                    setattr(manager, name, wrap(name, original))
            output = self._trace_original_schedule()
        finally:
            for name, (original, was_local) in originals.items():
                if was_local:
                    setattr(manager, name, original)
                else:
                    delattr(manager, name)
        end = time.monotonic_ns()
        self._schedule_trace.write(plane='real', monotonic_ns=start,
                                   schedule_duration_ns=end - start,
                                   scheduled_tokens=dict(output.num_scheduled_tokens),
                                   allocation_attempts=attempts,
                                   before=before, after=snapshot(self))
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        start = time.monotonic_ns()
        result = self._trace_original_update(scheduler_output, model_runner_output)
        self._schedule_trace.write(plane='real', event='model_output',
                                   monotonic_ns=start, finished_ns=time.monotonic_ns(),
                                   scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
                                   after=snapshot(self))
        return result


def attach(scheduler, directory):
    # Resolve factories first (including PolicyScheduler's async selection).
    # Bind wrappers to this instance; changing __class__ is incompatible with
    # some native scheduler layouts. Original bound methods retain their MRO.
    scheduler._trace_original_schedule = scheduler.schedule
    scheduler.schedule = MethodType(SchedulingTraceMixin.schedule, scheduler)
    if hasattr(scheduler, 'update_from_output'):
        scheduler._trace_original_update = scheduler.update_from_output
        scheduler.update_from_output = MethodType(SchedulingTraceMixin.update_from_output, scheduler)
    scheduler._schedule_trace = ScheduleTrace(directory, 'real')
    return scheduler


class DiagnosticScheduler:
    def __new__(cls, *args, **kwargs):
        config = kwargs['vllm_config']
        diagnostic = config.additional_config['schedule_diagnostics']
        name = diagnostic['base_class']
        if name is None:
            name = ('vllm.v1.core.sched.async_scheduler.AsyncScheduler'
                    if config.scheduler_config.async_scheduling else
                    'vllm.v1.core.sched.scheduler.Scheduler')
        if isinstance(name, str):
            module, attr = name.rsplit('.', 1)
            base = getattr(importlib.import_module(module), attr)
        else:
            base = name
        return attach(base(*args, **kwargs), diagnostic['directory'])
