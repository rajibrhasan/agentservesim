"""Opt-in vLLM v0.19 scheduler observations for policy execution.

Loaded through scheduler_cls, without modifying the installed vLLM checkout.
Both sync and async scheduler implementations retain their native scheduling
behavior. This module exposes observations; it does not enable Autellix's
step planner or implement a CPU transfer worker.
"""
import time


class PolicyObservations:
    def schedule(self):
        output = super().schedule()
        self._policy_last_scheduled = dict(output.num_scheduled_tokens)
        return output

    def kv_protection_stats(self):
        stats = super().kv_protection_stats()
        pool = self.kv_cache_manager.block_pool
        now = time.time()
        # Request tracking entries outlive individual block protections.
        # Report surviving, unexpired blocks, rather than dictionary presence.
        protected = {}
        for tag, block_ids in self.kv_cache_manager._protected_requests.items():
            count = sum(pool._protected.get(bid, float('-inf')) > now
                        for bid in set(block_ids))
            if count:
                protected[tag] = count
        stats['policy_observation'] = {
            'schema_version': 1,
            'protected_blocks_by_tag': protected,
            # Shape of the last scheduled batch, excluding calls that have
            # completed since it was formed. This is not queue backlog.
            'scheduled_query_tokens': sum(
                n for rid, n in self._policy_last_scheduled.items()
                if rid in self.requests),
            'running_context_tokens': sum(
                r.num_computed_tokens for r in self.running),
        }
        return stats


class PolicyScheduler:
    """Factory preserving vLLM's effective async_scheduling selection."""

    def __new__(cls, *args, **kwargs):
        config = kwargs['vllm_config']
        if config.scheduler_config.async_scheduling:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler
            base = AsyncScheduler
        else:
            from vllm.v1.core.sched.scheduler import Scheduler
            base = Scheduler
        scheduler_type = type('ObservedScheduler', (PolicyObservations, base), {})
        scheduler = scheduler_type(*args, **kwargs)
        scheduler._policy_last_scheduled = {}
        return scheduler
