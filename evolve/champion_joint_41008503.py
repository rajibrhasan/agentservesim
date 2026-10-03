"""Evolved joint retention and scheduling policy.

Callbacks share observed program state. Deadlines use absolute seconds;
lower priority runs first. Only the EVOLVE-BLOCK region is mutable."""

from harness.retention import RetentionPolicy
from harness.scheduling import SchedulingPolicy


# EVOLVE-BLOCK-START
class EvolvedRetention(RetentionPolicy):
    """Seed: fixed 2 s pin after every turn, released on the next
    arrival (the ttl-2 half of the best measured tuple, ttl2-cont)."""

    TAU_S = 2.0

    def __init__(self):
        # Aggregate state only (no per-program tables).
        self.gap_ema_by_tool = {}

    def on_turn_complete(self, pcb, request_id, now):
        expected_gap = self.gap_ema_by_tool.get(pcb.tool_name)
        if expected_gap is not None and expected_gap > 1.5 * self.TAU_S:
            return None
        return ("protect", now + self.TAU_S)

    def on_turn_arrival(self, pcb, now):
        gap = pcb.gap_elapsed_s(now)
        if gap is not None and pcb.tool_name is not None:
            prev = self.gap_ema_by_tool.get(pcb.tool_name)
            self.gap_ema_by_tool[pcb.tool_name] = (
                gap if prev is None else 0.8 * prev + 0.2 * gap)
        return "release"


class EvolvedScheduling(SchedulingPolicy):
    """Seed: pinned-first, then program-FCFS with a bounded small-context bonus (the best program of the scheduling-only search, +1.8% over pinned-first FCFS at the saturated rate)."""

    engine_args = {"scheduling_policy": "priority"}
    _CLASS = 1 << 40
    _CTX_BAND = 10_000

    def __init__(self):
        self.epoch = None

    def priority(self, pcb, now):
        if self.epoch is None:
            self.epoch = now
        arrival = pcb.arrival_ts if pcb.arrival_ts is not None else now
        pinned = 0 if pcb.kv_protected else 1
        ctx = pcb.context_tokens if pcb.context_tokens is not None else 0
        return (pinned * self._CLASS
                + int((arrival - self.epoch) * 1000)
                + min(int(ctx), self._CTX_BAND))

    def victim(self, candidates, now):
        if not candidates:
            return None
        return min(
            range(len(candidates)),
            key=lambda i: (candidates[i].prompt_tokens
                           + candidates[i].generated_tokens))

    def admit(self, pcb, now, view):
        # If this turn's context is already resident (pinned), always
        # let it try: it's cheap to admit and doesn't need eviction.
        if pcb.kv_protected:
            return True
        util = view.kv_utilization
        if util is not None and util > 0.9:
            free = view.kv_free_tokens or 0
            evictable = view.kv_evictable_tokens or 0
            # Account for unpinned contexts that remain resident under
            # LRU: only the uncached prompt portion needs new KV space.
            prompt = view.prompt_tokens or pcb.context_tokens or 0
            cached = view.cached_tokens or 0
            needed = max(0, prompt - cached)
            if needed > free + evictable:
                return False
        return True
# EVOLVE-BLOCK-END
