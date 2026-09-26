"""Initial policy from joint search 41008503, generation zero.

Restored from program 60b6ab54-1f6d-4dee-b182-85fb8ba85198.
Decision code is unchanged; imports use the shared policy bases.
"""
from .base import RetentionPolicy, SchedulingPolicy

# EVOLVE-BLOCK-START
class EvolvedRetention(RetentionPolicy):
    """Seed: fixed 2 s pin after every turn, released on the next
    arrival (the ttl-2 half of the best measured tuple, ttl2-cont)."""

    TAU_S = 2.0

    def __init__(self):
        # Aggregate state only (no per-program tables).
        self.gap_ema_by_tool = {}

    def on_turn_complete(self, pcb, request_id, now):
        return ("protect", now + self.TAU_S)

    def on_turn_arrival(self, pcb, now):
        gap = pcb.gap_elapsed_s(now)
        if gap is not None and pcb.tool_name is not None:
            prev = self.gap_ema_by_tool.get(pcb.tool_name)
            self.gap_ema_by_tool[pcb.tool_name] = (
                gap if prev is None else 0.8 * prev + 0.2 * gap)
        return "release"


class EvolvedScheduling(SchedulingPolicy):
    """Seed: pinned-first, then program-FCFS with a bounded
    small-context bonus (the best program of the scheduling-only
    search, +1.8% over pinned-first FCFS at the saturated rate).

    Order: turns whose KV is resident (kv_protected) run before turns
    that must re-prefill; within a class, earlier programs first, with
    contexts under _CTX_BAND tokens moved up by at most _CTX_BAND ms of
    arrival credit. victim() and admit() are the engine defaults.
    """

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
        # Engine default: preempt the running request with the largest
        # priority value. Return an index into candidates to override.
        return None

    def admit(self, pcb, now, view):
        # Engine default: every waiting turn is offered to the engine.
        return True
# EVOLVE-BLOCK-END
