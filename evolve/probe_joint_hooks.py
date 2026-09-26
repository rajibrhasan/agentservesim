"""Mechanism probe for the wider scheduling hooks (not a search seed):
the joint seed with victim() and admit() overridden, so a simulator run
exercises both paths end to end. victim: preempt the running request
whose recompute is cheapest (fewest computed tokens); admit: hold
unpinned turns while the pool is above 90% and something is running."""

from harness.retention import RetentionPolicy
from harness.scheduling import SchedulingPolicy


# EVOLVE-BLOCK-START
class EvolvedRetention(RetentionPolicy):
    TAU_S = 2.0

    def __init__(self):
        self.gap_ema_by_tool = {}

    def on_turn_complete(self, pcb, request_id, now):
        return ("protect", now + self.TAU_S)

    def on_turn_arrival(self, pcb, now):
        return "release"


class EvolvedScheduling(SchedulingPolicy):
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
        # Cheapest recompute first: fewest tokens currently held.
        return min(range(len(candidates)),
                   key=lambda i: candidates[i].computed_tokens)

    def admit(self, pcb, now, view):
        if view.n_running == 0:
            return True
        if pcb.kv_protected:
            return True
        return view.kv_utilization < 0.9
# EVOLVE-BLOCK-END
