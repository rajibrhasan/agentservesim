"""Host-side arrival plumbing for the unchanged Gate policy."""
from dataclasses import replace

from .gate import EvolvedRetention


class ArrivalAdapter:
    @staticmethod
    def handles(policy):
        return isinstance(policy, EvolvedRetention)

    def __init__(self):
        self._seen = {}
        self._pending = {}

    def observe(self, policy, pcb, now):
        if not isinstance(policy, EvolvedRetention):
            policy.observe_arrival(pcb, now)
            return
        if pcb.gap_started_ts is None:
            return
        key = (pcb.turn_idx, pcb.gap_started_ts, pcb.tool_name)
        if self._seen.get(pcb.program_id) == key:
            return
        # Gate's original callback reads gap_elapsed_s(now). Project only its
        # observation view; never move the real gap clock or pin deadline.
        observed = pcb
        if pcb.completed_tool_duration_s is not None:
            observed = replace(pcb, gap_started_ts=now - pcb.completed_tool_duration_s)
        action = policy.on_turn_arrival(observed, now)
        self._seen[pcb.program_id] = key
        self._pending[pcb.program_id] = action

    def action(self, policy, pcb, now):
        if not isinstance(policy, EvolvedRetention):
            return policy.on_turn_arrival(pcb, now)
        self.observe(policy, pcb, now)
        # Learning precedes the priority stamp, but the transport release
        # remains after it. Duplicate callbacks cannot train or release twice.
        return self._pending.pop(pcb.program_id, None)
