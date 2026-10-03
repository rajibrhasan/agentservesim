
from typing import Callable, Optional

from .base import KVPolicy, SchedulingPolicy


class TTLRetention(KVPolicy):
    """Protect for now + tau at each gap start; release on the next
    turn's arrival. tau comes from the per-tool prediction when a
    predictor is given, else the fixed default."""

    @classmethod
    def from_config(cls, cfg):
        return cls(tau_s=cfg.tau_s)

    def __init__(
        self,
        tau_s: float,
        predictor: Optional[Callable[[Optional[str]], Optional[float]]] = None,
    ) -> None:
        self.tau_s = tau_s
        self.predictor = predictor

    def on_turn_complete(self, pcb, request_id, now):
        tau = None
        if self.predictor is not None:
            tau = self.predictor(pcb.tool_name)
        if tau is None:
            tau = self.tau_s
        return ("protect", now + tau)

    def on_turn_arrival(self, pcb, now):
        return "release"


class ProgramFCFSScheduling(SchedulingPolicy):
    """Priority = program's first-arrival time in ms since the run
    epoch, constant across the program's turns."""

    engine_args = {"scheduling_policy": "priority"}

    def __init__(self, epoch: Optional[float] = None) -> None:
        # The run epoch is a run constant, not program state.
        self.epoch = epoch

    def priority(self, pcb, now):
        if self.epoch is None:
            self.epoch = now
        arrival = pcb.arrival_ts if pcb.arrival_ts is not None else now
        return int(round((arrival - self.epoch) * 1000))
