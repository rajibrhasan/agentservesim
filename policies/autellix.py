
from .base import SchedulingPolicy

PAPER = "Autellix 2502.13965 (PLAS/ATLAS)"


class AutellixScheduling(SchedulingPolicy):
    """Priority = program's attained service (ms) at submission,
    accumulated from measured service seconds of completed turns."""

    engine_args = {"scheduling_policy": "priority"}

    def priority(self, pcb, now):
        return int(round(pcb.attained_service_s * 1000.0))
