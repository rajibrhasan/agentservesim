"""Autellix (arXiv:2502.13965), the PLAS half.

Priority is the program's attained service at submission, so a program that has
already consumed a lot of compute yields to one that has not. One plane only:
Autellix says nothing about what is kept in KV or where a turn runs.

The paper's MLFQ half is not implemented (arena spec: corner `reprioritize`).
"""
from .base import SchedulingPolicy

PAPER = "Autellix 2502.13965 (PLAS/ATLAS)"


class AutellixScheduling(SchedulingPolicy):
    """Priority = program's attained service (ms) at submission,
    accumulated from measured service seconds of completed turns."""

    engine_args = {"scheduling_policy": "priority"}

    def priority(self, pcb, now):
        return int(round(pcb.attained_service_s * 1000.0))
