"""Diagnostic (not a search seed): the ttl2-cont ordering rule rewritten
WITHOUT reading pcb.kv_protected. Hypothesis: under arrival-release
retention the protected flag is True for every continuing turn and False
only for a program's first turn, so pinned-first == "new programs last,
then program-FCFS". If so this file reproduces ttl2-cont's lb20_j0.1 JCT
(4,115.9 s) exactly."""

from harness.scheduling import SchedulingPolicy


# EVOLVE-BLOCK-START
class EvolvedScheduling(SchedulingPolicy):
    engine_args = {"scheduling_policy": "priority"}
    _CLASS = 1 << 40

    def __init__(self):
        self.epoch = None

    def priority(self, pcb, now):
        if self.epoch is None:
            self.epoch = now
        arrival = pcb.arrival_ts if pcb.arrival_ts is not None else now
        new_program = 1 if pcb.turns_completed == 0 else 0
        return new_program * self._CLASS + int(round((arrival - self.epoch) * 1000))
# EVOLVE-BLOCK-END
