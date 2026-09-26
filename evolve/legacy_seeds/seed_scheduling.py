"""Scheduling policy candidate for the automated policy search.

This file is the unit the search mutates. Only the region between the
EVOLVE-BLOCK markers changes; everything outside it is the fixed
contract. The class is loaded by the simulator as ``--scheduling
evolved`` and drives the Scheduling Plane: at submission of every turn
it stamps an integer priority that orders the engine's waiting queue.

Contract (policies/base.py, SchedulingPolicy):

  priority(pcb, now) -> int | None
      Called once per turn, at submission. The engine runs with
      vLLM's priority queue: among WAITING requests, the SMALLEST
      priority is admitted first, with arrival order as the tiebreak.
      Returning None leaves the turn unstamped, which is stock FCFS
      behaviour for that turn.

What the priority can and cannot do (vLLM v1 semantics):
  - It orders the WAITING queue only. Running requests are never
    reordered by it.
  - Under memory pressure the scheduler preempts the running request
    with the LARGEST priority value first, so a high number is both
    "admitted last" and "preempted first".
  - It is stamped once per turn and does not change while the turn
    waits; a policy that wants aging must express it through values it
    computes at stamp time.

Observation boundary: the ONLY inputs are ``now`` (seconds) and the
Program Control Block fields below. Nothing about the future (the
turn's output length, the next tool gap, the program's remaining turns)
is available. Reading pcb.program_id or pcb.kv_request_id is forbidden,
and keeping any per-program table inside the policy is forbidden:
program state lives in the PCB, the policy holds only aggregate state.

  pcb.turn_idx            index of the turn being submitted (0-based)
  pcb.turns_completed     turns this program has completed so far
  pcb.arrival_ts          first arrival of the program (seconds)
  pcb.attained_service_s  total service time the program has received
  pcb.context_tokens      tokens of context this turn carries
  pcb.tool_name           name of the tool whose gap just ended
                          (None when unknown)
  pcb.in_gap, pcb.gap_started_ts, pcb.gap_elapsed_s(now)
                          tool-gap state of the program
  pcb.kv_protected, pcb.kv_deadline_ts, pcb.kv_instance
                          current KV residency state: kv_protected says
                          the program's context is still resident, so
                          this turn can skip re-prefilling it

Fitness: mean program JCT of the stock configuration divided by mean
program JCT under this policy, on a saturated cell and then a second
arrival rate; the reported score is the mean over the cells run so far,
and the minimum over cells is reported alongside it. 1.0 is parity with
stock FCFS; higher is better. Ordering cannot create throughput: it can
only decide who waits. Short turns behind long ones lose more than they
gain, and a program that never reaches the front of the queue drags the
mean down however well the others do.
"""

from harness.scheduling import SchedulingPolicy


# EVOLVE-BLOCK-START
class EvolvedScheduling(SchedulingPolicy):
    """Seed: Continuum pinned-first program-FCFS (arXiv:2511.02230) —
    the best measured policy on the leaderboard cells (12-21% mean-JCT
    win over FCFS, PLAS, and every retention variant).

    Two-level order: turns of programs whose KV is still resident
    (kv_protected) go before turns that must re-prefill; within each
    class, program-level FCFS by first arrival. Rationale: a resident
    context finishes its turn without paying re-prefill, so serving it
    first frees the pool sooner; program seniority bounds starvation.
    """

    # The engine must run its priority queue for a stamp to mean anything.
    engine_args = {"scheduling_policy": "priority"}

    _CLASS = 1 << 40

    def __init__(self):
        # Aggregate state only (no per-program tables).
        self.epoch = None

    def priority(self, pcb, now):
        if self.epoch is None:
            self.epoch = now
        arrival = pcb.arrival_ts if pcb.arrival_ts is not None else now
        pinned = 0 if pcb.kv_protected else 1
        return pinned * self._CLASS + int(round((arrival - self.epoch) * 1000))
# EVOLVE-BLOCK-END
