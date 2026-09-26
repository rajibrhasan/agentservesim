"""Scheduling policy candidate for the automated policy search.

This file is the unit the search mutates. Only the region between the
EVOLVE-BLOCK markers changes; everything outside it is the fixed
contract. The class is loaded by the simulator as ``--scheduling
evolved`` and drives the Scheduling Plane: at submission of every turn
it stamps an integer priority that orders the engine's waiting queue.

Contract (harness/scheduling.py, SchedulingPolicy):

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
    """Pinned-first with mild context bias; favour earlier turns.

    1) KV-resident turns (kv_protected) effectively arrive earlier.
    2) Within each class we gently prefer smaller contexts, but cap
       their effect so arrival_ts still dominates.
    3) Earlier turns in a program (lower turn_idx) get a slight edge so
       long workflows flush more promptly once started.
    """

    engine_args = {"scheduling_policy": "priority"}
    _CTX_BAND = 10_000  # size where we stop distinguishing huge contexts
    _PIN_ADV_S = 30  # bounded arrival-time head start for resident KV
    _TURN_WEIGHT = 50  # ms advantage per earlier turn_idx

    def __init__(self):
        self.epoch = None

    def priority(self, pcb, now):
        if self.epoch is None:
            self.epoch = now
        arrival = pcb.arrival_ts if pcb.arrival_ts is not None else now
        ctx = pcb.context_tokens if pcb.context_tokens is not None else 0
        # Resident KV gets a bounded arrival-time head start; earlier
        # turns also get a small reduction so flows already in progress
        # are less likely to be delayed mid-workflow.
        adv = 0 if pcb.kv_protected else self._PIN_ADV_S
        turn_idx = pcb.turn_idx if pcb.turn_idx is not None else 0
        turn_bias_s = (min(turn_idx, 100) * self._TURN_WEIGHT) / 1000.0
        return int((arrival - self.epoch + adv - turn_bias_s) * 1000) + min(int(ctx), self._CTX_BAND)
# EVOLVE-BLOCK-END
