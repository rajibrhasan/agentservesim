"""Joint policy candidate (retention + scheduling) for the automated
policy search.

This file is the unit the search mutates. Only the region between the
EVOLVE-BLOCK markers changes; everything outside it is the fixed
contract. The simulator loads it as ``--retention evolved --scheduling
evolved``: EvolvedRetention decides which programs' KV stays resident
across tool gaps, EvolvedScheduling decides which waiting turn runs,
which running turn is preempted under pressure, and whether a waiting
turn is submitted to the engine at all. The two classes read the SAME
Program Control Block, so a retention decision is visible to the
scheduler as pcb.kv_protected on the program's next turn, and a
scheduling decision changes when (and whether) the protected KV is
ever used. Evolve them together.

Retention contract (policies/base.py, RetentionPolicy):

  on_turn_complete(pcb, request_id, now) -> ("protect", deadline_ts)
        | ("evict", None) | None
      Called when a turn's response finishes and the program enters
      its tool gap. "protect" pins the program's context in the KV
      pool until deadline_ts (seconds, absolute) or the next arrival;
      "evict" drops it now; None leaves it to the engine's LRU.
  on_turn_arrival(pcb, now) -> "release" | None
      Called when the program's next turn arrives. "release" unpins
      the context (it stays cached, exposed to LRU); None keeps the
      pin until its deadline.
  A pin holds pool space. Under pressure the engine's valve breaks
  pins expired-first, then latest-deadline-first, so pinning more
  than the pool can hold is not free: it evicts other programs' KV.
  self.signals.kv_utilization (0..1, may be None) is the pool state.

Scheduling contract (policies/base.py, SchedulingPolicy):

  priority(pcb, now) -> int | None
      Once per turn at submission. Among WAITING requests the
      SMALLEST value is admitted first; arrival breaks ties. None =
      unstamped (stock FCFS for that turn). Running requests are
      never reordered.
  victim(candidates, now) -> int | None
      Under memory pressure, which RUNNING request to preempt (index
      into candidates, a list of VictimView: pcb, priority,
      prompt_tokens, computed_tokens, generated_tokens, is_prefill).
      Preemption is by RECOMPUTE: the victim loses its whole KV and
      re-prefills prompt + tokens generated so far. None = engine
      default (largest priority value, latest arrival).
  admit(pcb, now, view) -> bool
      Gateway-side admission gate, called in queue order every
      scheduling tick for each waiting turn. False holds the turn this
      tick (it keeps its place); True lets the engine try to fit it.
      view is a QueueView: n_running, n_waiting, n_inflight,
      kv_utilization, kv_free_tokens, kv_evictable_tokens,
      prompt_tokens, cached_tokens. Holding is never free: an idle
      engine admits the head regardless, and a held turn's program
      keeps its KV pinned (if pinned) while it waits.

Observation boundary (both classes): ``now`` (seconds) and the PCB:
  pcb.turn_idx, pcb.turns_completed, pcb.arrival_ts,
  pcb.attained_service_s, pcb.context_tokens, pcb.tool_name,
  pcb.in_gap, pcb.gap_started_ts, pcb.gap_elapsed_s(now),
  pcb.kv_protected, pcb.kv_deadline_ts, pcb.kv_instance
Nothing about the future (output length, next gap, remaining turns).
Reading pcb.program_id or pcb.kv_request_id is forbidden; per-program
tables inside the policy are forbidden (aggregate statistics such as
a running mean of gaps by tool name are allowed).

Fitness: mean program JCT of the stock configuration (LRU cache, FCFS)
divided by mean program JCT under this candidate, on a saturated cell
and then a second arrival rate; the score is the mean over the cells
run so far, the minimum is reported alongside, and vs_seed reports the
same ratio against this seed. Higher is better.
"""

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
