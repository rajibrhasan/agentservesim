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

Retention contract (harness/retention.py, RetentionPolicy):

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

Scheduling contract (harness/scheduling.py, SchedulingPolicy):

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

# Copy of evolve/champion_joint_41008503.py (the search's joint champion, the
# 2026-09-15 board's "gate" point) promoted to a named policy: `--retention
# gate --scheduling gate` runs it through the same flag path as the published
# values instead of a staged candidate file. The original stays in evolve/ as
# the record of the run; only these two imports differ (the harness shim
# resolves to these same classes).
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
        # Long-gap tools: don't hold pool space with a pin that will
        # expire long before the tool returns. Leave the context to
        # the engine's own LRU instead of forcing an eviction, so a
        # cache hit is still possible if the pool happens to have
        # room. Short-gap tools: pin so the resident context is used.
        expected_gap = self.gap_ema_by_tool.get(pcb.tool_name)
        if expected_gap is not None and expected_gap > 1.5 * self.TAU_S:
            return None
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
        # Preemption discards the victim's whole KV and re-prefills
        # prompt + generated tokens, so sacrifice whichever running
        # request is cheapest to recompute rather than the engine's
        # default (largest priority value, latest arrival).
        if not candidates:
            return None
        return min(
            range(len(candidates)),
            key=lambda i: (candidates[i].prompt_tokens
                           + candidates[i].generated_tokens))

    def admit(self, pcb, now, view):
        # If this turn's context is already resident (pinned), always
        # let it try: it's cheap to admit and doesn't need eviction.
        if pcb.kv_protected:
            return True
        # Otherwise, only hold it back when the pool is tight and this
        # turn's prompt plainly cannot fit even after every evictable
        # block is freed: admitting it now would just force a
        # preemption that fails to make room, or evicts other
        # programs' KV for nothing. Let it wait one tick so turns that
        # do fit can proceed; free space grows as running turns finish.
        util = view.kv_utilization
        if util is not None and util > 0.9:
            free = view.kv_free_tokens or 0
            evictable = view.kv_evictable_tokens or 0
            # Account for unpinned contexts that remain resident under
            # LRU: only the uncached prompt portion needs new KV space.
            prompt = view.prompt_tokens or pcb.context_tokens or 0
            cached = view.cached_tokens or 0
            needed = max(0, prompt - cached)
            if needed > free + evictable:
                return False
        return True
# EVOLVE-BLOCK-END
