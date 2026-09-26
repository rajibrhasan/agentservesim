"""Joint policy candidate seeded from Continuum (Li et al., 2025,
arXiv:2511.02230, mirror of the released vllm-continuum code): the best
measured policy on the collected SWE-bench cells (1.4-2.1x over stock
LRU/FCFS, counters verified).

This file is the unit the search mutates. Only the region between the
EVOLVE-BLOCK markers changes. The simulator loads it as ``--retention
evolved --scheduling evolved``: EvolvedRetention decides which programs'
KV stays resident across tool gaps, EvolvedScheduling decides which
waiting turn runs, which running turn is preempted under pressure, and
whether a waiting turn is submitted to the engine at all. Both read the
SAME Program Control Block.

How the two halves couple in THIS seed (important):
  - EvolvedRetention sets release_event = "scheduled": a pin made at a
    turn's completion is held until the program's NEXT turn is ADMITTED
    to the running batch, i.e. through its whole queue wait. Under
    memory pressure the engine's valve breaks pins (expired-first, then
    latest-deadline-first) to admit the queue head.
  - Because pins can be broken by the valve, pcb.kv_protected at the
    next turn's submission is a REAL signal: True means the context is
    still resident, False means it was reclaimed (or never pinned).
    EvolvedScheduling's pinned-first rule depends on that. With an
    arrival-released policy (release_event = "arrival") the flag is
    True for every continuing turn and carries no information.
  - Holding pins through the queue reserves pool space for idle
    programs. It pays when the reserved context is used soon; it costs
    when many programs wait and the pool cannot hold them all.

Retention contract (policies/base.py, RetentionPolicy):
  on_turn_complete(pcb, request_id, now) -> ("protect", deadline_ts[, info])
        | ("evict", None) | None
  on_turn_arrival(pcb, now) -> "release" | None
  observe_arrival(pcb, now) is called at each arrival BEFORE the record
  changes: pcb.tool_name and pcb.gap_elapsed_s(now) describe the gap that
  just ended (the place to learn per-tool statistics).
  Class attribute release_event: "arrival" or "scheduled" (see above).

Scheduling contract (policies/base.py, SchedulingPolicy):
  priority(pcb, now) -> int | None      smallest runs first; None = FCFS
  victim(candidates, now) -> int | None index into VictimView list (pcb,
        priority, prompt_tokens, computed_tokens, generated_tokens,
        is_prefill) of the running request to preempt by recompute;
        None = engine default (largest priority value)
  admit(pcb, now, view) -> bool         gateway gate per waiting turn per
        tick; view = QueueView(n_running, n_waiting, n_inflight,
        kv_utilization, kv_free_tokens, kv_evictable_tokens,
        prompt_tokens, cached_tokens); an idle engine admits the head
        regardless

Observation boundary (both classes): ``now`` (seconds) and the PCB:
  pcb.turn_idx, pcb.turns_completed, pcb.arrival_ts,
  pcb.attained_service_s, pcb.context_tokens,
  pcb.gap_mean_s, pcb.gap_n            (this program's PAST gaps)
  pcb.in_gap, pcb.gap_started_ts, pcb.gap_elapsed_s(now)
  pcb.kv_protected, pcb.kv_deadline_ts, pcb.kv_instance
  pcb.tool_name          the tool the program just called (set at the
                         turn's completion, cleared when the next turn is
                         released): sed, grep, cat, python, git, find, pip,
                         python3, ls, echo, ... (24-26 distinct per cell)
Nothing about the future. Reading pcb.program_id or pcb.kv_request_id
and per-program tables inside the policy are forbidden.

Fitness: stock (LRU, FCFS) mean program JCT divided by the candidate's,
on a saturated cell then a second arrival rate; the mean over cells is
the score, the minimum and vs_seed are reported alongside. A candidate
whose code is identical after removing comments and formatting is not
simulated again.
"""

from harness.retention import RetentionPolicy
from harness.scheduling import SchedulingPolicy


# EVOLVE-BLOCK-START
class EvolvedRetention(RetentionPolicy):
    """Seed: Continuum as released. The mean execution time of each TOOL
    is learned online from the gaps observed at arrival; after a turn,
    pin for PIN_S iff the tool just called has mean <= THRESHOLD_S (or no
    history yet: cold start pins). Slow tools are not pinned (their
    blocks fall to the ordinary LRU). The pin is held until the next
    turn is admitted (release_event = "scheduled")."""

    release_event = "scheduled"
    PIN_S = 2.0
    THRESHOLD_S = 2.0

    def __init__(self):
        # Aggregate statistics by tool name (allowed): running sum and
        # count of observed gaps per tool, across all programs.
        self.gap_sum_by_tool = {}
        self.gap_n_by_tool = {}

    def observe_arrival(self, pcb, now):
        gap = pcb.gap_elapsed_s(now)
        if gap is None or pcb.tool_name is None:
            return None
        self.gap_sum_by_tool[pcb.tool_name] = self.gap_sum_by_tool.get(pcb.tool_name, 0.0) + gap
        self.gap_n_by_tool[pcb.tool_name] = self.gap_n_by_tool.get(pcb.tool_name, 0) + 1
        return None

    def on_turn_complete(self, pcb, request_id, now):
        n = self.gap_n_by_tool.get(pcb.tool_name, 0)
        mean = (self.gap_sum_by_tool[pcb.tool_name] / n) if n else None
        if mean is not None and mean > self.THRESHOLD_S:
            return None
        return ("protect", now + self.PIN_S,
                {"tool": pcb.tool_name, "mean_gap_s": mean, "n_gaps": n})

    def on_turn_arrival(self, pcb, now):
        return "release"


class EvolvedScheduling(SchedulingPolicy):
    """Seed: Continuum's queue order. Turns whose context is still
    resident (kv_protected) go before turns that must re-prefill; within
    each class, program-level FCFS by first arrival. victim() and admit()
    are the engine defaults."""

    engine_args = {"scheduling_policy": "priority"}
    _CLASS = 1 << 40

    def __init__(self):
        self.epoch = None

    def priority(self, pcb, now):
        if self.epoch is None:
            self.epoch = now
        arrival = pcb.arrival_ts if pcb.arrival_ts is not None else now
        pinned = 0 if pcb.kv_protected else 1
        return pinned * self._CLASS + int(round((arrival - self.epoch) * 1000))

    def victim(self, candidates, now):
        # Engine default: preempt the running request with the largest
        # priority value. Return an index into candidates to override.
        return None

    def admit(self, pcb, now, view):
        # Engine default: every waiting turn is offered to the engine.
        return True
# EVOLVE-BLOCK-END
