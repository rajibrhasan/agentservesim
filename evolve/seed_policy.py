"""Policy candidate for the automated search.

This file is the unit the search mutates. Only the region between the
EVOLVE-BLOCK markers changes; everything outside it is the fixed contract.

One class per plane, in one file -- the way `policies/continuum.py` and
`policies/stock.py` are written, and the way every recorded champion is
written:

    class EvolvedRetention(KVPolicy)            decides retention
    class EvolvedScheduling(SchedulingPolicy)   decides scheduling
    class EvolvedRouting(RoutingPolicy)         decides routing

There is no axis to declare. A plane is decided by its class being present;
whatever is absent falls back to the engine's own rule, so deleting a class is
a legal edit. The seed below defines retention and scheduling; routing is left
to the engine.

The classes do not see each other. Where they interact, they do it through the
PCB -- the scheduler reads `pcb.kv_protected`, which the retention policy set.
If a policy genuinely needs one piece of state across two planes, name the same
class on both flags and it is built once.

RETENTION -- on_turn_complete(pcb, request_id, now) -> None | (action, deadline)
    Called when a turn finishes and the program enters its tool gap.
    "protect": hold this program's KV until deadline (absolute seconds).
    "evict":   drop it now; the next turn re-prefills.
    None:      let the blocks compete in the ordinary LRU queue.
  on_turn_arrival(pcb, now) -> "release" | None
    Called when the next turn arrives. "release" ends a standing protection.

SCHEDULING -- priority(pcb, now) -> int | None
    Queue order for a waiting turn; LOWER runs first. None leaves it unstamped.
  victim(candidates, now) -> int | None
    Under memory pressure, which RUNNING request to preempt (index into
    candidates: VictimView with pcb, priority, prompt_tokens, computed_tokens,
    generated_tokens, is_prefill). None = engine default.
  admit(pcb, now, view) -> bool
    Admission gate, in queue order each tick. False holds the turn this tick;
    it keeps its place. view is a QueueView: n_running, n_waiting, n_inflight,
    kv_utilization, kv_free_tokens, kv_evictable_tokens, prompt_tokens,
    cached_tokens.

ROUTING -- route(pcb, now) -> int | (int, dict) | None
    Which instance the turn runs on. None = engine default.

Observation boundary: `now` (seconds) and the PCB:
  pcb.turn_idx, pcb.turns_completed, pcb.arrival_ts, pcb.attained_service_s,
  pcb.context_tokens, pcb.tool_name, pcb.tool_mean_gap_s, pcb.in_gap,
  pcb.gap_started_ts, pcb.gap_elapsed_s(now), pcb.kv_protected,
  pcb.kv_deadline_ts, pcb.kv_instance
Nothing about the future (output length, this gap's duration, remaining turns).
Reading pcb.program_id or pcb.kv_request_id is forbidden, and so are
per-program tables inside the policy; aggregate statistics such as a running
mean of gaps by tool name are allowed.

Fitness: mean program JCT of the stock configuration (LRU cache, FCFS) divided
by mean program JCT under this candidate, on a saturated cell and then a second
arrival rate; the score is the mean over the cells run so far, the minimum is
reported alongside, and vs_seed reports the same ratio against this seed.
Higher is better.
"""

from policies.base import KVPolicy, SchedulingPolicy


# EVOLVE-BLOCK-START
class EvolvedRetention(KVPolicy):
    """Seed: fixed-horizon protection (the published TTL policy).

    Protect for TAU seconds after every turn, release on the next arrival. The
    horizon is a constant; it does not depend on the tool, the context size, or
    what the policy has observed so far.
    """

    TAU_S = 2.0

    def __init__(self):
        # Aggregate state only (no per-program tables): running statistics of
        # observed gaps by tool name are allowed here.
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
    """Seed: Continuum pinned-first program-FCFS (arXiv:2511.02230), the best
    measured policy on the leaderboard cells.

    Two-level order: turns of programs whose KV is still resident go before
    turns that must re-prefill; within each class, program-level FCFS by first
    arrival. A resident context finishes its turn without paying re-prefill, so
    serving it first frees the pool sooner; program seniority bounds
    starvation.

    Note what couples this to the retention class above: `pcb.kv_protected` is
    set by whatever retention policy is running. The two planes meet through
    the PCB, not through each other -- neither class can see the other's state,
    and neither needs to.
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

    # victim() and admit() are left to the engine: define them to take over
    # preemption choice and the admission gate.
# EVOLVE-BLOCK-END
