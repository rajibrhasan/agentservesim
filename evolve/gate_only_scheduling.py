"""Gate-only scheduling policy for the real-serving check (harness drop-in
``harness/evolved_scheduling.py``). Exactly the EvolvedScheduling class of
evolve/ablation_gate_only.py: pinned-first program-FCFS priority (the seed
scheduler) plus the admission gate found by the joint search
(evolve/champion_joint_41008503.py). Pair with --retention ttl
--retention-tau 2 to reproduce the simulator's ablation_gate_only tuple."""

from harness.scheduling import SchedulingPolicy


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
        # ABLATION (gate only): the champion_joint_41008503 admission
        # gate, verbatim, on top of the unchanged seed. Hold a turn one
        # tick when the pool is tight and its uncached prompt cannot fit
        # in free + evictable blocks (i.e. admitting it would break
        # other programs' pins).
        if pcb.kv_protected:
            return True
        util = view.kv_utilization
        if util is not None and util > 0.9:
            free = view.kv_free_tokens or 0
            evictable = view.kv_evictable_tokens or 0
            prompt = view.prompt_tokens or pcb.context_tokens or 0
            cached = view.cached_tokens or 0
            needed = max(0, prompt - cached)
            if needed > free + evictable:
                return False
        return True
