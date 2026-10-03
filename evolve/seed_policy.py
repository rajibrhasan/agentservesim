"""Starting policy for joint retention and scheduling search.

Edit only the EVOLVE-BLOCK region. Each Evolved* class controls its named
plane; omitted planes retain engine defaults. Callbacks receive observed
program state and time in seconds. Retention returns ("protect", deadline),
("evict", None), or None; arrival may return "release". Lower scheduling
priority runs first. victim() returns a candidate index or None; admit()
returns whether to admit the call. Future information and program identities
are forbidden. Fitness is mean stock JCT / candidate JCT across search cells."""

from policies.base import KVPolicy, SchedulingPolicy


# EVOLVE-BLOCK-START
class EvolvedRetention(KVPolicy):
    """Seed: fixed-horizon protection (the published TTL policy)."""

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
    """Seed: Continuum pinned-first program-FCFS (arXiv:2511.02230), the best measured policy on the leaderboard cells."""

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
