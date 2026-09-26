"""A worked example of a policy that spans all three axes from one state.

Not a proposal and not tuned -- it exists so that "how would someone write a
unified policy" has an answer you can read and run. Load it with:

    python -m serving --planes program --kv-pool-tokens N \\
        --policy policies.example_unified:ContextAwarePolicy

What makes it unified rather than three policies in a trenchcoat is the shared
dictionary `self._value`. Each axis reads the SAME per-program valuation, so
the three decisions cannot disagree about which programs matter:

  * retention pins the context of a program it considers valuable
  * scheduling gives that program a better priority, and holds a low-value
    program at the gate when the pool is tight
  * routing sends its next turn back to the instance holding that context

Split across three objects, each would have to recompute the valuation from the
PCB, and the moment one of them used a slightly different rule the three knobs
would be pulling in different directions for reasons no counter would show.
That is the failure a unified policy is meant to prevent, and it is why sharing
a FILE (the evolve harness's `evolved_joint.py` shim) is not the same thing as
sharing an OBJECT.

Only the methods it defines are attached. It happens to define all three axes;
a policy that defined two would keep the engine's own rule for the third.
"""
from typing import Optional

from .program import ProgramControlBlock


class ContextAwarePolicy:
    """Value a program by the context it has built, act on that value thrice."""

    # Engine-launch configuration, read by the harness the same way the
    # published values' is. Prefix caching must be on or there is nothing
    # to retain.
    engine_flags = {"enable_prefix_caching": True, "kv_protection": True}
    engine_args: dict = {}
    release_event = "arrival"

    def __init__(self, pin_s: float = 8.0, hold_above: float = 0.9) -> None:
        self.pin_s = pin_s
        self.hold_above = hold_above
        #: program_id -> the valuation every axis reads. The whole point.
        self._value = {}

    # ------------------------------------------------------------ shared

    def _revalue(self, pcb: ProgramControlBlock) -> float:
        """How much this program's resident context is worth keeping.

        More turns completed means more context built and more still to come;
        a slow tool means the context sits idle a long time and is worth less
        per byte held. Deliberately simple -- the shape of the rule is not the
        point here, the fact that ONE rule feeds three decisions is.
        """
        gap = pcb.tool_mean_gap_s
        idle_penalty = 1.0 if gap is None else 1.0 / (1.0 + gap)
        v = (1 + pcb.turns_completed) * idle_penalty
        self._value[pcb.program_id] = v
        return v

    # --------------------------------------------------------- retention

    def on_turn_complete(self, pcb: ProgramControlBlock, request_id: str,
                         now: float) -> Optional[tuple]:
        v = self._revalue(pcb)
        if v < 1.0:
            # Low value: say so explicitly rather than staying silent. A
            # mechanism check has to tell "decided to do nothing" from "no
            # decision was made".
            return ("none", None, {"value": v})
        return ("protect", now + self.pin_s, {"value": v})

    def on_turn_arrival(self, pcb: ProgramControlBlock,
                        now: float) -> Optional[str]:
        return "release"

    # -------------------------------------------------------- scheduling

    def priority(self, pcb: ProgramControlBlock, now: float) -> Optional[int]:
        v = self._value.get(pcb.program_id)
        if v is None:
            return None            # never seen: no opinion, engine order
        # Lower number = served earlier, matching the engine's convention.
        return -int(v * 100)

    def admit(self, pcb: ProgramControlBlock, now: float, view) -> bool:
        if view.kv_utilization < self.hold_above:
            return True
        # Pool is tight: let through only what this policy already values, and
        # never hold a turn whose context is mostly resident already -- that
        # one is cheap to run and holding it wastes the hit.
        if view.prompt_tokens and view.cached_tokens / view.prompt_tokens > 0.5:
            return True
        return self._value.get(pcb.program_id, 0.0) >= 1.0

    # ----------------------------------------------------------- routing

    def route(self, pcb: ProgramControlBlock, now: float):
        home = pcb.kv_instance
        if home is None:
            return self._least_loaded(), {"pin": "new"}
        if self._value.get(pcb.program_id, 0.0) >= 1.0:
            # Valuable context: go back to it. This is the sentence the three
            # axes exist to say together -- it was pinned BECAUSE the next turn
            # is coming back here.
            return home, {"pin": "affinity", "value": self._value[pcb.program_id]}
        return self._least_loaded(), {"pin": "rebalance"}
