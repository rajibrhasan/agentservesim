"""Gap-start preserve/discard policy and optional simulator copy enqueue.

The native InferCept iteration controller lives in bench.core.infercept_scheduler.
This callback alone is not a full InferCept implementation. It must never
allocate a transfer budget from a forward pass that has already completed.
"""
from typing import Callable, Optional

from .base import KVPolicy
from .utils import waste_model
from .utils.waste_model import WasteProfile

PAPER = "InferCept 2402.01869"


class InferceptKV(KVPolicy):
    """InferCept-style min-waste retain-vs-evict callback: at each
    gap start, retain (protect) if the predicted occupancy waste of
    keeping the context resident is below the recompute waste of
    re-prefilling it, else evict now.

    Wastes come from waste_model (Eq. 2 vs Eq. 4) with a measured
    hardware profile. The gap prediction comes from the per-tool
    predictor (fallback: default_gap_s). load_probe, when given,
    returns (inflight_tokens, running_ctx_tokens) for the discard
    side; without it the model assumes an idle engine, which makes
    recompute cheapest and eviction most likely (conservative for
    retention). All inputs and both scores are logged in the
    decision's info dict, so the simulator mirror replays the
    comparison. Native session swapping is implemented by the engine
    controller, independently of this callback.

    The protect deadline is now + predicted gap: past it the blocks
    stay hit-able but become first reclaim candidates under pressure
    (expired-first valve), which degrades gracefully when the gap
    prediction was short.
    """

    @classmethod
    def from_config(cls, cfg):
        from .utils.waste_model import WasteProfile
        return cls(WasteProfile.from_json(cfg.min_waste_profile),
                   default_gap_s=cfg.default_gap_s)

    def __init__(
        self,
        profile: "WasteProfile",
        default_gap_s: float,
        predictor: Optional[Callable[[Optional[str]], Optional[float]]] = None,
        load_probe: Optional[Callable[[], tuple[int, int]]] = None,
    ) -> None:
        self.profile = profile
        self.default_gap_s = default_gap_s
        self.predictor = predictor
        self.load_probe = load_probe
        self.defer_swap = False

    def on_turn_complete(self, pcb, request_id, now):
        context_tokens = pcb.context_tokens
        if not context_tokens:
            raise ValueError(
                "min-waste requires context_tokens per turn (PCB has "
                f"context_tokens={context_tokens!r} for {pcb.program_id})"
            )
        gap_s = None
        if self.predictor is not None:
            gap_s = self.predictor(pcb.tool_name)
        if gap_s is None:
            gap_s = self.default_gap_s
        inflight, running_ctx = (0, 0)
        if self.load_probe is not None:
            inflight, running_ctx = self.load_probe()
        w_p = waste_model.preserve_waste(context_tokens, gap_s)
        w_d = waste_model.discard_waste(
            context_tokens, inflight, running_ctx, self.profile
        )
        info = {
            "gap_pred_s": gap_s,
            "context_tokens": context_tokens,
            "inflight_tokens": inflight,
            "running_ctx_tokens": running_ctx,
            "w_preserve": w_p,
            "w_discard": w_d,
        }
        if self.defer_swap:
            # This callback only enqueues a source. The upcoming iteration
            # owns bandwidth allocation and completion accounting.
            info['transfer_pending'] = True
            return ('swap', None, info)
        if w_p <= w_d:
            return ("protect", now + gap_s, info)
        return ("evict", None, info)

    def on_turn_arrival(self, pcb, now):
        return "release"


def make_kv(min_waste_profile=None, default_gap_s=1.0, **_):
    """InferCept needs a measured waste profile, not a scalar knob.

    Here rather than in the registry because a paper's construction is the
    paper's business: the registry should not have to know that this one reads
    a JSON file while Continuum takes a float.
    """
    if min_waste_profile is None:
        raise ValueError(
            "infercept needs --min-waste-profile: its decision is a comparison "
            "against measured recompute and occupancy costs, and there is no "
            "default that would mean anything.")
    return InferceptKV(WasteProfile.from_json(min_waste_profile),
                       default_gap_s=default_gap_s)
