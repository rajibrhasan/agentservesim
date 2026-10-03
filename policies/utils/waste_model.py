

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WasteProfile:
    a: float  # ms per token (T_fwd slope)
    c: float  # ms (T_fwd intercept)
    S: int    # saturation knee, tokens per forward pass

    @classmethod
    def from_json(cls, path: str) -> "WasteProfile":
        with open(path) as f:
            d = json.load(f)
        return cls(a=float(d["a"]), c=float(d["c"]), S=int(d["S"]))


def t_fwd_s(profile: WasteProfile, num_tokens: int, c: float | None = None) -> float:
    """Forward-pass time in seconds for a batch of num_tokens tokens."""
    if c is None:
        c = profile.c
    return (profile.a * num_tokens + c) / 1000.0


def preserve_waste(ctx_tokens: int, gap_s: float) -> float:
    """InferCept Eq. 2: token*seconds of KV occupancy if the context is
    retained on the GPU for the (predicted) gap."""
    return max(0.0, gap_s) * ctx_tokens


def discard_waste(
    ctx_tokens: int,
    inflight_tokens: int,
    running_ctx_tokens: int,
    profile: WasteProfile,
) -> float:
    """InferCept Eq. 4 (WasteChunkDiscard): token*seconds wasted if the
    context is evicted now and re-prefilled at the next turn in chunks
    of c_h = max(S - inflight_tokens, 1), including the slowdown
    imposed on the currently running batch (running_ctx_tokens)."""
    c_h = max(profile.S - inflight_tokens, 1)
    n = max((ctx_tokens + c_h - 1) // c_h - 1, 0)
    f_ch = profile.a * c_h / 1000.0
    f_s = t_fwd_s(profile, profile.S)
    w_self = f_s * (1 + n) * n / 2 * c_h
    w_other = f_ch * n * running_ctx_tokens
    last_toks = ctx_tokens - n * c_h
    if last_toks > 0:
        w_other += t_fwd_s(profile, last_toks, c=0.0) * running_ctx_tokens
    return w_self + w_other


#: Cluster-config hardware name -> the tag used in profile filenames.
_HW_TAG = {"RTXPRO6000": "rtx6000", "B200": "b200", "L4": "l4",
           "H100": "h100", "A100": "a100"}


def resolve_profile(hardware: str, model: str, tp_size: int,
                    profiles_dir: str | None = None) -> str:
    """Path to the InferCept waste profile measured on THIS hardware."""
    if profiles_dir is None:
        profiles_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "profiles")
    tag = _HW_TAG.get(hardware, str(hardware).lower())
    size = model.rsplit("-", 1)[-1] if "-" in model else model   # Llama-3.1-70B -> 70B
    name = f"infercept_profile_{tag}_{size}_tp{int(tp_size)}.json"
    path = os.path.join(profiles_dir, name)
    if os.path.exists(path):
        return path
    have = sorted(f for f in os.listdir(profiles_dir)
                  if f.startswith("infercept_profile_")) if os.path.isdir(
                      profiles_dir) else []
    raise FileNotFoundError(
        f"no InferCept waste profile for hardware={hardware} model={model} "
        f"tp={tp_size} (looked for {name}). Profile this platform with "
        f"experiments/validation/infercept_sweep.py, or pass "
        f"--min-waste-profile explicitly. Available: {have}")
