#!/usr/bin/env python3
"""Derive perturbed agentic traces from existing sub_requests JSONLs.

Operating-point sweep support (docs/iclr-impact-sweep.md): take validated
traces and manufacture the preconditions for policy impact one knob at a
time, keeping token ids and per-turn shapes intact so the derived trace
stays grounded in the real workload.

Operations (composable, applied in this order):
  1. prefix selection      --source path[:N] (repeatable; first N programs)
  2. gap scaling           --gap-scale F     (tool_duration_ns *= F)
  3. tail amplification    --tail-amplify F --tail-quantile Q
                           (gaps strictly above the pooled Q-quantile of
                            positive gaps get *= F; the rest untouched)
  4. arrival redraw        --jps X --seed S  (seeded shuffle of the merged
                            program list, then Poisson arrivals; required
                            when >1 source, optional otherwise -- without
                            it embedded arrival_time_ns are kept)

With >1 source, session_ids get a per-source prefix (basename-derived)
unless they already carry one, mirroring workloads/mixed naming.

Example (mixture 1:4 at JPS 0.1):
  python workloads/generators/perturb_traces.py \
    --source workloads/swebench/swebench_jps0.1_n50.jsonl:10 \
    --source workloads/bfcl_v4/bfcl_jps0.1_n185.jsonl:40 \
    --jps 0.1 --seed 42 --out /orange/.../mix1to4_jps0.1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def load_prefix(spec: str):
    if ":" in spec and not spec.split(":")[-1].startswith("/"):
        path, n = spec.rsplit(":", 1)
        n = int(n)
    else:
        path, n = spec, None
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if n is not None and len(rows) == n:
                break
    if n is not None and len(rows) < n:
        raise SystemExit(f"{path}: asked for {n} programs, found {len(rows)}")
    # Prefix traces are only valid taken in arrival order.
    if any("arrival_time_ns" in r for r in rows):
        rows.sort(key=lambda r: r.get("arrival_time_ns", 0))
    tag = re.split(r"[_.]", Path(path).name)[0]
    return tag, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="trace JSONL, optionally path:N for the first N programs")
    ap.add_argument("--gap-scale", type=float, default=1.0)
    ap.add_argument("--tail-amplify", type=float, default=1.0)
    ap.add_argument("--tail-quantile", type=float, default=0.75)
    ap.add_argument("--jps", type=float, default=None,
                    help="redraw Poisson arrivals at this rate (required for >1 source)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    loaded = [load_prefix(s) for s in a.source]
    if len(loaded) > 1 and a.jps is None:
        raise SystemExit("mixtures need --jps: embedded arrivals from "
                         "different sources are not comparable")

    programs = []
    for tag, rows in loaded:
        for r in rows:
            r = dict(r)
            if len(loaded) > 1 and not r["session_id"].startswith(tag):
                r["session_id"] = f"{tag}__{r['session_id']}"
            programs.append(r)

    # Tail threshold over pooled positive gaps, computed before any scaling.
    thresh = None
    if a.tail_amplify != 1.0:
        gaps = sorted(s["tool_duration_ns"]
                      for p in programs for s in p["sub_requests"]
                      if s["tool_duration_ns"] > 0)
        if not gaps:
            raise SystemExit("no positive gaps to amplify")
        thresh = gaps[min(len(gaps) - 1, int(a.tail_quantile * (len(gaps) - 1)))]

    for p in programs:
        subs = []
        for s in p["sub_requests"]:
            s = dict(s)
            d = s["tool_duration_ns"]
            if thresh is not None and d > thresh:
                d = int(d * a.tail_amplify)
            s["tool_duration_ns"] = int(d * a.gap_scale)
            subs.append(s)
        p["sub_requests"] = subs

    rng = random.Random(a.seed)
    if a.jps is not None:
        rng.shuffle(programs)
        t = 0.0
        for p in programs:
            t += rng.expovariate(a.jps)
            p["arrival_time_ns"] = int(t * 1e9)
    programs.sort(key=lambda r: r.get("arrival_time_ns", 0))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in programs:
            f.write(json.dumps(p) + "\n")

    n = len(programs)
    turns = sum(len(p["sub_requests"]) for p in programs)
    pos = sorted(s["tool_duration_ns"] / 1e9
                 for p in programs for s in p["sub_requests"]
                 if s["tool_duration_ns"] > 0)
    span = programs[-1].get("arrival_time_ns", 0) / 1e9
    gp50 = pos[len(pos) // 2] if pos else 0.0
    gp90 = pos[int(0.9 * (len(pos) - 1))] if pos else 0.0
    print(f"wrote {out}  programs={n} turns={turns} arrival_span={span:.0f}s "
          f"gap_p50={gp50:.2f}s gap_p90={gp90:.2f}s"
          + (f" tail_thresh={thresh/1e9:.2f}s" if thresh is not None else ""))


if __name__ == "__main__":
    main()
