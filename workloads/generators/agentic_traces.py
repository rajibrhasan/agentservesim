#!/usr/bin/env python3
"""Build per-JPS agentic sub_requests traces from collected trajectories.

Supersedes the old E5 pipeline (make_e5_traces.py + e5_to_subrequests.py)
whose 16k context handling predated the current profiling reach:
- SWE-bench was postprocessed with --max-input-toks 16000 (per-turn tail
  truncation; 23/54 programs clipped, 869 turns dropped) and BFCL turns
  were tail-capped at 16384 ids.
- Here: NO per-turn capping. Programs whose largest context
  (input_toks + output_toks) exceeds --max-ctx (default 131072, the
  Llama-3.1-8B max-model-len) are skipped whole and reported, keeping
  every kept program's turns intact.

Arrivals are a seeded Poisson process per JPS (exponential
inter-arrivals, mean 1/jps, random.Random(seed)), programs in source
order.

Output rows (consumed natively by both sides of the validation:
``python -m bench run`` chain mode and ``python -m serving``):
    {"session_id", "arrival_time_ns",
     "sub_requests": [{"input_toks", "output_toks", "tool_duration_ns",
                       "input_tok_ids": [int...]}, ...]}

Usage:
  python workloads/generators/agentic_traces.py \
    --source <trajectories.jsonl> --name swebench \
    --out-dir workloads/swebench --jps 0.02 0.04 0.06 0.08 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NS_PER_MS = 1_000_000


def convert_program(obj: dict, streaming: bool = False) -> dict:
    turns = sorted(obj["turns"], key=lambda t: t["turn_idx"])
    n = len(turns)
    sub_requests = []
    for i, t in enumerate(turns):
        if streaming and "stream_delta_tok_ids" not in t:
            raise ValueError('streaming conversion needs explicit stream_delta_tok_ids; '
                             'regenerate from original tool messages')
        ids = t["stream_delta_tok_ids"] if streaming else t["input_tok_ids"]
        tool_ms = int(t.get("tool_ms", 0)) if i < n - 1 else 0
        # Fix output_toks=0 turns (ported from the old make_e5_traces.py).
        # Those are tool-call turns where the collection model returned
        # tool_calls instead of free-form text; the retokenizer only counts
        # message["content"] so the count comes out 0 even though 100-250
        # tokens were emitted. Fall back to collection_output_toks (the
        # collection model's own count; minor tokenizer drift but far
        # closer than clamp-to-1), capped at 2048 to neutralize corrupted
        # upstream counts (~60-82k on runaway turns) that would otherwise
        # trigger vLLM max-model-len rejections.
        out_toks = int(t["output_toks"])
        if out_toks < 1:
            fallback = int(t.get("collection_output_toks", 0))
            out_toks = max(1, min(fallback, 2048))
        sub_requests.append({
            "tool": t.get("tool"),
            "input_toks": len(ids),
            "output_toks": out_toks,
            "tool_duration_ns": tool_ms * NS_PER_MS,
            "input_tok_ids": ids,
        })
    result = {"session_id": obj["program_id"], "sub_requests": sub_requests}
    if streaming:
        result['input_mode'] = 'streaming-deltas'
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Collected trajectories JSONL (program/turns schema)")
    ap.add_argument("--name", required=True,
                    help="Workload name used in output filenames")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jps", type=float, nargs="+",
                    default=[0.02, 0.04, 0.06, 0.08, 0.1])
    ap.add_argument("--max-ctx", type=int, default=131072,
                    help="Skip programs whose largest context "
                         "(input+output tokens of any turn) exceeds this "
                         "(default: 131072 = Llama-3.1-8B max-model-len)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument('--streaming', action='store_true',
                    help='Use explicit tool deltas for live streaming sessions; '
                         'creates a different workload from full-prompt replay')
    args = ap.parse_args()

    kept, skipped = [], []
    with open(args.source) as f:
        for line in f:
            s = convert_program(json.loads(line), streaming=args.streaming)
            lengths = [t["input_toks"] + t["output_toks"] for t in s["sub_requests"]]
            worst = sum(lengths) if args.streaming else max(lengths)
            (kept if worst <= args.max_ctx else skipped).append(
                (s, worst))
    print(f"{args.source}: kept {len(kept)} programs, "
          f"skipped {len(skipped)} with max ctx > {args.max_ctx}")
    for s, worst in skipped:
        print(f"  skipped {s['session_id']} (max ctx {worst})")

    sessions = [s for s, _ in kept]
    n = len(sessions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for jps in args.jps:
        rng = random.Random(args.seed)
        t = 0.0
        rows = []
        for s in sessions:
            t += rng.expovariate(jps)
            rows.append({**s, "arrival_time_ns": int(t * 1e9)})
        # JSONL sorted by arrival (source order is arrival order here,
        # but keep the sort explicit for safety).
        rows.sort(key=lambda r: r["arrival_time_ns"])
        jtag = f"{jps:g}"
        name = args.name + ('_streaming' if args.streaming else '')
        path = out_dir / f"{name}_jps{jtag}_n{n}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        span = rows[-1]["arrival_time_ns"] / 1e9
        turns = sum(len(r["sub_requests"]) for r in rows)
        print(f"wrote {path}  ({n} programs, {turns} turns, "
              f"arrival span {span:.0f}s)")


if __name__ == "__main__":
    main()
