#!/usr/bin/env python3
"""Convert SWE-bench leaderboard mini-swe-agent trajectories (format
``mini-swe-agent-1.1``, per-message ``extra.timestamp`` + API usage) into
the program/turns trajectories JSONL that ``agentic_traces.py`` consumes.

Source: the public swe-bench-submissions S3 bucket (anonymous GET), e.g.
``bash-only/20260217_mini-v2.0.0_claude-4-5-sonnet-high/trajs/...``.

Per assistant turn i (1-indexed over assistant messages):
  input_tok_ids  — Llama retokenization of the accumulated conversation
                   up to (excluding) that assistant message. Built
                   incrementally (each message tokenized once with
                   add_special_tokens=False, concatenated after a role
                   header line), so turn i's ids are a strict prefix
                   extension of turn i-1's — required for the simulator's
                   prefix caching to see the real reuse pattern.
  output_toks    — the API-reported completion_tokens for that call
                   (includes reasoning tokens: real decode work). The
                   visible assistant text, not the reasoning, is what
                   enters the next turn's context — matching production
                   serving of reasoning models.
  tool_ms        — last tool message timestamp minus this assistant
                   message timestamp: measured wall-clock tool execution
                   time, excluding the next API call's queue/decode time
                   (which the simulator regenerates).

Message content replayed as-is (mini's observation truncation included):
it is exactly what the agent sent to the API, and the API-reported
prompt_tokens confirm it.

Usage:
  python workloads/generators/convert_leaderboard_trajs.py \
    --traj-dir /path/to/trajs --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --out trajectories.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def convert(path: str, tok) -> dict | None:
    with open(path) as f:
        d = json.load(f)
    fmt = d.get("trajectory_format", "")
    if not fmt.startswith("mini-swe-agent-1"):
        print(f"  SKIP {os.path.basename(path)}: format {fmt!r}")
        return None
    msgs = d["messages"]
    ids: list[int] = []
    turns = []
    i = 0
    # leading system/user messages
    while i < len(msgs) and msgs[i]["role"] != "assistant":
        ids += _msg_ids(tok, msgs[i])
        i += 1
    turn_idx = 0
    stream_delta = None
    while i < len(msgs):
        m = msgs[i]
        assert m["role"] == "assistant", m["role"]
        ex = m.get("extra", {})
        usage = (ex.get("response") or {}).get("usage") or {}
        out_toks = int(usage.get("completion_tokens") or 0)
        a_ts = ex.get("timestamp")
        j = i + 1
        last_tool_ts = None
        tail = []
        while j < len(msgs) and msgs[j]["role"] != "assistant":
            t = msgs[j].get("extra", {}).get("timestamp")
            if t:
                last_tool_ts = t
            tail.append(msgs[j])
            j += 1
        tool_ms = 0
        if a_ts and last_tool_ts and last_tool_ts >= a_ts:
            tool_ms = int((last_tool_ts - a_ts) * 1000)
        turns.append({
            "turn_idx": turn_idx,
            "input_tok_ids": list(ids),
            "output_toks": 0,               # force API fallback below
            "collection_output_toks": out_toks,
            "tool_ms": tool_ms,
            "tool": _tool_label(m),
            "stream_delta_tok_ids": list(ids) if stream_delta is None else stream_delta,
        })
        turn_idx += 1
        # context grows by the visible assistant text + tool observations
        ids += _msg_ids(tok, m)
        stream_delta = []
        for t in tail:
            delta = _msg_ids(tok, t)
            ids += delta
            stream_delta += delta
        i = j
    if not turns:
        return None
    return {"program_id": d.get("instance_id") or
            os.path.basename(path).split(".")[0],
            "turns": turns}


_SKIP_FIRST = {"cd", "export", "env", "sudo", "nohup", "time"}


def _tool_label(m: dict):
    """Tool name for the turn: basename of the first command word of the
    bash action, skipping a leading cd/export/env segment (the rule used
    for the collected SWE-bench traces: 24-26 distinct labels such as
    sed, grep, cat, python, git, find, pip). None when the assistant
    message carries no bash action."""
    cmd = None
    for tc in (m.get("tool_calls") or []):
        args = (tc.get("function") or {}).get("arguments") or ""
        try:
            obj = json.loads(args) if isinstance(args, str) else args
            cmd = obj.get("command") if isinstance(obj, dict) else None
        except Exception:
            cmd = args if isinstance(args, str) else None
        if cmd:
            break
    if not cmd:
        return None
    for seg in [x for part in cmd.split("\n") for x in part.split("&&")]:
        toks = seg.strip().split()
        while toks and ("=" in toks[0] and not toks[0].startswith("=")):
            toks = toks[1:]                      # VAR=value prefixes
        if not toks or toks[0] in _SKIP_FIRST:
            continue
        return os.path.basename(toks[0])
    return None


def _msg_ids(tok, m: dict) -> list[int]:
    content = m.get("content") or ""
    if isinstance(content, list):    # anthropic-style content blocks
        content = "".join(c.get("text", "") for c in content
                          if isinstance(c, dict))
    # Tool-calling turns (mini v2 uses them for most bash actions) carry
    # the command in tool_calls[].function.arguments with content=None —
    # that text is real context (heredoc file writes can be huge).
    # Reasoning/thinking_blocks are deliberately NOT appended: a Llama
    # deployment would not replay Claude thinking into context.
    for tc in (m.get("tool_calls") or []):
        content += (tc.get("function") or {}).get("arguments") or ""
    return tok.encode(f"{m['role']}: {content}\n", add_special_tokens=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--tokenizer",
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)
    files = sorted(glob.glob(os.path.join(a.traj_dir, "*.traj.json")))
    print(f"{len(files)} trajectories in {a.traj_dir}")
    n = 0
    maxctx = []
    with open(a.out, "w") as f:
        for p in files:
            prog = convert(p, tok)
            if prog is None:
                continue
            f.write(json.dumps(prog) + "\n")
            n += 1
            maxctx.append(max(len(t["input_tok_ids"]) +
                              t["collection_output_toks"]
                              for t in prog["turns"]))
            if n % 50 == 0:
                print(f"  {n} done", flush=True)
    print(f"wrote {n} programs -> {a.out}")
    if maxctx:
        s = sorted(maxctx)
        print(f"max ctx (Llama ids + out): p50={s[len(s)//2]:,} "
              f"p90={s[int(.9 * len(s))]:,} max={s[-1]:,} "
              f"mean={st.fmean(maxctx):,.0f}")


if __name__ == "__main__":
    main()
