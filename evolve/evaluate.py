"""OpenEvolve evaluator with staged simulation and candidate-result caching.

Fitness is mean stock JCT / candidate JCT across evaluated cells.
Candidates must pass the static observation-boundary check."""

import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sandbox  # noqa: E402
import simrun   # noqa: E402

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # grid runner / tests without openevolve installed
    class EvaluationResult:
        def __init__(self, metrics, artifacts=None):
            self.metrics, self.artifacts = metrics, artifacts or {}

CACHE_DIR = os.environ.get(
    "EVOLVE_CACHE", os.path.join(simrun.SCRATCH, "cache"))
TIMEOUT = 5400  # s per cell; stock on rtx70b_swe50_j0.02 takes ~55 min
FAIL = {"combined_score": 0.0, "protect_frac": 0.0, "evict_frac": 0.0,
        "p99_ratio": 0.0, "jct_ratio_min": 0.0, "cells": 0.0}


def _normalized_source(src):
    """The candidate with docstrings, comments, and formatting removed:
    two candidates that differ only cosmetically get the same key, so a
    cosmetic edit costs no simulation and is reported as a duplicate."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
                if not body:
                    body.append(ast.Pass())
    return ast.dump(tree, include_attributes=False)


def _hash(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        key = _normalized_source(raw.decode("utf-8"))
    except UnicodeDecodeError:
        key = raw
    if isinstance(key, str):
        key = key.encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16]


# The seed's scorecards (same cache, keyed by its normalized hash) give a
# second reported ratio: seed JCT / candidate JCT, so the log shows the
# gain over the starting policy and not only over stock.
SEED_PATH = os.environ.get("EVOLVE_SEED_PATH")


def _seed_cards():
    if not SEED_PATH or not os.path.exists(SEED_PATH):
        return {}
    return _cache_get(_hash(SEED_PATH))


def _cache_get(h):
    p = os.path.join(CACHE_DIR, h + ".json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _cache_put(h, cards):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, h + ".json"), "w") as f:
        json.dump(cards, f, indent=1, sort_keys=True)


def _run_cells(program_path, cells):
    """Scorecards for the given cells, from cache where present."""
    h = _hash(program_path)
    cards = _cache_get(h)
    todo = [c for c in cells if c not in cards]
    duplicate = bool(cards) and not todo
    if todo:
        workdir = tempfile.mkdtemp(prefix="hr_", dir=simrun.SCRATCH)
        try:
            root = simrun.stage_harness(program_path, workdir)
            for c in todo:
                cards[c] = simrun.run_cell(
                    c, harness_root=root,
                    policy_flags=simrun.flags_for(program_path),
                    timeout=TIMEOUT)
                _cache_put(h, cards)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    out = {c: cards[c] for c in cells}
    return out, duplicate


def _score(cards):
    """Metrics + artifacts from a set of cell scorecards."""
    ratios, p99s, table = [], [], []
    dec = {"protect": 0, "evict": 0, "none": 0, "release": 0}
    for c, card in cards.items():
        if card.get("error") or card.get("jct_mean") is None:
            return FAIL, {"failure_cell": c, "error": card.get("error"),
                          "stderr": card.get("stderr_tail", "")}
        base = simrun.baseline(c, compute=False)
        r = base["jct_mean"] / card["jct_mean"]
        r99 = base["jct_p99"] / card["jct_p99"] if card["jct_p99"] else 0.0
        ratios.append(r); p99s.append(r99)
        for k in dec:
            dec[k] += (card.get("decisions") or {}).get(k, 0)
        table.append(f"{c:22s} stock={base['jct_mean']:8.3f}s cand={card['jct_mean']:8.3f}s "
                     f"ratio={r:.4f} p99_ratio={r99:.4f} wall={card['wall_s']}s")
    n_turn = sum(dec.values()) or 1
    metrics = {
        # Upper clamp only guards against a degenerate divide; the per-tool
        # Continuum seed already scores 2.4 on the collected cells, so the
        # old cap of 2.0 would have made every stronger candidate look equal.
        "combined_score": max(0.0, min(10.0, statistics.fmean(ratios))),
        "jct_ratio_min": min(ratios),
        "p99_ratio": statistics.fmean(p99s),
        "protect_frac": dec["protect"] / n_turn,
        "evict_frac": dec["evict"] / n_turn,
        "cells": float(len(ratios)),
    }
    seed = _seed_cards()
    vs = [seed[c]["jct_mean"] / card["jct_mean"] for c, card in cards.items()
          if c in seed and seed[c].get("jct_mean") and card.get("jct_mean")]
    if vs:
        metrics["vs_seed"] = statistics.fmean(vs)
        metrics["vs_seed_min"] = min(vs)
    artifacts = {"cells": "\n".join(table),
                 "decisions": json.dumps(dec)}
    if vs:
        artifacts["vs_seed"] = " ".join(f"{c}={v:.4f}" for c, v in
                                        zip([c for c in cards if c in seed], vs))
    return metrics, artifacts


def _stage(program_path, cells):
    ok, reasons = sandbox.check_file(program_path)
    if not ok:
        # Keep every rejected candidate with its reasons: the only place the
        # search's failure modes are visible before a checkpoint.
        rej = os.path.join(os.environ.get("EVOLVE_CACHE", "."), "rejections")
        os.makedirs(rej, exist_ok=True)
        with open(os.path.join(rej, f"{_hash(program_path)}.py"), "w") as f:
            f.write("# REJECTED: " + "; ".join(reasons) + "\n" + open(program_path).read())
    if not ok:
        return EvaluationResult(
            metrics=dict(FAIL, sandbox=0.0),
            artifacts={"sandbox": "rejected: " + "; ".join(reasons)})
    cards, duplicate = _run_cells(program_path, cells)
    metrics, artifacts = _score(cards)
    metrics["sandbox"] = 1.0
    if duplicate:
        artifacts["duplicate"] = (
            "behaviourally identical to a program already evaluated "
            "(same code after removing comments, docstrings, and "
            "formatting); no simulation was run. Change the decision "
            "rule, not its presentation.")
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_stage1(program_path):
    if not simrun.STAGE1:
        raise SystemExit(
            "nothing to score on. Set EVOLVE_DATASET and EVOLVE_CLUSTER_CONFIG "
            "to the same two files you would pass `python -m serving` as "
            "--dataset and --cluster-config.")
    return _stage(program_path, simrun.STAGE1)


def evaluate_stage2(program_path):
    return _stage(program_path, simrun.STAGE1 + simrun.STAGE2)


def evaluate_stage3(program_path):
    return _stage(program_path, simrun.STAGE1 + simrun.STAGE2 + simrun.STAGE3)


def evaluate(program_path):
    return evaluate_stage3(program_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    a = ap.parse_args()
    fn = {1: evaluate_stage1, 2: evaluate_stage2, 3: evaluate_stage3}[a.stage]
    r = fn(a.program)
    print(json.dumps(r.metrics, indent=1))
    for k, v in r.artifacts.items():
        print(f"--- {k} ---\n{v}")
