"""Grid floor: the published retention policies over the evaluator's
cells, scored exactly as a search candidate would be. This is the bar
an evolved policy has to clear (paper-2 plan, M1 / M2 gate).

Points: cache-lru (stock, ratio 1.0 by construction), evict-always,
ttl at several horizons, min-waste with the measured B200 8B profile.
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simrun  # noqa: E402

def min_waste_flags(cell):
    """InferCept needs a forward-time profile measured on the cell's hardware.

    It is now derived from that cell's cluster config, so there is nothing to
    map here. This used to be a table of container paths keyed by cell-name
    PREFIX -- so a new cell silently got whichever profile its name happened to
    to match, or the B200 8B default, and was scored with another platform's
    forward times.
    """
    return ["--retention", "min-waste"]


GRID = {
    # agent tuple is ttl tau=2 (Continuum) + plas + session affinity;
    # tau sensitivity at 5 and 20. min-waste needs a per-(model,GPU)
    # InferCept profile; none exists for RTX 70B, so it is off the grid.
    "cache-lru":    ["--retention", "cache-lru"],
    "lru-plas":     ["--retention", "cache-lru", "--scheduling", "plas"],
    # evict-always exits at startup on the RTX config (2026-08-27); degenerate reference, dropped.
    "ttl-2":        ["--retention", "ttl", "--retention-tau", "2"],
    "ttl-5":        ["--retention", "ttl", "--retention-tau", "5"],
    "ttl-20":       ["--retention", "ttl", "--retention-tau", "20"],
    "ttl2-plas":    ["--retention", "ttl", "--retention-tau", "2", "--scheduling", "plas"],
    "continuum":    ["--retention", "continuum", "--scheduling", "continuum"],
    "ttl2-cont":    ["--retention", "ttl", "--retention-tau", "2", "--scheduling", "continuum"],
    "saga-ttl":     ["--retention", "saga-ttl", "--retention-tau", "2", "--scheduling", "continuum"],
    # SAGA (arXiv:2605.00528) as a unified policy on the two axes a
    # single instance exposes: pressure-scaled retention (Alg. 1, base
    # tau 2 s) plus program-level ordering. Its scheduling half is
    # "workflow-atomic, program-level, task-completion-time fair", which
    # maps to program-fcfs (workflow-atomic by arrival) or plas
    # (fairness by attained service); both are run. saga-fcfs isolates
    # the retention half under the stock queue. Work stealing and
    # session affinity are multi-instance and out of scope here.
    "saga-fcfs":     ["--retention", "saga-ttl", "--retention-tau", "2"],
    "saga-progfcfs": ["--retention", "saga-ttl", "--retention-tau", "2", "--scheduling", "program-fcfs"],
    "saga-plas":     ["--retention", "saga-ttl", "--retention-tau", "2", "--scheduling", "plas"],
}

# Factorial attribution grid (best-per-axis): full retention x scheduling
# cross product, so each axis's marginal best and the interaction term are
# identifiable. Names are "<retention>.<scheduling>"; some combos coincide
# with legacy GRID points (lru.fcfs == cache-lru, ttl2.plas == ttl2-plas) —
# the analysis can merge those results instead of re-running them.
_RET = {
    "lru":   ["--retention", "cache-lru"],
    "ttl2":  ["--retention", "ttl", "--retention-tau", "2"],
    "ttl5":  ["--retention", "ttl", "--retention-tau", "5"],
    "ttl20": ["--retention", "ttl", "--retention-tau", "20"],
    "saga":  ["--retention", "saga-ttl", "--retention-tau", "2"],
    "cont":  ["--retention", "continuum"],
}
_SCHED = {
    "fcfs":  [],
    "plas":  ["--scheduling", "plas"],
    "pfcfs": ["--scheduling", "program-fcfs"],
    "cont":  ["--scheduling", "continuum"],
}
FACTORIAL = {f"{rn}.{sn}": rf + sf
             for rn, rf in _RET.items() for sn, sf in _SCHED.items()}

# Clairvoyant headroom probes (sim-only; harness/oracle.py): each axis's
# bound alone, and both together.
ORACLE = {
    "oracle-ttl":  ["--retention", "oracle-ttl"],
    "oracle-srpt": ["--scheduling", "oracle-srpt"],
    "oracle-both": ["--retention", "oracle-ttl", "--scheduling", "oracle-srpt"],
}

GRID.update(FACTORIAL)
GRID.update(ORACLE)
FACTORIAL_POINTS = sorted(FACTORIAL)
ORACLE_POINTS = sorted(ORACLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*",
                    default=simrun.STAGE1 + simrun.STAGE2 + simrun.STAGE3 + simrun.HOLDOUT)
    ap.add_argument("--points", nargs="*", default=list(GRID))  # "min-waste" also accepted (profile per cell)
    ap.add_argument("--out", default=os.path.join(HERE, "grid_results.json"))
    a = ap.parse_args()
    results = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            results = json.load(f)
    for point in a.points:
        results.setdefault(point, {})
        for cell in a.cells:
            if cell in results[point] and not results[point][cell].get("error"):
                continue
            card = simrun.run_cell(cell, policy_flags=(min_waste_flags(cell) if point == "min-waste" else GRID[point]))
            results[point][cell] = card
            with open(a.out, "w") as f:
                json.dump(results, f, indent=1, sort_keys=True)
            try:
                base = simrun.baseline(cell, compute=False)
                r = (base["jct_mean"] / card["jct_mean"]) if card["jct_mean"] else float("nan")
            except KeyError:   # no stored stock baseline for this cell (cell scans)
                r = float("nan")
            print(f"{point:13s} {cell:20s} jct={card['jct_mean'] if card['jct_mean'] else 'ERR':>9} "
                  f"ratio={r:.4f} wall={card['wall_s']}s {card['error'] or ''}", flush=True)
    print("\n=== grid floor (mean ratio over cells; higher is better) ===")
    for point, cells in results.items():
        rs = []
        for cell, card in cells.items():
            if card.get("jct_mean"):
                try:
                    rs.append(simrun.baseline(cell, compute=False)["jct_mean"] / card["jct_mean"])
                except KeyError:
                    pass
        if rs:
            print(f"  {point:13s} mean={statistics.fmean(rs):.4f} min={min(rs):.4f} n={len(rs)}")


if __name__ == "__main__":
    main()
