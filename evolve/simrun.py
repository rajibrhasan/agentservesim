"""Simulator driver for the policy search: stage a candidate into a scratch
copy of the contract, run one trace cell, return the scorecard.

**The invocation moved.** `run_cell`, the cell catalogue and the mechanism
check now live in ``agentservesim/runtime/`` -- the arena's runner needs them
and cannot import this package, which would drag the OpenEvolve search into a
benchmark meant to run with no simulator installed. They are re-exported here
unchanged, so `simrun.run_cell(...)` keeps working in all 33 sbatch scripts.

What remains here is what is genuinely about searching: which knob is being
evolved, the cascade's fitness cells, the hold-out, staging a candidate, and
the cached stock baselines each candidate is scored against.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.cells import (CELLS, IMPACT_CELLS, IMPACT_TRACES,   # noqa: F401
                           LB_CELLS, ARENA_CELLS, for_arena)
from runtime.invoke import run_cell, _run_cell_once              # noqa: F401
from runtime.mechanism import (BACKEND_DEATH, mechanism_check,   # noqa: F401
                               _decision_stats, _mechanism_stats,
                               _read_workflows)
from runtime.paths import (AS_ROOT, ENGINE, MAS, PYDEPS, REPO,   # noqa: F401
                           SCRATCH, SIF)

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.join(HERE, "baselines.json")


def _cells_env(name, default, alias=None):
    """Comma-separated cells from the environment, else the default.

    A cell is a registered name from `runtime.cells.CELLS` or an inline
    `trace.jsonl:cluster.json` pair, so searching on your own workload needs no
    edit to this repository.

    `alias` is the older variable name. STAGE1/STAGE2 said where a cell sat in
    the cascade; SEARCH/CHECK say what it is for, which is what someone
    choosing one needs to know.
    """
    v = os.environ.get(name)
    if v is None and alias:
        v = os.environ.get(alias)
    if v is None:
        return list(default)
    return [c for c in v.split(",") if c]


def _search_cells():
    """The cells fitness is measured on.

    The common case is one workload and one cluster config, named the way
    `python -m serving` names them, so nothing new has to be learned:

        EVOLVE_DATASET=my/trace.jsonl
        EVOLVE_CLUSTER_CONFIG=my/cluster.json

    Both are required because scoring a candidate IS running a simulation, and
    a simulation is a workload ON hardware: the same policy can win under one
    KV budget and lose under another, so a fitness number without a cluster
    config would not mean anything.

    `EVOLVE_SEARCH_CELL` takes a comma-separated list instead, for scoring on
    several at once; each entry is a name from `runtime.cells.CELLS` or an
    inline `trace.jsonl:cluster.json`.

    EMPTY by default. The framework ships no cells: every name that could be a
    default points at a trace this project generated, and a default that cannot
    resolve on someone else's machine is worse than none.
    """
    wl = os.environ.get("EVOLVE_DATASET")
    cl = os.environ.get("EVOLVE_CLUSTER_CONFIG")
    if wl and cl:
        return [f"{wl}:{cl}"]
    return _cells_env("EVOLVE_SEARCH_CELL", [], alias="EVOLVE_STAGE1")


STAGE1 = _search_cells()
STAGE2 = _cells_env("EVOLVE_CHECK_CELL", [], alias="EVOLVE_STAGE2")
STAGE3 = _cells_env("EVOLVE_STAGE3", [])
#: A further round, scored only on candidates that passed the ones before.
HOLDOUT = _cells_env("EVOLVE_HOLDOUT", [])


# One candidate file, one flag. The candidate is ONE class deciding as many
# planes as it defines methods for, loaded by `--policy module:Class`, so there
# is no axis to choose and no per-axis file name to drop it in as.
#
# EVOLVE_AXIS used to select among three seeds, three file names and three flag
# sets; the "joint" one wrote two shim modules so that two classes in one file
# could be reached as two axes -- two instances coordinating through a shared
# module, which is the thing `--policy` exists to make unnecessary.
#
# Scheduling candidates still want a retention policy underneath them, because
# a pinned-first rule reads pcb.kv_protected and nothing stamps it otherwise.
# That is a baseline choice, not an axis: EVOLVE_BASE_RETENTION sets it.
CANDIDATE_FILE = "candidate.py"
_MODULE = "candidate"
_BASE_RETENTION = os.environ.get("EVOLVE_BASE_RETENTION", "")

#: The class that decides each plane. One class per plane, several per file --
#: the convention `policies/continuum.py` and every recorded champion follow.
PLANE_CLASS = {"retention": "EvolvedRetention",
               "scheduling": "EvolvedScheduling",
               "routing": "EvolvedRouting"}
_AXIS_FLAG = {"retention": "--retention", "scheduling": "--scheduling",
              "routing": "--routing"}


def planes_of(candidate_path):
    """The planes a candidate decides, read from the classes it defines."""
    import ast as _ast
    tree = _ast.parse(open(candidate_path).read())
    defined = {n.name for n in _ast.walk(tree)
               if isinstance(n, _ast.ClassDef)}
    return [p for p, c in PLANE_CLASS.items() if c in defined]


def flags_for(candidate_path):
    """Simulator flags that run this candidate.

    Each plane's flag names that plane's class, so everything runs on the
    REQUEST planes -- the ones validated against real hardware (+2.4% on the
    board cell, against the program planes' +20.3%). Sending a multi-plane
    candidate to `--planes program` was tried and reverted: shared state is not
    worth searching on an unvalidated engine, and naming the same class twice
    yields one instance anyway.
    """
    planes = planes_of(candidate_path)
    base = (["--retention", _BASE_RETENTION, "--retention-tau", "2"]
            if _BASE_RETENTION and "retention" not in planes else [])
    flags = []
    for p in planes:
        flags += [_AXIS_FLAG[p], f"{_MODULE}:{PLANE_CLASS[p]}"]
    return base + flags


#: Flags for the default seed. Kept because seven of this project's sbatch
#: scripts pass `simrun.AXIS_FLAGS` to `run_cell`, and they were written when
#: the flags were a module constant rather than a function of the candidate.
#: New callers use `flags_for(candidate_path)`.
AXIS_FLAGS = flags_for(os.path.join(HERE, "seed_policy.py"))


def stage_harness(candidate_path, workdir):
    """Copy the candidate into an isolated directory. Returns the --policy-root.

    Just the one file. The candidate used to be dropped INSIDE a full copy of
    `policies/` (and the `harness/` shim beside it) because the engine resolved
    it by a fixed dotted name, `harness.evolved_retention`, which only works if
    the candidate is a member of that package. Named as `module:Class` it is an
    ordinary module on the path, importing `policies.base` from the repo the
    same way anyone's own policy does -- so there is nothing to copy, nothing
    to keep in step with the repo's own packages, and one mechanism instead of
    two.
    """
    shutil.copy(candidate_path, os.path.join(workdir, CANDIDATE_FILE))
    return workdir


def load_baselines():
    if os.path.exists(BASELINES):
        with open(BASELINES) as f:
            return json.load(f)
    return {}


def baseline(cell, compute=True):
    """Stock tuple D scorecard for a cell, cached in baselines.json."""
    b = load_baselines()
    if cell in b:
        return b[cell]
    if not compute:
        raise KeyError(f"no baseline for {cell}; run baseline.py")
    card = run_cell(cell)
    if card["error"]:
        raise RuntimeError(f"baseline for {cell} failed: {card['error']}\n{card['stderr_tail']}")
    b[cell] = card
    with open(BASELINES, "w") as f:
        json.dump(b, f, indent=1, sort_keys=True)
    return card


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="compute/refresh stock baselines")
    ap.add_argument("cells", nargs="*", default=list(CELLS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for c in a.cells:
        if a.force:
            b = load_baselines(); b.pop(c, None)
            with open(BASELINES, "w") as f:
                json.dump(b, f, indent=1, sort_keys=True)
        card = baseline(c)
        print(f"{c:22s} n={card['n']:3d} jct_mean={card['jct_mean']:.3f}s "
              f"p99={card['jct_p99']:.3f}s wall={card['wall_s']}s")
