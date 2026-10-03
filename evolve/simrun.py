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
    """Read comma-separated cell names or trace:cluster pairs, with a legacy alias."""
    v = os.environ.get(name)
    if v is None and alias:
        v = os.environ.get(alias)
    if v is None:
        return list(default)
    return [c for c in v.split(",") if c]


def _search_cells():
    """Read EVOLVE_DATASET and EVOLVE_CLUSTER_CONFIG, or EVOLVE_SEARCH_CELL.
    
    No search cell is selected by default. Both workload and deployment are
    required to score a policy."""
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
    """Select request-plane CLI flags for the classes defined by this candidate."""
    planes = planes_of(candidate_path)
    base = (["--retention", _BASE_RETENTION, "--retention-tau", "2"]
            if _BASE_RETENTION and "retention" not in planes else [])
    flags = []
    for p in planes:
        flags += [_AXIS_FLAG[p], f"{_MODULE}:{PLANE_CLASS[p]}"]
    return base + flags


AXIS_FLAGS = flags_for(os.path.join(HERE, "seed_policy.py"))


def stage_harness(candidate_path, workdir):
    """Stage candidate.py alone; shared policy modules are imported from the repository."""
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
        raise KeyError(f"no baseline for {cell}; run python evolve/simrun.py with this cell")
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
