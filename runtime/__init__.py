"""How to run AgentServingSim. No search, no benchmark, no policy logic.

Split out of ``evolve/simrun.py`` on 2026-09-12, when the arena became a
standalone repository. The split is forced by a dependency that could not
stand: the benchmark's runner needed `run_cell`, `run_cell` lived inside the
OpenEvolve search package, and so evaluating this simulator meant importing a
policy-search framework. The arena is supposed to run with no simulator
installed at all.

    paths      where the checkout, container and scratch live
    cells      the cell catalogue, plus the arena-name mapping
    mechanism  did the policy a run claims actually fire
    invoke     run_cell: build the command, run it, return a scorecard

Two consumers, one invocation:

    from runtime import run_cell            # the arena's runner
    from evolve import simrun               # the policy search (a shim on this)

`evolve/simrun.py` re-exports everything here, so the 33 sbatch scripts that
call `simrun.run_cell` keep working unchanged. What stays in `simrun` is what
is genuinely about searching: the axis, the cascade stages, the hold-out, and
staging a candidate into a scratch copy of the contract.
"""

from .cells import ARENA_CELLS, CELLS, for_arena            # noqa: F401
from .invoke import run_cell                                # noqa: F401
from .mechanism import BACKEND_DEATH, mechanism_check       # noqa: F401
from .paths import AS_ROOT, ENGINE, REPO, SCRATCH, SIF      # noqa: F401

__all__ = ["ARENA_CELLS", "CELLS", "for_arena", "run_cell", "mechanism_check",
           "BACKEND_DEATH", "AS_ROOT", "ENGINE", "REPO", "SCRATCH", "SIF"]
