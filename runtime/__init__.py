"""Shared simulation execution, cell definitions, and result collection.

The search evaluator uses these helpers through evolve/simrun.py."""

from .cells import ARENA_CELLS, CELLS, for_arena            # noqa: F401
from .invoke import run_cell                                # noqa: F401
from .mechanism import BACKEND_DEATH, mechanism_check       # noqa: F401
from .paths import AS_ROOT, ENGINE, REPO, SCRATCH, SIF      # noqa: F401

__all__ = ["ARENA_CELLS", "CELLS", "for_arena", "run_cell", "mechanism_check",
           "BACKEND_DEATH", "AS_ROOT", "ENGINE", "REPO", "SCRATCH", "SIF"]
