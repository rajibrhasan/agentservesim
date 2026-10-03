"""The runtime split must not have changed a single simulator invocation."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
AS_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, AS_ROOT)

from runtime import cells, invoke, paths              # noqa: E402

#: The pre-split evolve/simrun.py, vendored beside this test. It is the
#: evidence for the equivalence claim, so it lives in the repository rather
#: than in a scratch directory that outlives nothing.
GOLDEN = os.environ.get("SIMRUN_PRESPLIT",
                        os.path.join(HERE, "simrun_presplit.py"))

CELL = "swe20_kv40_j0.06"
FLAGS = ["--retention", "ttl", "--retention-tau", "2", "--scheduling", "plas"]


class _Captured(Exception):
    """Raised in place of running the simulator, carrying the argv."""

    def __init__(self, argv):
        self.argv = argv
        super().__init__("captured")


def _capture(module, cell=CELL, flags=FLAGS):
    """argv `module._run_cell_once` would have executed."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["argv"] = list(cmd)
        raise _Captured(cmd)

    real = module.subprocess.run
    module.subprocess.run = fake_run
    try:
        module._run_cell_once(cell, None, flags, 60, False)
    except _Captured:
        pass
    finally:
        module.subprocess.run = real
    return seen.get("argv")


def _load_golden():
    """Load the pre-split driver by path, with its deployment constants pinned."""
    import importlib.util
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("simrun_presplit", GOLDEN)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    for const in ("REPO", "AS_ROOT", "MAS", "SIF", "PYDEPS", "SCRATCH"):
        setattr(mod, const, getattr(paths, const))
    return mod


def _normalize(argv):
    """Drop the per-run temp paths: a uuid tag differs every call by design."""
    out = []
    for a in argv:
        if "/cell_" in a or "/in_" in a:
            out.append("<scratch>")
        else:
            out.append(a)
    return out


# ------------------------------------------------------------- equivalence

#: The only differences the 2026-09-12 unification is allowed to have made to
#: the command line. The engine and the policy contract became one directory,
#: so the second bind mounted the same host path twice and the harness root
#: moved into the repository. Everything else must still match, and the
#: container-side path /app/LLMServingSim is deliberately unchanged so the
#: simulator's own relative lookups resolve exactly as before.
EXPECTED_REMOVED = ["--bind", "<repo>:/app/agentservesim"]
EXPECTED_CHANGED = {"--harness-root": ("/app/agentservesim", "/app/LLMServingSim")}


@pytest.mark.skipif(not os.path.exists(GOLDEN),
                    reason=f"pre-split simrun.py not available at {GOLDEN}")
def test_invocation_differs_only_by_the_unification():
    """Every difference from the pre-split driver must be an intended one.

    This started life as a byte-identity check. The unification made two
    deliberate changes, so identity is now the wrong assertion -- but dropping
    the test would give up the guard exactly when the command line is being
    edited. It instead enumerates the intended changes and fails on any other.
    """
    before = _normalize(_capture(_load_golden()))
    after = _normalize(_capture(invoke))

    # apply the intended changes to the OLD argv, then demand equality
    expect = []
    skip = 0
    for i, tok in enumerate(before):
        if skip:
            skip -= 1
            continue
        if tok == "--bind" and i + 1 < len(before) \
                and before[i + 1].endswith(":/app/agentservesim"):
            skip = 1                       # drop the flag and its value
            continue
        expect.append(tok)
    for flag, (old, new) in EXPECTED_CHANGED.items():
        if flag in expect and expect[expect.index(flag) + 1] == old:
            expect[expect.index(flag) + 1] = new

    assert expect == after, (
        "the command line changed in a way the unification does not account "
        "for; every number measured after it would be on a different "
        "simulator than every number before")


def test_the_removed_bind_was_a_duplicate_not_a_loss():
    """Dropping /app/agentservesim is only safe because it was the same
    directory. If REPO and AS_ROOT ever diverge again, the contract would
    silently stop being mounted."""
    assert paths.AS_ROOT == paths.REPO
    assert os.path.isdir(os.path.join(paths.REPO, "harness"))


def test_harness_root_still_contains_the_contract():
    """The in-container harness root must hold the policies package -- else
    every run falls back to whatever `import policies` finds, or nothing.

    Checked by the files the contract actually lives in rather than by the old
    per-axis ones: `retention.py`, `scheduling.py` and `routing.py` were split
    into one module per paper, so asserting their names would test the layout
    of 2026-09 rather than the property that matters.
    """
    argv = _capture(invoke)
    root = argv[argv.index("--harness-root") + 1]
    assert root == "/app/LLMServingSim"
    # the host directory bound at that container path
    bind = next(b for b in argv if b.endswith(":/app/LLMServingSim"))
    host = bind.split(":")[0]
    pkg = os.path.join(host, "policies")
    assert os.path.isdir(pkg), f"{host}/policies missing"
    for f in ("__init__.py", "base.py", "program.py", "continuum.py"):
        assert os.path.exists(os.path.join(pkg, f)), f"policies/{f} missing"
    # and the legacy shim, which the evolve sandbox's allowlist still needs
    assert os.path.exists(os.path.join(host, "harness", "__init__.py"))


def test_paths_resolve_to_the_real_checkout():
    """The constants the equivalence test pins must be right on their own.

    `_load_golden` overwrites the golden's REPO / AS_ROOT with these, so a
    wrong value here would be copied onto both sides and the diff would pass
    while every run bound the wrong directory.
    """
    assert os.path.isdir(paths.REPO), f"REPO {paths.REPO} is not a directory"
    assert os.path.isdir(os.path.join(paths.REPO, "serving")), \
        f"REPO {paths.REPO} does not look like the simulator checkout"
    assert os.path.isdir(os.path.join(paths.AS_ROOT, "harness")), \
        f"AS_ROOT {paths.AS_ROOT} does not hold the policy contract"
    assert os.path.basename(paths.REPO) == "AgentServingSim"


def test_paths_honour_their_environment_overrides(monkeypatch):
    """A checkout somewhere else must be reachable without editing source."""
    import importlib
    monkeypatch.setenv("EVOLVE_SIM_REPO", "/somewhere/else/LLMServingSim")
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.REPO == "/somewhere/else/LLMServingSim"
    finally:
        monkeypatch.delenv("EVOLVE_SIM_REPO")
        importlib.reload(paths)


# ------------------------------------------------- properties, golden or not

def test_engine_settings_are_present():
    argv = _capture(invoke)
    for flag in ("--dtype", "--block-size", "--max-num-seqs",
                 "--max-num-batched-tokens"):
        assert flag in argv, f"{flag} missing: this run is not comparable"


def test_policy_flags_reach_the_simulator():
    argv = _capture(invoke)
    assert "--retention" in argv and argv[argv.index("--retention") + 1] == "ttl"
    assert "--scheduling" in argv and argv[argv.index("--scheduling") + 1] == "plas"


def test_default_harness_root_is_the_checkout_not_a_scratch_copy():
    """A run with no staged candidate must use the committed contract."""
    argv = _capture(invoke)
    assert argv[argv.index("--harness-root") + 1] == "/app/LLMServingSim"


def test_staged_harness_root_overrides_it():
    seen = {}

    def fake_run(cmd, **kw):
        seen["argv"] = list(cmd)
        raise _Captured(cmd)

    real = invoke.subprocess.run
    invoke.subprocess.run = fake_run
    try:
        invoke._run_cell_once(CELL, "/tmp/staged", FLAGS, 60, False)
    except _Captured:
        pass
    finally:
        invoke.subprocess.run = real
    argv = seen["argv"]
    assert argv[argv.index("--harness-root") + 1] == "/evolve_harness"
    assert "/tmp/staged:/evolve_harness" in argv


def test_decision_log_dir_is_always_requested():
    """Both hosts' counters are read from the decision logs; a run without
    them produces a JCT that cannot be checked against its own mechanism."""
    assert "--decision-log-dir" in _capture(invoke)


# ------------------------------------------------------------------ cells

def test_every_arena_cell_maps_to_a_real_cell():
    for arena_name, key in cells.ARENA_CELLS.items():
        assert key in cells.CELLS, f"{arena_name} -> unknown cell {key}"


def test_for_arena_returns_none_rather_than_raising():
    """The arena renders a missing cell as a blank with a reason."""
    assert cells.for_arena("no_such_cell") is None


def test_cell_entries_are_trace_and_cluster_pairs():
    for name, entry in cells.CELLS.items():
        assert len(entry) == 2, name
        assert entry[0].endswith(".jsonl"), f"{name}: {entry[0]} is not a trace"
        assert entry[1].endswith(".json"), f"{name}: {entry[1]} is not a config"


# ------------------------------------------------------------------ shim

def test_simrun_still_exports_what_the_sbatch_scripts_call():
    sys.path.insert(0, os.path.join(AS_ROOT, "evolve"))
    import simrun
    for name in ("run_cell", "stage_harness", "AXIS_FLAGS", "STAGE1", "STAGE2",
                 "HOLDOUT", "CELLS", "mechanism_check", "baseline"):
        assert hasattr(simrun, name), f"simrun.{name} disappeared in the split"


def test_simrun_run_cell_is_the_runtime_one():
    """Not a copy: the same function object, so they cannot drift apart."""
    sys.path.insert(0, os.path.join(AS_ROOT, "evolve"))
    import simrun
    assert simrun.run_cell is invoke.run_cell


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
