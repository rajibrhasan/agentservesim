"""Unit tests for the search plumbing that runs without a simulator:
the observation-boundary check and the scorecard-to-fitness math."""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
sys.path.insert(0, EVOLVE)
import sandbox  # noqa: E402
import evaluate  # noqa: E402
import simrun  # noqa: E402

SEED = os.path.join(EVOLVE, "seed_policy.py")


def test_seed_passes_sandbox():
    ok, reasons = sandbox.check_file(SEED)
    assert ok, reasons


@pytest.mark.parametrize("snippet,needle", [
    ("import os", "import of 'os'"),
    ("import time", "import of 'time'"),
    ("import random", "import of 'random'"),
    ("x = open('f')", "call to 'open'"),
    ("x = pcb.program_id", "'.program_id'"),
    ("x = pcb.kv_request_id", "'.kv_request_id'"),
    ("x = pcb.__dict__", "'.__dict__'"),
    ("x = getattr(pcb, name)", "call to 'getattr'"),
])
def test_sandbox_rejects(snippet, needle):
    src = open(SEED).read().replace(
        "        return (\"protect\", now + self.TAU_S)",
        "        " + snippet + "\n        return (\"protect\", now + self.TAU_S)")
    ok, reasons = sandbox.check_source(src)
    assert not ok
    assert any(needle in r for r in reasons), reasons


def test_sandbox_requires_a_policy_class():
    ok, reasons = sandbox.check_source("from policies.base import KVPolicy\n")
    assert not ok and any("no policy class" in r for r in reasons)


def test_a_candidate_must_decide_at_least_one_plane():
    """Deciding nothing is not a policy. Deciding ONE is: whatever a candidate
    leaves out falls back to the engine's own rule, which is the whole reason
    one class can stand in for what used to be three axis-specific seeds."""
    ok, reasons = sandbox.check_source(
        "class Other:\n    def helper(self):\n        return 1\n")
    assert not ok and any("no policy class" in r for r in reasons), reasons
    for cls, method, sig in (
            ("EvolvedRetention", "on_turn_complete", "self, pcb, request_id, now"),
            ("EvolvedScheduling", "priority", "self, pcb, now"),
            ("EvolvedRouting", "route", "self, pcb, now")):
        ok, reasons = sandbox.check_source(
            f"class {cls}:\n    def {method}({sig}):\n        return None\n")
        assert ok, (cls, reasons)


def _card(mean, p99, decisions=None, error=None):
    return {"jct_mean": mean, "jct_p99": p99, "wall_s": 1.0, "n": 20,
            "decisions": decisions or {"protect": 10, "evict": 0, "none": 10, "release": 10},
            "error": error, "stderr_tail": ""}


def test_score_is_stock_over_candidate(monkeypatch):
    base = {"a": _card(10.0, 20.0), "b": _card(10.0, 20.0)}
    monkeypatch.setattr(simrun, "baseline", lambda c, compute=True: base[c])
    m, art = evaluate._score({"a": _card(8.0, 16.0), "b": _card(12.5, 25.0)})
    assert m["combined_score"] == pytest.approx((10 / 8 + 10 / 12.5) / 2)
    assert m["jct_ratio_min"] == pytest.approx(0.8)
    assert m["p99_ratio"] == pytest.approx((20 / 16 + 20 / 25) / 2)
    assert m["protect_frac"] == pytest.approx(20 / 60)
    assert m["cells"] == 2.0
    assert "a" in art["cells"] and json.loads(art["decisions"])["protect"] == 20


def test_failed_cell_is_zero_with_feature_keys(monkeypatch):
    monkeypatch.setattr(simrun, "baseline", lambda c, compute=True: _card(10.0, 20.0))
    m, art = evaluate._score({"a": _card(None, None, error="simulator exit 1")})
    assert m["combined_score"] == 0.0
    for k in ("protect_frac", "evict_frac"):
        assert k in m  # MAP-Elites raises if a feature dimension is missing
    assert art["failure_cell"] == "a"


def test_stage_lists_are_disjoint_and_known():
    stages = simrun.STAGE1 + simrun.STAGE2 + simrun.STAGE3
    assert len(stages) == len(set(stages))
    assert all(c in simrun.CELLS for c in stages)
    for ds, cfg in simrun.CELLS.values():
        assert os.path.exists(os.path.join(simrun.REPO, ds)), ds
        assert os.path.exists(os.path.join(simrun.REPO, cfg)), cfg


# ------------------------------------------------- one class, several planes


def test_one_class_can_decide_several_planes():
    """The reason the seeds collapsed into one. Two classes in one file had to
    coordinate through module state; one object holds its own."""
    src = ("class EvolvedRetention:\n"
           "    def on_turn_complete(self, pcb, request_id, now):\n"
           "        return None\n"
           "class EvolvedScheduling:\n"
           "    def priority(self, pcb, now):\n"
           "        return 1\n"
           "class EvolvedRouting:\n"
           "    def route(self, pcb, now):\n"
           "        return 0\n")
    ok, reasons = sandbox.check_source(src)
    assert ok, reasons


def test_literal_getattr_allowed_dynamic_rejected():
    base = open(SEED).read()
    ok, _ = sandbox.check_source(base.replace(
        "        return (\"protect\", now + self.TAU_S)",
        "        u = getattr(self.signals, 'kv_utilization', None)\n"
        "        return (\"protect\", now + self.TAU_S)"))
    assert ok


def test_scheduling_base_hooks_default_to_engine():
    sys.path.insert(0, os.path.dirname(EVOLVE))
    from harness.scheduling import SchedulingPolicy, QueueView
    p = SchedulingPolicy()
    assert p.victim([], 0.0) is None
    view = QueueView(1, 1, 0, 0.5, 100, 100, 10, 0)
    assert p.admit(None, 0.0, view) is True


def test_normalized_hash_ignores_cosmetic_edits(tmp_path):
    a = open(SEED).read()
    b = a.replace("Seed: fixed-horizon protection", "SEED (reworded)").replace(
        "        return \"release\"", "        # a comment\n        return \"release\"")
    c = a.replace("TAU_S = 2.0", "TAU_S = 3.0")
    pa, pb, pc = tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"
    pa.write_text(a); pb.write_text(b); pc.write_text(c)
    assert evaluate._hash(str(pa)) == evaluate._hash(str(pb))
    assert evaluate._hash(str(pa)) != evaluate._hash(str(pc))


def test_stage_harness_drops_one_candidate(tmp_path):
    """One file, nothing else.

    The candidate used to be dropped inside a full copy of `policies/`, with
    the `harness/` shim beside it, because the engine resolved it by the fixed
    dotted name `harness.evolved_retention` -- which requires membership of
    that package. Named as `module:Class` it is an ordinary module, so the
    staging copies the candidate and stops.
    """
    root = simrun.stage_harness(SEED, str(tmp_path))
    assert os.listdir(root) == [simrun.CANDIDATE_FILE]


def test_a_staged_candidate_imports_the_repo_policies(tmp_path):
    """The candidate is standalone but still gets its base class from the
    repo, so there is no second copy of `policies/` to drift."""
    import subprocess
    import sys as _sys

    root = simrun.stage_harness(SEED, str(tmp_path))
    repo = os.path.dirname(EVOLVE)
    probe = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "import candidate, policies.base\n"
        "assert candidate.KVPolicy is policies.base.KVPolicy\n"
        "print('ok')\n" % (root, repo))
    out = subprocess.run([_sys.executable, "-c", probe], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr[-600:]


def test_flags_put_every_plane_on_the_same_spec(tmp_path):
    """A multi-plane candidate names one spec on each flag it decides, and the
    adapter builds ONE instance for all of them -- so shared state does not
    cost the validated engine. Routing such a policy to `--planes program` was
    tried and reverted: the request planes are the ones checked against real
    hardware."""
    p = tmp_path / "multi.py"
    p.write_text("class EvolvedRetention:\n"
                 "    def on_turn_complete(self, pcb, rid, now): return None\n"
                 "class EvolvedScheduling:\n"
                 "    def priority(self, pcb, now): return 1\n")
    flags = simrun.flags_for(str(p))
    assert "--planes" not in flags, flags
    assert "candidate:EvolvedRetention" in flags, flags
    assert "candidate:EvolvedScheduling" in flags, flags
    assert "--retention" in flags and "--scheduling" in flags


def test_the_system_message_matches_every_config():
    """One message, not seven copies.

    It was inlined into each config with no shared source, and
    `system_message.txt` was read by nothing -- so `config_openrouter.yaml`
    had already drifted from the rest, and the mutator was told something
    different depending on which config launched the run.
    """
    import glob
    import yaml
    txt = " ".join(open(os.path.join(EVOLVE, "system_message.txt")).read().split())
    configs = glob.glob(os.path.join(EVOLVE, "config_*.yaml"))
    assert configs, "no configs found"
    for f in configs:
        msg = (yaml.safe_load(open(f)).get("prompt") or {}).get("system_message", "")
        assert " ".join(msg.split()) == txt, f"{os.path.basename(f)} has drifted"


def test_the_system_message_names_every_plane_class():
    """The mutator is told which classes it may write. When the seed grew a
    scheduling class the message still said `EvolvedRetention` only, so the
    search would have been instructed to ignore half the file."""
    import simrun
    msg = open(os.path.join(EVOLVE, "system_message.txt")).read()
    for cls in simrun.PLANE_CLASS.values():
        assert cls in msg, f"{cls} is not described to the mutator"
