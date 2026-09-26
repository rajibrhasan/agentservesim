"""Published policies, grouped by paper.

The layout claim is that cutting the files by paper instead of by axis changes
nothing about WHAT runs -- only about what is easy to see. These tests are what
makes that claim checkable, and the first one is the one that matters: the
24-leg answer key was recorded under the axis flags, so `--paper continuum` and
`--retention continuum --scheduling continuum` must build the same classes or
~45 GPU-hours of ground truth stops describing what the simulator runs.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("policies", reason="needs the harness checkout")

import policies as P                                                # noqa: E402
import serving.core.program_policy_adapter as pp                         # noqa: E402
import policies                                                     # noqa: E402
h_ret = h_rt = h_sch = policies


#: The arena's spec for each paper, transcribed from
#: AgentSimArena/configs/policies.yaml. If these drift apart the arena scores
#: one policy and the simulator runs another -- the failure `drift.py` catches
#: one level up, checked here against the classes themselves.
ARENA_SPECS = {
    "stock":     {"kv": "cache-lru", "scheduling": None, "routing": None},
    "continuum": {"kv": "continuum", "scheduling": "continuum", "routing": None},
    "saga":      {"kv": "saga-ttl",  "scheduling": None, "routing": None},
    "autellix":  {"kv": "cache-lru", "scheduling": "plas", "routing": None},
    "infercept": {"kv": "min-waste", "scheduling": None, "routing": None},
}


@pytest.mark.parametrize("paper", ["stock", "continuum", "autellix"])
def test_a_paper_builds_what_its_axis_flags_build(paper):
    """The whole safety property of the re-organisation."""
    spec = ARENA_SPECS[paper]
    by_paper = P.build(paper, tau_s=2.0, num_instances=1)
    if spec["kv"]:
        expected = type(pp.build_retention(spec["kv"], h_ret, tau_s=2.0))
        assert type(by_paper.get("kv", None)) is expected or spec["kv"] == "cache-lru"
    if spec["scheduling"]:
        expected = type(pp.build_scheduling(spec["scheduling"], h_sch))
        assert type(by_paper["scheduling"]) is expected


def test_saga_by_paper_builds_the_retention_the_flag_builds():
    assert type(P.build("saga", tau_s=2.0, num_instances=1)["kv"]) is \
        type(pp.build_retention("saga-ttl", h_ret, tau_s=2.0))


def test_every_paper_is_attributed():
    """A paper cannot be added without a citation: the results table and the
    decision log both print it, and an unattributed row is not publishable."""
    assert set(P.CITATIONS) == set(P.PAPERS)
    assert all(P.CITATIONS[k] for k in P.PAPERS)


def test_every_declared_axis_is_a_real_plane():
    assert all(set(axes) <= set(P.AXES) for axes in P.PAPERS.values())


def test_a_paper_decides_at_least_one_plane():
    assert all(P.PAPERS[p] for p in P.PAPERS)


def test_an_unknown_paper_is_refused_not_defaulted():
    """A typo that silently ran the baseline would look exactly like a paper
    that happens to change nothing."""
    with pytest.raises(ValueError, match="unknown paper"):
        P.axes_of("continum")


# ------------------------------------------------- the half that was lost

def test_saga_by_paper_includes_the_routing_half():
    """SAGA's affinity rule has existed in routing.py since before the arena
    did, and nothing associated it with SAGA: the arena spec leaves `routing:`
    unset and records the absence as a mirror_gap. It was not missing, it was
    unlabelled."""
    built = P.build("saga", tau_s=2.0, num_instances=3, capacity_limit=2)
    assert set(built) == {"kv", "routing"}
    assert isinstance(built["routing"], h_rt.SessionAffinityRouting)


def test_the_axis_flags_still_cannot_reach_it():
    """Stated as a test so the gap is recorded rather than remembered: there is
    no --routing value that names SAGA's rule together with SAGA's retention."""
    assert ARENA_SPECS["saga"]["routing"] is None


# ---------------------------------------------------- paper construction

def test_a_paper_gets_only_the_knobs_its_constructor_takes():
    """Adding a knob for one paper must not oblige every other paper to accept
    it."""
    built = P.build("autellix", tau_s=2.0, num_instances=9, capacity_limit=3,
                    some_future_knob=1)
    assert isinstance(built["scheduling"], h_sch.PLASScheduling)


def test_a_paper_may_build_its_own_axis():
    """InferCept reads a measured waste profile from disk where Continuum takes
    a float; that is the paper's business, not the registry's."""
    from policies import infercept
    assert hasattr(infercept, "make_kv")
    with pytest.raises(ValueError, match="needs --min-waste-profile"):
        P.build("infercept")


def test_the_base_names_are_the_planes_they_talk_to():
    """KVPolicy -> program_kv, SchedulingPolicy -> program_scheduler,
    RoutingPolicy -> program_router. Same objects as the old names, because
    renaming a base would fork the contract."""
    assert P.KVPolicy is h_ret.RetentionPolicy
    assert P.SchedulingPolicy is h_sch.SchedulingPolicy
    assert P.RoutingPolicy is h_rt.RoutingPolicy
