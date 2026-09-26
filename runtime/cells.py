"""The cell catalogue: what trace, on what cluster configuration.

Split out of evolve/simrun.py on 2026-09-12. A cell is the simulator-side
equivalent of an arena cell -- a workload paired with a memory budget -- and it
is data, so it belongs somewhere both the policy search and the arena runner
can read without importing the other.

`ARENA_CELLS` maps arena cell names onto entries here. It lives on this side
deliberately: which cluster JSON reproduces a given KV pool is a fact about
THIS simulator, and a benchmark that carried that table would be encoding one
entrant's file layout into its own source.
"""

import os

from .paths import MAS, REPO  # noqa: F401  (REPO kept for callers that expect it)

CELLS = {
    "bfcl20_kv40_j0.06":  ("workloads/bfcl_v4/evolve/bfcl20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_B200_kv40.json"),
    "bfcl20_kv40_j0.02":  ("workloads/bfcl_v4/evolve/bfcl20_jps0.02_n20.jsonl",
                           "experiments/validation/cluster_B200_kv40.json"),
    "bfcl20_kv40_j0.1":   ("workloads/bfcl_v4/evolve/bfcl20_jps0.1_n20.jsonl",
                           "experiments/validation/cluster_B200_kv40.json"),
    "bfcl20_kv180_j0.06": ("workloads/bfcl_v4/evolve/bfcl20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_B200.json"),
    "swe20_kv24_j0.06":   ("workloads/swebench/evolve/swebench20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_B200_kv24.json"),
    "swe20_kv20_j0.06":   ("workloads/swebench/evolve/swebench20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_B200_kv20.json"),
    "swe20_kv40_j0.06":   ("workloads/swebench/evolve/swebench20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_B200_kv40.json"),
    # Exploratory caps for choosing the pressure axis.
    "bfcl20_kv24_j0.06":  ("workloads/bfcl_v4/evolve/bfcl20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_B200_kv24.json"),
    "bfcl20_kv22_j0.06":  ("workloads/bfcl_v4/evolve/bfcl20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_B200_kv22.json"),
    "bfcl20_kv20_j0.06":  ("workloads/bfcl_v4/evolve/bfcl20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_B200_kv20.json"),
    # RTX PRO 6000 x2, Llama-3.1-70B tp2: the validated saturated regime
    # (pool holds ~7 full SWE-bench contexts). The 10-program trace is the
    # search objective (~10-15 min/eval); the 50-program trace is the
    # hold-out for final candidates only (~1 h/eval) and also has a real
    # replay (experiments/replay/results/rtx6000_70b_swebench__*).
    "rtx70b_swe10_j0.02": ("experiments/validation/regress_baselines/swebench_jps0.02_first10.jsonl",
                           "experiments/validation/cluster_rtx6000_70b.json"),
    "rtx70b_swe50_j0.02": ("workloads/swebench/swebench_jps0.02_n50.jsonl",
                           "experiments/validation/cluster_rtx6000_70b.json"),
    # Identical to rtx70b_swe50_j0.02 except the host tier is 16 GB instead of
    # the board's 512 GB. The board default never refuses a swap, so
    # infercept-swap scored 926.7 s there against 5018.1 s on an independent
    # run using a 16 GB tier (results/final/validation/infercept/history/sim/rtx_j0.02). This
    # cell isolates the host-memory bound; the GPU side is unchanged.
    "rtx70b_swe50_j0.02_cpu16": ("workloads/standard/swebench50/swebench_jps0.02_n50.jsonl",
                                 "experiments/validation/cluster_rtx6000_70b_cpu16.json"),
    # 20 programs at jps 0.06: sustained saturation (10 programs at 0.02
    # only overlap mildly: stock JCT 171 s vs 3674 s on the 50-program cell).
    "rtx70b_swe50_j0.1":  ("workloads/swebench/swebench_jps0.1_n50.jsonl",
                           "experiments/validation/cluster_rtx6000_70b.json"),
    "rtx70b_swe20_j0.06": ("workloads/swebench/evolve/swebench20_jps0.06_n20.jsonl",
                           "experiments/validation/cluster_rtx6000_70b.json"),
    # Prefixes of the validated 50-program traces (arrival times embedded, so
    # a prefix is a valid trace): candidates for a cell where policies
    # actually separate (swe20_j0.06 gave identical JCT for every policy).
    "rtx70b_f20_j0.02": ("workloads/swebench/evolve/swebench_first20_jps0.02.jsonl",
                         "experiments/validation/cluster_rtx6000_70b.json"),
    # Operating-point sweep: where do agent policies beat stock? Rate axis
    # (30-program SWE-bench prefixes) and workload-mixture axis (equal counts
    # of long SWE-bench and short BFCL programs in one stream, i.e. Autellix's
    # "Mixed" and SAGA's heavy/light tenant mix, which is where head-of-line
    # blocking between program classes exists at all).
    "rtx70b_f30_j0.04": ("workloads/swebench/evolve/swebench_first30_jps0.04.jsonl",
                         "experiments/validation/cluster_rtx6000_70b.json"),
    "rtx70b_f30_j0.06": ("workloads/swebench/evolve/swebench_first30_jps0.06.jsonl",
                         "experiments/validation/cluster_rtx6000_70b.json"),
    "rtx70b_mix_j0.02": ("workloads/mixed/mix20x2_jps0.02.jsonl",
                         "experiments/validation/cluster_rtx6000_70b.json"),
    "rtx70b_mix_j0.04": ("workloads/mixed/mix20x2_jps0.04.jsonl",
                         "experiments/validation/cluster_rtx6000_70b.json"),
    "rtx70b_f30_j0.02": ("workloads/swebench/evolve/swebench_first30_jps0.02.jsonl",
                         "experiments/validation/cluster_rtx6000_70b.json"),
    "rtx70b_f20_j0.1":  ("workloads/swebench/evolve/swebench_first20_jps0.1.jsonl",
                         "experiments/validation/cluster_rtx6000_70b.json"),
    "rtx70b_f30_j0.1":  ("workloads/swebench/evolve/swebench_first30_jps0.1.jsonl",
                         "experiments/validation/cluster_rtx6000_70b.json"),
}

# Impact sweep (docs/iclr-impact-sweep.md): star design around
# rtx70b_f30_j0.1 manufacturing the preconditions for policy impact.
# Derived traces live on /orange (home quota); recipe is in the filename.
IMPACT_TRACES = os.path.join(MAS, "workloads", "impact")
_RTX = "experiments/validation/cluster_rtx6000_70b.json"
_B200TP1 = "experiments/validation/cluster_B200_llama70b_tp1.json"
_B200PHI = "experiments/validation/cluster_B200_phi35moe_tp1.json"
_RTX70BX2 = "experiments/validation/cluster_rtx6000_70b_x2.json"
_RTXPHI = "experiments/validation/cluster_rtx6000_phi35moe_tp1.json"
_B200X2TP1 = "experiments/validation/cluster_B200_llama70b_tp1_x2.json"
_B200 = "experiments/validation/cluster_B200.json"
CELLS.update({
    # AgentSimArena second platform: B200 x1, Llama-3.1-70B, TP1 (config sized
    # from the real stock probe's kv_cache_tokens; see agentsimarena plan).
    "b200tp1_swe50_j0.02": ("workloads/swebench/swebench_jps0.02_n50.jsonl", _B200TP1),
    # Phi-3.5-MoE tp1 on B200, same SWE-bench trace as the Llama cells. The
    # first cell to exercise the MoE profile tables end to end: nothing in
    # configs/cluster named Phi, so the tables had never been read by a run.
    # Its pool is model arithmetic, not a measured engine probe -- no Phi
    # replay exists, so this cell has no hardware counterpart.
    "b200phi_swe50_j0.02": ("workloads/standard/swebench50/swebench_jps0.02_n50.jsonl", _B200PHI),
    # Four cells added 2026-09-19 to exercise paths the board never touched:
    # multi-instance routing, the MoE tables under contention, and BFCL.
    # rtx70bx2_bfcl20: two 70B tp2 instances, so --routing has two places to
    # send a program; the single-instance board cells made every routing
    # policy a no-op (autellix_route returned the identical JCT to plain MLFQ).
    "rtx70bx2_bfcl150_j0.06": ("workloads/standard/bfcl150/bfcl_jps0.06_n150.jsonl", _RTX70BX2),
    # Phi on RTX: 68,864 KV tokens against the B200 Phi cell's 688,179, so the
    # MoE tables are read under real memory pressure rather than a pool that
    # never fills.
    "rtxphi_swelb100_j0.1": ("workloads/standard/swelb100/swelb100_jps0.1_n100.jsonl", _RTXPHI),
    "b200tp1_8b_swelb100_j0.1": ("workloads/standard/swelb100/swelb100_jps0.1_n100.jsonl", _B200),
    "b200x2_70b_bfcl150_j0.06": ("workloads/standard/bfcl150/bfcl_jps0.06_n150.jsonl", _B200X2TP1),
    "b200tp1_lb20_j0.1":   (f"{IMPACT_TRACES}/swelb20_jps0.1.jsonl", _B200TP1),
    # held-out arrival rate: same workload + hardware as b200tp1_swe50_j0.02, double the rate
    "b200tp1_swe50_j0.04": ("workloads/swebench/swebench_jps0.04_n50.jsonl", _B200TP1),
})
CELLS.update({
    # G: gap scale (tool_duration x gamma), embedded j0.1 arrivals kept
    "imp_g4_j0.1":   (f"{IMPACT_TRACES}/swe30_g4_jps0.1.jsonl", _RTX),
    "imp_g16_j0.1":  (f"{IMPACT_TRACES}/swe30_g16_jps0.1.jsonl", _RTX),
    "imp_g64_j0.1":  (f"{IMPACT_TRACES}/swe30_g64_jps0.1.jsonl", _RTX),
    # T: tail-only amplification (gaps > p75 get x32)
    "imp_t32_j0.1":  (f"{IMPACT_TRACES}/swe30_t32_jps0.1.jsonl", _RTX),
    # R: rate axis, fresh seeded arrivals on the same 30 programs
    "imp_j0.15":     (f"{IMPACT_TRACES}/swe30_jps0.15.jsonl", _RTX),
    "imp_j0.2":      (f"{IMPACT_TRACES}/swe30_jps0.2.jsonl", _RTX),
    # P: pool scarcity via shrunk-KV cluster configs, base trace
    "imp_kvhalf_j0.1":    ("workloads/swebench/evolve/swebench_first30_jps0.1.jsonl",
                           "experiments/validation/cluster_rtx6000_70b_kvhalf.json"),
    "imp_kvquarter_j0.1": ("workloads/swebench/evolve/swebench_first30_jps0.1.jsonl",
                           "experiments/validation/cluster_rtx6000_70b_kvquarter.json"),
    # G x P interaction preview
    "imp_g16_kvquarter_j0.1": (f"{IMPACT_TRACES}/swe30_g16_jps0.1.jsonl",
                               "experiments/validation/cluster_rtx6000_70b_kvquarter.json"),
    # M: SWE:BFCL mixtures (Autellix head-of-line story)
    "imp_mix1to1_j0.1": (f"{IMPACT_TRACES}/mix1to1_jps0.1.jsonl", _RTX),
    "imp_mix1to4_j0.1": (f"{IMPACT_TRACES}/mix1to4_jps0.1.jsonl", _RTX),
    "imp_mix1to4_j0.2": (f"{IMPACT_TRACES}/mix1to4_jps0.2.jsonl", _RTX),
})

IMPACT_CELLS = [c for c in CELLS if c.startswith("imp_")]

# LB: SWE-bench leaderboard traces (mini-swe-agent v2 + Sonnet 4.5 high,
# public S3; convert_leaderboard_trajs.py). Real measured tool gaps
# (mean 0.87 s std 4.1, matching Continuum) + 1.55x heavier contexts and
# 10x heavier decode than the qwen-3.6 collection — no synthetic knobs.
CELLS.update({
    "lb40_j0.05":        (f"{IMPACT_TRACES}/swelb40_jps0.05.jsonl", _RTX),
    "lb40_j0.1":         (f"{IMPACT_TRACES}/swelb40_jps0.1.jsonl", _RTX),
    "lb60_j0.1":         (f"{IMPACT_TRACES}/swelb60_jps0.1.jsonl", _RTX),
    "lb40_kvhalf_j0.05": (f"{IMPACT_TRACES}/swelb40_jps0.05.jsonl",
                          "experiments/validation/cluster_rtx6000_70b_kvhalf.json"),
    "lb40_kvhalf_j0.1":  (f"{IMPACT_TRACES}/swelb40_jps0.1.jsonl",
                          "experiments/validation/cluster_rtx6000_70b_kvhalf.json"),
    "lb60_kvhalf_j0.1":  (f"{IMPACT_TRACES}/swelb60_jps0.1.jsonl",
                          "experiments/validation/cluster_rtx6000_70b_kvhalf.json"),
    # Screening cell for policy search: first-20-arrivals prefix of lb40
    # (913 turns, ~35-40 min/eval). Fitness cell iff it preserves the
    # lb40_j0.1 policy ranking.
    "lb20_j0.1":  (f"{IMPACT_TRACES}/swelb20_jps0.1.jsonl", _RTX),
    "lb20_j0.05": (f"{IMPACT_TRACES}/swelb20_jps0.05.jsonl", _RTX),
})

LB_CELLS = [c for c in CELLS if c.startswith("lb")]

# The standard workload matrix: three agentic workloads x five arrival rates,
# every experiment from 2026-09-15 on. One program count per workload so a
# number is comparable across rates without renormalising, and one trace
# family per workload so "the BFCL cell" names exactly one file. Contexts are
# capped at 100k tokens, under the 114,768-token pool the RTX config reports,
# so no turn is rejected by max_model_len. Sources and the five derived rate
# files sit together under each directory; `src.jsonl` is the unsplit trace
# each rate was sampled from.
STD = os.path.join(MAS, "workloads", "standard")
_RATES = ("0.02", "0.04", "0.06", "0.08", "0.1")
CELLS.update({
    f"std_bfcl150_j{j}": (f"{STD}/bfcl150/bfcl_jps{j}_n150.jsonl", _RTX)
    for j in _RATES})
CELLS.update({
    f"std_swe50_j{j}": (f"{STD}/swebench50/swebench_jps{j}_n50.jsonl", _RTX)
    for j in _RATES})
CELLS.update({
    f"std_swelb100_j{j}": (f"{STD}/swelb100/swelb100_jps{j}_n100.jsonl", _RTX)
    for j in _RATES})

#: The standard matrix, in the order results should be reported.
STD_CELLS = [f"std_{w}_j{j}" for w in ("bfcl150", "swe50", "swelb100")
             for j in _RATES]



#: arena cell name -> cell key above. The arena names a deployment by its
#: hardware and trace; this says which of our configurations realizes it.
ARENA_CELLS = {
    "rtx_swe50_j0.02":  "rtx70b_swe50_j0.02",
    "rtx_lb20_j0.1":    "lb20_j0.1",
    "b200_swe50_j0.02": "b200tp1_swe50_j0.02",
    "b200_lb20_j0.1":   "b200tp1_lb20_j0.1",
    "b200_swe50_j0.04": "b200tp1_swe50_j0.04",
}


def for_arena(name):
    """Cell key for an arena cell name, or None if this simulator has no
    configuration that reproduces it. None is an answer, not an error: the
    arena reports it as a blank with a reason."""
    return ARENA_CELLS.get(name)


def resolve(cell):
    """(dataset, cluster_config) for a cell NAME or an inline `trace:config`.

    A registered name is a convenience for workloads used repeatedly. It must
    not be the only way in: the names in `CELLS` point at traces this project
    generated, and requiring someone to edit this file before they can search
    on their own workload makes their workload a second-class input.

        resolve("rtx70b_swe50_j0.02")                      # registered
        resolve("my/trace.jsonl:configs/cluster/mine.json")  # anything
    """
    if cell in CELLS:
        return CELLS[cell]
    if ":" in cell:
        dataset, _, cluster = cell.rpartition(":")
        if dataset and cluster:
            return dataset, cluster
    raise KeyError(
        f"unknown cell {cell!r}. Use a name from runtime.cells.CELLS, or give "
        f"the pair directly as 'path/to/trace.jsonl:path/to/cluster.json'.")
