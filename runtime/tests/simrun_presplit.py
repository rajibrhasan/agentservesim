"""Simulator driver for the policy search: stage a candidate into a scratch copy of
the harness, run one trace cell, return the scorecard."""

import ast
import csv
import getpass
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
AS_ROOT = os.path.dirname(HERE)                       # agentservesim/
REPO = os.environ.get(
    "EVOLVE_SIM_REPO",
    os.path.join(os.path.dirname(AS_ROOT), "LLMServingSim"))
MAS = os.environ.get("EVOLVE_MAS", "/opt/artifact-data/masservingsim")
SIF = os.path.join(MAS, "sifs", "sim.sif")
PYDEPS = os.path.join(MAS, "sim_pydeps")
SCRATCH = os.environ.get(
    "EVOLVE_SCRATCH", "/dev/shm/evolve_" + getpass.getuser())
BASELINES = os.path.join(HERE, "baselines.json")

ENGINE = ["--dtype", "bfloat16", "--block-size", "16",
          "--max-num-seqs", "128", "--max-num-batched-tokens", "16384",
          "--num-reqs", "0", "--log-level", "WARNING"]

# name -> (dataset, cluster config). All 20-program subtraces; the
# cluster config carries the KV budget (kv40 = 40 GB NPU memory, of
# which ~24 GB is KV after the 8B weights; the plain config is 180 GB).
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
IMPACT_TRACES = "/opt/artifact-data/masservingsim/workloads/impact"
_RTX = "experiments/validation/cluster_rtx6000_70b.json"
_B200TP1 = "experiments/validation/cluster_B200_llama70b_tp1.json"
CELLS.update({
    # AgentSimArena second platform: B200 x1, Llama-3.1-70B, TP1 (config sized
    # from the real stock probe's kv_cache_tokens; see agentsimarena plan).
    "b200tp1_swe50_j0.02": ("workloads/swebench/swebench_jps0.02_n50.jsonl", _B200TP1),
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
    # Full 500-program headline runs (~9x lb60 turns; ~1-2 days wall).
    # kvhalf infeasible at n=500: max ctx 64,628 > halved pool 57,384.
    "lb500_j0.05": ("/opt/artifact-data/masservingsim/workloads/leaderboard/swelb_jps0.05_n500.jsonl", _RTX),
    "lb500_j0.1":  ("/opt/artifact-data/masservingsim/workloads/leaderboard/swelb_jps0.1_n500.jsonl", _RTX),
})

LB_CELLS = [c for c in CELLS if c.startswith("lb")]

def _cells_env(name, default):
    """Comma-separated cell list from the environment (EVOLVE_STAGE1, ...),
    else the default. Lets a search pick its fitness cells without editing
    this file (a running search has already imported it)."""
    v = os.environ.get(name)
    if v is None:
        return list(default)
    return [c for c in v.split(",") if c]


STAGE1 = _cells_env("EVOLVE_STAGE1", ["lb20_j0.1"])          # screening cell, saturated
STAGE2 = _cells_env("EVOLVE_STAGE2", ["lb20_j0.05"])         # second rate: must hold up at both
STAGE3 = _cells_env("EVOLVE_STAGE3", [])                     # (cascade never reaches the hold-out)
HOLDOUT = _cells_env("EVOLVE_HOLDOUT", ["lb40_j0.1", "lb60_kvhalf_j0.1"])  # final candidates only
# Collected-SWE-bench variant (Continuum-seeded search, 2026-09-03):
#   EVOLVE_STAGE1=rtx70b_f30_j0.1 EVOLVE_STAGE2=rtx70b_swe50_j0.1
#   EVOLVE_HOLDOUT=rtx70b_swe50_j0.02,imp_kvhalf_j0.1


# Which knob the search evolves: retention (default) or scheduling. The
# axis picks the harness file the candidate is dropped in as and the
# simulator flag that selects it.
AXIS = os.environ.get("EVOLVE_AXIS", "retention")
AXIS_FILE = {"retention": "evolved_retention.py",
             "scheduling": "evolved_scheduling.py",
             "joint": "evolved_joint.py"}[AXIS]
# The scheduling axis runs with gap-scoped ttl-2 retention: the seed's
# pinned-first rule reads pcb.kv_protected, which only exists when a
# retention policy stamps protections (best measured tuple = ttl2-cont).
# The joint axis evolves both classes from one file; both knobs select it.
AXIS_FLAGS = {"retention": ["--retention", "evolved"],
              "scheduling": ["--retention", "ttl", "--retention-tau", "2",
                             "--scheduling", "evolved"],
              "joint": ["--retention", "evolved", "--scheduling", "evolved"]}[AXIS]


def stage_harness(candidate_path, workdir):
    """Copy harness/ into workdir and drop the candidate in under the
    axis's file name. Returns the --harness-root to pass."""
    dst = os.path.join(workdir, "harness")
    shutil.copytree(
        os.path.join(AS_ROOT, "harness"), dst,
        ignore=shutil.ignore_patterns("__pycache__", "tests", "profiles"))
    shutil.copy(candidate_path, os.path.join(dst, AXIS_FILE))
    if AXIS == "joint":
        # retention.py / scheduling.py each import their evolved_* module
        # at the bottom; shim both onto the single joint candidate.
        for shim, cls in (("evolved_retention.py", "EvolvedRetention"),
                          ("evolved_scheduling.py", "EvolvedScheduling")):
            with open(os.path.join(dst, shim), "w") as f:
                f.write(f"from .evolved_joint import {cls}  # noqa: F401\n")
    return workdir


def _read_workflows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [int(r["jct_ns"]) / 1e9 for r in rows]


def _decision_stats(log_dir):
    counts = {"protect": 0, "evict": 0, "release": 0, "none": 0}
    p = os.path.join(log_dir, "retention.jsonl")
    if not os.path.exists(p):
        return counts
    with open(p) as f:
        for line in f:
            try:
                a = json.loads(line).get("action")
            except ValueError:
                continue
            if a in counts:
                counts[a] += 1
    return counts


BACKEND_DEATH = "ASTRA-Sim backend terminated"

_STATS_RE = re.compile(r"KV protection stats:\s*(\{.*?\})")


def _mechanism_stats(stdout, log_dir):
    """Adapter event counters printed at the end of a run, plus the number
    of priority stamps in the scheduling decision log."""
    out = {}
    m = _STATS_RE.search(stdout or "")
    if m:
        try:
            out.update(ast.literal_eval(m.group(1)))
        except (ValueError, SyntaxError):
            pass
    p = os.path.join(log_dir, "scheduling.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            out["stamps"] = sum(1 for _ in f)
    return out


def mechanism_check(flags, mech):
    """Reasons a run's counters contradict the policy it claims to have run.
    A policy whose defining event never fired produced a number for some
    OTHER policy (the engine default), and that number must not be
    reported: the faithful-Continuum lb results (Sep 2026) were exactly
    this, release-at-admission never firing on session traces."""
    flags = list(flags or [])
    def val(k):
        return flags[flags.index(k) + 1] if k in flags and flags.index(k) + 1 < len(flags) else None
    ret, sch = val("--retention"), val("--scheduling")
    need = []
    # Evolved candidates are exempt: never protecting / never stamping is a
    # legitimate decision rule for a search candidate, not a broken mechanism.
    if ret in ("ttl", "saga-ttl", "min-waste", "oracle-ttl"):
        need += [("protected", "retention never protected a context")]
    if ret in ("ttl", "saga-ttl", "min-waste", "oracle-ttl"):
        need += [("released", "retention never released a protection")]
    if ret == "continuum":
        need += [("protected", "continuum never pinned"),
                 ("admission_releases", "continuum never released a pin at admission")]
    if sch in ("program-fcfs", "plas", "continuum", "oracle-srpt"):
        need += [("stamps", "priority scheduling never stamped a turn")]
    if not mech:
        return ["no mechanism counters found in simulator output"] if need else []
    return [why for k, why in need if not mech.get(k)]


def run_cell(cell, harness_root=None, policy_flags=None, timeout=None,
             keep=False, retries=2):
    """Run one cell. policy_flags None = stock tuple D (no adapter).
    Returns a scorecard dict; on failure 'error' is set and jct_mean
    is None. ASTRA-Sim occasionally aborts (SIGABRT) within seconds of
    start, independent of the workload; such runs are retried."""
    if timeout is None:
        # lb60 leaderboard cells run ~5400s wall; default was too tight.
        timeout = int(os.environ.get("EVOLVE_CELL_TIMEOUT", 5400))
    # Retry only the early abort: a backend that died after running for
    # a while (OOM kill at the job limit, hours in) will die again, and
    # three such attempts cost 9 h on lb60_kvhalf_j0.1 (job 40897537_2).
    early_s = int(os.environ.get("EVOLVE_RETRY_IF_UNDER_S", 900))
    for attempt in range(retries + 1):
        card = _run_cell_once(cell, harness_root, policy_flags, timeout, keep)
        early = (card.get("wall_s") or 0) < early_s
        if not (card["error"] and BACKEND_DEATH in card["stderr_tail"] and early):
            card["attempts"] = attempt + 1
            return card
    card["attempts"] = retries + 1
    return card


def _run_cell_once(cell, harness_root, policy_flags, timeout, keep):
    dataset, cluster = CELLS[cell]
    tag = uuid.uuid4().hex[:10]
    os.makedirs(SCRATCH, exist_ok=True)
    inputs = os.path.join(SCRATCH, "in_" + tag)
    outdir = tempfile.mkdtemp(prefix="cell_" + tag + "_", dir=SCRATCH)
    out_csv = os.path.join(outdir, "run.csv")
    log_dir = os.path.join(outdir, "decisions")
    binds = ["--bind", REPO + ":/app/LLMServingSim",
             "--bind", AS_ROOT + ":/app/agentservesim",
             "--bind", "/orange:/orange",
             "--bind", os.path.dirname(SCRATCH) + ":" + os.path.dirname(SCRATCH)]
    if harness_root is not None:
        binds += ["--bind", harness_root + ":/evolve_harness"]
    cmd = (["apptainer", "exec"] + binds +
           ["--env", "PYTHONPATH=" + PYDEPS, "--pwd", "/app/LLMServingSim", SIF,
            "python3", "-m", "serving",
            "--cluster-config", cluster, "--dataset", dataset,
            "--output", out_csv, "--inputs-root", inputs,
            "--decision-log-dir", log_dir] + ENGINE)
    if harness_root is not None:
        cmd += ["--harness-root", "/evolve_harness"]
    else:
        cmd += ["--harness-root", "/app/agentservesim"]
    if policy_flags:
        cmd += list(policy_flags)
    t0 = time.time()
    card = {"cell": cell, "outdir": outdir if keep else None,
            "flags": list(policy_flags or []), "wall_s": None,
            "jct_mean": None, "jct_p50": None, "jct_p99": None, "n": 0,
            "decisions": None, "error": None, "stderr_tail": "",
            # Which simulator semantics this run used. Without this the
            # utilization semantics of a validated run was unrecoverable
            # (2026-09-12); see docs/sim-vllm-parity.md.
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith("SIM_")},
            # Effective values of defaults that live in the simulator, so an
            # UNSET variable is still recoverable from the card. Must track
            # serving/core/unified_policy_adapter.py -- the adapter audit's rule.
            "env_effective": {
                "SIM_KV_UTIL_SEMANTICS": os.environ.get("SIM_KV_UTIL_SEMANTICS", "vllm"),
                "SIM_GATE_PREFIX_PROBE": os.environ.get("SIM_GATE_PREFIX_PROBE", "0")}}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        card["wall_s"] = round(time.time() - t0, 1)
        log_dir_env = os.environ.get("EVOLVE_LOG_DIR")
        if log_dir_env:
            os.makedirs(log_dir_env, exist_ok=True)
            lp = os.path.join(log_dir_env, f"{cell}_{tag}.log")
            with open(lp, "w") as lf:
                lf.write(" ".join(cmd) + "\n\n" + (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
            card["log"] = lp
        card["stderr_tail"] = (p.stderr or "")[-6000:]
        wf = out_csv[:-4] + "_workflows.csv"
        if p.returncode != 0 or not os.path.exists(wf):
            card["error"] = f"simulator exit {p.returncode}"
            tail = (p.stdout or "")[-1500:]
            card["stderr_tail"] = (card["stderr_tail"] + "\n--- stdout tail ---\n" + tail)[-7000:]
            return card
        j = _read_workflows(wf)
        j.sort()
        mech = _mechanism_stats(p.stdout, log_dir)
        problems = mechanism_check(policy_flags, mech)
        card["mechanism"] = mech
        if problems:
            # Never report a JCT for a policy whose mechanism did not run.
            card["error"] = "mechanism check failed: " + "; ".join(problems)
            card["stderr_tail"] = (card["stderr_tail"] + "\n--- stdout tail ---\n"
                                   + (p.stdout or "")[-1500:])[-7000:]
            return card
        card.update(n=len(j), jct_mean=statistics.fmean(j),
                    jct_p50=j[len(j) // 2],
                    jct_p99=j[min(len(j) - 1, int(round(0.99 * (len(j) - 1))))],
                    decisions=_decision_stats(log_dir))
        return card
    except subprocess.TimeoutExpired:
        card["wall_s"] = round(time.time() - t0, 1)
        card["error"] = f"timeout after {timeout}s"
        return card
    finally:
        if not keep:
            shutil.rmtree(inputs, ignore_errors=True)
            shutil.rmtree(outdir, ignore_errors=True)


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
