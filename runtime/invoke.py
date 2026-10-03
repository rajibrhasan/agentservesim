import os
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid

from .cells import CELLS, resolve
from .mechanism import (BACKEND_DEATH, _decision_stats, _mechanism_stats,
                        _read_workflows, mechanism_check)
from .paths import AS_ROOT, ENGINE, PYDEPS, REPO, SCRATCH, SIF

#: How the simulator is launched. "apptainer" and "docker" both run the image
#: named by SIM_IMAGE (or the .sif at runtime.paths.SIF); "none" runs python
#: directly, for a host that already has ASTRA-Sim and the deps importable.
CONTAINER = os.environ.get("SIM_CONTAINER", "apptainer")
IMAGE = os.environ.get("SIM_IMAGE", "")


def _require_docker_image(image):
    """Fail with the command that creates the image, not with docker's
    "Unable to find image" -- which is true but does not say that this image is
    built locally from the repo's Dockerfile rather than pulled."""
    import subprocess
    try:
        subprocess.run(["docker", "image", "inspect", image],
                       check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(
            "SIM_CONTAINER=docker but docker is not on PATH. Use "
            "SIM_CONTAINER=apptainer, or SIM_CONTAINER=none to run the "
            "simulator directly.") from None
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"docker image {image!r} not found. It is built from this "
            f"repository, not pulled: run ./scripts/docker-sim.sh (which "
            f"builds it on first use), or `docker build -t {image} .`") from None


def _container_cmd(mounts):
    """Argv prefix that puts the simulator in front of the given mounts.

    Apptainer was the only supported runtime, hardcoded here, while the README
    tells people to install with Docker -- so following the instructions and
    then running a search failed on a call nobody had mentioned.
    """
    if CONTAINER == "none":
        return []
    if CONTAINER == "docker":
        _require_docker_image(IMAGE or "agentservesim:latest")
        args = ["docker", "run", "--rm"]
        for src, dst in mounts:
            args += ["-v", f"{src}:{dst}"]
        args += ["-e", "PYTHONPATH=" + PYDEPS, "-w", "/app/LLMServingSim",
                 IMAGE or "agentservesim:latest"]
        return args
    args = ["apptainer", "exec"]
    for src, dst in mounts:
        args += ["--bind", f"{src}:{dst}"]
    args += ["--env", "PYTHONPATH=" + PYDEPS, "--pwd", "/app/LLMServingSim",
             IMAGE or SIF]
    return args


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


def _env_effective():
    """Effective values of the simulator's own defaults, so an UNSET variable
    is still recoverable from a scorecard.

    Read from the adapter rather than restated here. They were restated, and
    on 2026-09-15 the adapter's SIM_GATE_PREFIX_PROBE default changed from 0
    to 1 while this copy did not -- so cards claimed a setting the run had not
    used. That is the same ambiguity that made an earlier gate number
    unresolvable: its card predated this field entirely, and nothing said
    which way the probe had been set.
    """
    out = {}
    try:
        from serving.core import unified_policy_adapter as _a
        out["SIM_KV_UTIL_SEMANTICS"] = _a._KV_UTIL_SEMANTICS
        out["SIM_GATE_PREFIX_PROBE"] = "1" if _a._GATE_PREFIX_PROBE else "0"
    except Exception:
        # The adapter imports pandas via trace_generator; a caller without it
        # still gets the environment, just not the resolved defaults.
        out["SIM_KV_UTIL_SEMANTICS"] = os.environ.get("SIM_KV_UTIL_SEMANTICS", "unknown")
        out["SIM_GATE_PREFIX_PROBE"] = os.environ.get("SIM_GATE_PREFIX_PROBE", "unknown")
    try:
        from serving.core import scheduler as _s
        out["SIM_ADMIT_VALVE"] = "1" if _s._ADMIT_VALVE else "0"
    except Exception:
        out["SIM_ADMIT_VALVE"] = os.environ.get("SIM_ADMIT_VALVE", "unknown")
    try:
        from serving.core import router as _r
        out["SIM_DERIVE_OUTPUT_IDS"] = "1" if _r._DERIVE_FROM_SUCCESSOR else "0"
    except Exception:
        out["SIM_DERIVE_OUTPUT_IDS"] = os.environ.get("SIM_DERIVE_OUTPUT_IDS", "unknown")
    out["SIM_KV_INVARIANT"] = os.environ.get("SIM_KV_INVARIANT", "0")
    return out


def _run_cell_once(cell, harness_root, policy_flags, timeout, keep):
    dataset, cluster = resolve(cell)
    tag = uuid.uuid4().hex[:10]
    os.makedirs(SCRATCH, exist_ok=True)
    inputs = os.path.join(SCRATCH, "in_" + tag)
    # Create it rather than assume it. SCRATCH defaults to a /dev/shm path that
    # exists on this cluster and nowhere else, and mkdtemp's failure names a
    # temp directory rather than the setting that chose it.
    os.makedirs(SCRATCH, exist_ok=True)
    outdir = tempfile.mkdtemp(prefix="cell_" + tag + "_", dir=SCRATCH)
    out_csv = os.path.join(outdir, "run.csv")
    log_dir = os.path.join(outdir, "decisions")
    # One bind: since the 2026-09-12 unification REPO holds the engine AND
    # the contract, so the second mount (AS_ROOT -> /app/agentservesim) would
    # be the same host directory under a second name. The container path
    # /app/LLMServingSim is deliberately unchanged -- the simulator resolves
    # cluster configs, datasets and profiler tables relative to it.
    # Order is part of the command line the golden test pins: the measured
    # numbers are only comparable across runs if the invocation is identical.
    mounts = [(REPO, "/app/LLMServingSim")]
    if os.path.isdir("/orange"):
        mounts.append(("/orange", "/orange"))       # site trace storage, if present
    mounts.append((os.path.dirname(SCRATCH), os.path.dirname(SCRATCH)))
    if harness_root is not None:
        mounts.append((harness_root, "/evolve_harness"))
    cmd = (_container_cmd(mounts) +
           ["python3", "-m", "serving",
            "--cluster-config", cluster, "--dataset", dataset,
            "--output", out_csv, "--inputs-root", inputs,
            "--decision-log-dir", log_dir] + ENGINE)
    if harness_root is not None:
        cmd += ["--harness-root", "/evolve_harness"]
    else:
        cmd += ["--harness-root", "/app/LLMServingSim"]
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
            # These controls do not use the SIM_ prefix but change latency.
            # Record defaults too so a missing table's fallback is recoverable.
            "timing_env": {key: os.environ.get(key, default) for key, default in {
                "STEP_OVERHEAD_MODE": "auto", "HOST_OVERHEAD_BASE_NS": "0",
                "HOST_OVERHEAD_PER_SEQ_NS": "0", "MIX_OVERHEAD_PER_SEQ_NS": "0",
                "MIX_GATE_MODE": "flat", "MIX_GATE_MAX_SEQS": "128",
                "MIX_GATE_EXP": "1.0"}.items()},
            # Effective values of defaults that live in the simulator, so an
            # UNSET variable is still recoverable from the card. Must track
            # serving/core/unified_policy_adapter.py -- the adapter audit's rule.
            "env_effective": _env_effective()}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        card["wall_s"] = round(time.time() - t0, 1)
        # The simulator's own log is its stdout, and it carries far more than a
        # scorecard keeps: per-iteration batch composition, cache hits, memory.
        # Always write it beside the run, so `keep=True` preserves it; the
        # 2026-09-17 board run kept only mean/p50/p99 and no distribution or
        # cache question about that cell can be answered from what survived.
        body = (" ".join(cmd) + "\n\n" + (p.stdout or "")
                + "\n--- stderr ---\n" + (p.stderr or ""))
        with open(os.path.join(outdir, "run.log"), "w") as lf:
            lf.write(body)
        # EVOLVE_LOG_DIR additionally keeps a copy somewhere durable, for
        # callers whose outdir is scratch they are about to delete.
        log_dir_env = os.environ.get("EVOLVE_LOG_DIR")
        if log_dir_env:
            os.makedirs(log_dir_env, exist_ok=True)
            lp = os.path.join(log_dir_env, f"{cell}_{tag}.log")
            with open(lp, "w") as lf:
                lf.write(body)
            card["log"] = lp
        else:
            card["log"] = os.path.join(outdir, "run.log") if keep else None
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
