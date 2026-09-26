"""Run isolated profile replicas in an allocation; publish no production tables."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import statistics
import shutil
from pathlib import Path
import socket
import subprocess
import sys
import time


def run_group(commands, devices, out, phase):
    processes = []
    try:
        for i, (command, device) in enumerate(zip(commands, devices)):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["PYTHONHASHSEED"] = "0"
            for key in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
                env[key] = str(out / "caches" / str(i) / key.lower())
            log = open(out / f"{phase}_{i}.log", "w")
            p = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            processes.append((p, log))
        while any(p.poll() is None for p, _ in processes):
            if any(p.poll() not in (None, 0) for p, _ in processes):
                raise RuntimeError(f"{phase} failed; inspect {out}/{phase}_*.log")
            time.sleep(2)
        if any(p.returncode != 0 for p, _ in processes):
            raise RuntimeError(f"{phase} failed")
    finally:
        for p, log in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p, log in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--tp", default="1")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--resume-from", type=Path)
    a = p.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if a.resume_from is not None:
        resume_from = a.resume_from.resolve()
        if not resume_from.is_dir():
            raise FileNotFoundError(f"resume source does not exist: {resume_from}")
        manifest_path = resume_from / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"resume source has no manifest.json: {resume_from}"
            )
        prior = json.loads(manifest_path.read_text())
        prior_args = prior.get("args", {})
        # The TP set may differ: resuming exists precisely to add a degree to
        # what a prior run measured. Model and hardware still must match --
        # those decide what the copied CSVs mean.
        expected = {"model": a.model, "hardware": a.hardware}
        actual = {key: prior_args.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"resume source arguments differ: expected {expected}, got {actual}"
            )
        print(f"resuming from tp={prior_args.get('tp')!r} run, now profiling "
              f"tp={a.tp!r}", flush=True)
        for i in range(4):
            source = resume_from / f"replica_{i}"
            if not source.is_dir():
                raise FileNotFoundError(f"missing resume replica: {source}")
            shutil.copytree(source, out / f"replica_{i}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 4 or len(set(visible)) != 4:
        raise RuntimeError("Exactly four distinct allocated GPUs must be visible")
    if any(not device.isdigit() for device in visible):
        raise RuntimeError("This vLLM build requires numeric Slurm GPU IDs")
    # Record UUIDs, but retain Slurm's numeric namespace for vLLM's NVML lookup.
    import torch
    if torch.cuda.device_count() != 4:
        raise RuntimeError("CUDA does not see exactly four GPUs")
    uuids = [str(torch.cuda.get_device_properties(i).uuid) for i in range(4)]
    uuids = [u if u.startswith('GPU-') else 'GPU-' + u for u in uuids]
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs/model" / (a.model + ".json")
    manifest_args = vars(a).copy()
    if manifest_args["resume_from"] is not None:
        manifest_args["resume_from"] = str(manifest_args["resume_from"])
    manifest = {"hostname": socket.gethostname(), "gpu_uuids": uuids,
                "cuda_visible_devices": visible, "args": manifest_args,
                "model_config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                "versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "vllm", "transformers")},
                "cuda": torch.version.cuda, "started": time.time(),
                "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    telemetry_file = open(out / "gpu_telemetry.csv", "w")
    telemetry = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu",
        "--format=csv", "-l", "10"], stdout=telemetry_file)
    try:
        commands = [[sys.executable, "-m", "profiler.smoke", a.model, "--tp", a.tp,
                     "--output", str(out / f"smoke_{i}.json")] for i in range(4)]
        run_group(commands, visible, out, "smoke")
        (out / "SMOKE_PASSED").touch()
        if a.smoke_only:
            return
        commands = [[sys.executable, "-m", "profiler", "profile", a.model,
                     "--hardware", a.hardware, "--tp", a.tp, "--dtype", "bfloat16",
                     "--out-root", str(out / f"replica_{i}"),
                     "--max-num-batched-tokens", "16384", "--max-num-seqs", "128",
                     "--attention-max-kv", "131072", "--measurement-iterations", str(a.iterations)]
                    for i in range(4)]
        run_group(commands, visible, out, "layers")
        (out / "LAYERS_COMPLETE").touch()
        subprocess.run([sys.executable, "-m", "profiler.aggregate",
                        *[str(out / f"replica_{i}" / a.hardware / a.model / "bf16") for i in range(4)],
                        "--out", str(out / "aggregate" / a.hardware / a.model / "bf16")], check=True)
        if "2" in a.tp.split(","):
            commands = [[sys.executable, "experiments/validation/probe_allreduce.py",
                         "--model-config", str(cfg_path), "--output", str(out / f"allreduce_{i}.json")]
                        for i in range(2)]
            run_group(commands, [",".join(visible[:2]), ",".join(visible[2:])], out, "allreduce")
            pairs = [json.loads((out / f'allreduce_{i}.json').read_text()) for i in range(2)]
            if [(r['tokens'], r['bytes']) for r in pairs[0]['results']] != [
                    (r['tokens'], r['bytes']) for r in pairs[1]['results']]:
                raise ValueError('Collective pair coverage differs')
            averaged = []
            for left, right in zip(pairs[0]['results'], pairs[1]['results']):
                mean = statistics.mean([left['us'], right['us']])
                averaged.append({'tokens': left['tokens'], 'bytes': left['bytes'],
                                 'us': mean, 'pair_means_us': [left['us'], right['us']],
                                 'algbw_GBs': left['bytes'] / (mean * 1000)})
            (out / 'allreduce_mean.json').write_text(json.dumps({
                'gpu_pairs': [uuids[:2], uuids[2:]], 'results': averaged}, indent=2))
    finally:
        telemetry.terminate()
        telemetry.wait()
        telemetry_file.close()


if __name__ == "__main__":
    main()
