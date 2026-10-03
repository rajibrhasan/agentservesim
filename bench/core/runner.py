

from __future__ import annotations
import policies as _policies

import argparse
import asyncio
import dataclasses
import csv
import datetime
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

from bench.core import logger as log
from bench.core import recorder
from bench.core import policy_driver as policy_mod
from bench.core.tool_timing import wait_for_tool


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF model id passed verbatim to vllm.AsyncLLM.")
    p.add_argument("--dataset", required=True,
                   help="Path to a LLMServingSim-format JSONL workload "
                        "(produced by `python -m workloads.generators`).")
    p.add_argument("--output-dir", required=True, dest="output_dir",
                   help="Output directory for this run "
                        "(meta.json/requests.jsonl/timeseries.csv).")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   dest="tensor_parallel_size",
                   help="vLLM tensor_parallel_size.")
    p.add_argument("--data-parallel-size", type=int, default=1,
                   dest="data_parallel_size",
                   help="vLLM data_parallel_size (DP across engines).")
    p.add_argument("--enable-expert-parallel", action="store_true",
                   dest="enable_expert_parallel", default=False,
                   help="vLLM enable_expert_parallel for MoE models.")
    p.add_argument("--max-num-seqs", type=int, default=128,
                   dest="max_num_seqs",
                   help="vLLM scheduler max_num_seqs (per-engine running cap).")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048,
                   dest="max_num_batched_tokens",
                   help="vLLM scheduler max_num_batched_tokens.")
    p.add_argument("--max-model-len", type=int, default=None,
                   dest="max_model_len",
                   help="vLLM max_model_len (None = model's max).")
    p.add_argument("--dtype", default="bfloat16",
                   help="Model dtype.")
    p.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction,
                   default=None, help="Explicit async scheduling override for matched experiments.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   dest="kv_cache_dtype",
                   help="vLLM kv_cache_dtype.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                   dest="gpu_memory_utilization",
                   help="Fraction of GPU memory vLLM may use (default: 0.9). "
                        "Lower values shrink the KV pool (retention pressure).")
    p.add_argument("--num-gpu-blocks-override", type=int, default=None,
                   help="Explicit vLLM KV block count for matched-capacity replay.")
    p.add_argument("--block-size", type=int, default=None,
                   help="Explicit vLLM KV block size in tokens.")
    p.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction,
                   dest="enable_prefix_caching", default=True,
                   help="vLLM automatic prefix caching (default: on). "
                        "--no-enable-prefix-caching is the evict-always arm.")
    # Agent serving policy tuple (real side of serving/core/unified_policy_adapter.py).
    p.add_argument("--retention", type=_policies.policy_value("kv"),
                   default="cache-lru",
                   help="retention knob (ttl/min-waste launch the engine with "
                        "VLLM_KV_PROTECTION=1; evict-always disables prefix caching)")
    p.add_argument("--retention-tau", type=float, default=None,
                   help="Protection window in seconds; Continuum defaults to 2; "
                        "ttl and saga-ttl require an explicit value")
    p.add_argument("--retention-gap-default", type=float, default=1.0)
    p.add_argument("--policy-engine-observations", action="store_true",
                   help="Use the v0.19 observation scheduler for live protection "
                        "and min-waste load/tool-history inputs (experimental)")
    p.add_argument("--policy-engine-config", type=Path, default=None,
                   help="Explicit runtime JSON for the isolated vLLM policy engine")
    p.add_argument("--min-waste-profile", type=str, default=None)
    p.add_argument("--scheduling", type=_policies.policy_value("scheduling"),
                   default=None,
                   help="scheduling knob (program-fcfs/plas stamp vLLM priorities); "
                        "overrides --scheduling-policy")
    p.add_argument("--routing", type=_policies.policy_value("routing"),
                   default=None, help="routing knob across --num-instances engines")
    p.add_argument("--routing-capacity-limit", type=int, default=None)
    p.add_argument("--num-instances", type=int, default=1,
                   help="in-process AsyncLLM engines; instance i is pinned to the "
                        "i-th TP-sized slice of CUDA_VISIBLE_DEVICES")
    p.add_argument("--harness-root", type=str, default=None)
    p.add_argument("--decision-log-dir", type=str, default=None,
                   help="per-knob JSONL decision logs (parity input)")
    p.add_argument("--scheduling-policy", choices=["fcfs", "priority"],
                   dest="scheduling_policy", default="fcfs",
                   help="vLLM scheduler policy. 'priority' stamps each turn with "
                        "the program's first-arrival ms (program-FCFS).")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed for vLLM.")
    p.add_argument("--tick-seconds", type=float, default=1.0,
                   dest="tick_seconds",
                   help="Stat logger downsample interval (timeseries.csv row spacing).")
    p.add_argument("--num-reqs", type=int, default=0,
                   dest="num_reqs",
                   help="Cap on number of requests from the dataset (0 = all).")
    p.add_argument("--log-level", default="INFO",
                   dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logger verbosity (default: INFO).")


def run(args: argparse.Namespace) -> int:
    from bench.core.stat_logger import BenchStatLogger

    # Keep the run's own log beside its results, as the simulator does.
    log.configure(args.log_level,
                  log_file=str(Path(args.output_dir) / "run.log"))
    log.print_banner(
        "LLMServingSim Bench",
        f"vLLM end-to-end run -> {args.output_dir}",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests, workflows, sessions = _load_dataset(Path(args.dataset), cap=args.num_reqs)
    if not requests and not workflows and not sessions:
        raise ValueError(f"No requests loaded from {args.dataset}")
    if sessions:
        log.info("Loaded %d agentic sessions from %s (chain mode)",
                 len(sessions), args.dataset)
    elif workflows:
        log.info("Loaded %d DAG workflows from %s (DAG mode)",
                 len(workflows), args.dataset)
    else:
        log.info("Loaded %d requests from %s (flat mode)",
                 len(requests), args.dataset)

    BenchStatLogger.reset()
    asyncio.run(_drive(args, requests, workflows, sessions, output_dir))
    return 0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(path: Path, cap: int = 0) -> tuple[list[dict], list[dict], list[dict]]:
    """Read a LLMServingSim-format JSONL workload.

    Returns ``(flat_requests, dag_workflows, agentic_sessions)``:
    - Rows with ``nodes`` are multi-agent DAG workflows (kept as-is; their
      nodes are replayed dependency-aware, see ``_submit_all_dag``). DAG node
      prompts may omit ``input_tok_ids`` — random ids of the right length are
      synthesized (latency is length-driven, not content-driven).
    - Rows with ``sub_requests`` are linear agentic sessions (SWE-bench-style
      multi-turn chains); replayed turn-sequentially with tool gaps by
      ``_submit_all_chain``. Each sub-request must carry ``input_tok_ids``.
    - Everything else is a flat request; it must carry ``input_tok_ids``
      (bench cannot tokenize on the fly — the dataset tokenizer may differ
      from ``args.model``).
    A workload is all-flat, all-DAG, or all-chain in practice; mixing is allowed.
    """
    requests: list[dict] = []
    workflows: list[dict] = []
    sessions: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "nodes" in row:
                workflows.append(row)
            elif "sub_requests" in row:
                sessions.append(row)
            else:
                if "input_tok_ids" not in row or not row["input_tok_ids"]:
                    raise ValueError(
                        f"Row missing input_tok_ids in {path}; regenerate the "
                        f"dataset with `python -m workloads.generators`."
                    )
                requests.append(row)
            if cap and (len(requests) + len(workflows) + len(sessions)) >= cap:
                break
    return requests, workflows, sessions


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------

async def _drive(args: argparse.Namespace, requests: list[dict],
                 workflows: list[dict], sessions: list[dict],
                 output_dir: Path) -> None:
    # Imports deferred so `validate` / `--help` works without vLLM installed.
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    from bench.core.stat_logger import BenchStatLogger

    from bench.core import policy_driver as pol

    tuple_cfg = pol.TupleConfig(
        retention=args.retention,
        scheduling=args.scheduling or (
            "program-fcfs" if args.scheduling_policy == "priority" else "fcfs"),
        routing=args.routing,
        tau_s=args.retention_tau,
        default_gap_s=args.retention_gap_default,
        min_waste_profile=args.min_waste_profile,
        capacity_limit=args.routing_capacity_limit,
        harness_root=args.harness_root,
        log_dir=args.decision_log_dir,
    )
    tuple_cfg.engine_observations = args.policy_engine_observations
    policy_engine = None
    policy_engine_kwargs = {}
    if args.policy_engine_config is not None:
        policy_engine = json.loads(args.policy_engine_config.read_text())
        policy_engine.setdefault('engine_cls', 'bench.core.policy_engine.PolicyEngine')
        engine_name = policy_engine.get('name')
        if engine_name not in ('autellix', 'infercept', 'saga'):
            raise ValueError('supported policy engines: autellix, infercept, saga')
        from vllm.v1.engine.core import EngineCore
        if not hasattr(EngineCore, 'policy_metrics'):
            raise RuntimeError('policy engine requires the isolated patched vLLM checkout; '
                               'run experiments/validation/prepare_policy_engine.py first')
        if requests or workflows or not sessions:
            raise ValueError('policy engine replay currently requires program chains')
        if engine_name == 'saga':
            # SAGA keeps the stock engine: retention is its tool TTL, the
            # gateway routes/steals/publishes eviction order from observations.
            if args.retention not in ('saga-tool-ttl', 'saga-ttl'):
                raise ValueError('the SAGA engine needs --retention saga-tool-ttl (or saga-ttl)')
            if not args.policy_engine_observations:
                raise ValueError('the SAGA engine needs --policy-engine-observations '
                                 'for per-session cache residency')
            if tuple_cfg.scheduling != 'fcfs' or args.routing:
                raise ValueError('the SAGA gateway owns routing; use FCFS and omit --routing')
        elif (args.retention != 'cache-lru' or tuple_cfg.scheduling != 'fcfs'
                or args.routing):
            raise ValueError('an engine policy owns scheduling and routing; '
                             'use cache-lru and omit tuple overrides')
        tuple_cfg.engine_policy = engine_name
        if engine_name == 'autellix':
            policy_engine_kwargs = {
                'async_scheduling': False,
                'worker_extension_cls': 'bench.core.policy_worker.PolicyWorkerExtension',
                'additional_config': {'agent_policy': policy_engine},
            }
        elif engine_name == 'saga':
            if 'cpu_bytes_per_rank' not in policy_engine:
                raise ValueError('the SAGA engine needs cpu_bytes_per_rank for prefix-migration staging')
            # Stock scheduling plus the acknowledged prefix transport stealing needs.
            policy_engine_kwargs = {
                'async_scheduling': False,
                'scheduler_cls': 'bench.core.saga_scheduler.SagaScheduler',
                'worker_extension_cls': 'bench.core.policy_worker.PolicyWorkerExtension',
                'additional_config': {'agent_policy': {
                    'name': 'kv-migration',
                    'engine_cls': 'bench.core.policy_migration.MigrationHost',
                    'cpu_bytes_per_rank': int(policy_engine['cpu_bytes_per_rank']),
                    'time_model_calls': False}},
            }
        else:
            from .session_driver import infercept_engine_kwargs, validate_session_inputs
            validate_session_inputs(sessions)
            policy_engine_kwargs = infercept_engine_kwargs(
                policy_engine, num_instances=args.num_instances,
                observations_scheduler=args.policy_engine_observations)
    flags = pol.engine_flags(tuple_cfg.retention, tuple_cfg.scheduling)
    if flags["kv_protection"]:
        os.environ["VLLM_KV_PROTECTION"] = "1"
    enable_prefix_caching = args.enable_prefix_caching and flags["enable_prefix_caching"]
    scheduling_policy = flags["scheduling_policy"]
    if tuple_cfg.routing is not None and args.num_instances < 2:
        log.warning("--routing %s with a single instance is a no-op", tuple_cfg.routing)

    scheduler_cls = policy_engine_kwargs.pop(
        'scheduler_cls', 'bench.core.policy_scheduler.PolicyScheduler'
        if args.policy_engine_observations else None)
    if os.environ.get('BENCH_COMPLETION_KV_SNAPSHOT') == '1':
        if os.environ.get('GATE_UTIL_SEMANTICS', 'vllm') != 'vllm':
            raise ValueError('Completion KV snapshots require vllm utilization semantics')
        policy_engine_kwargs.setdefault('additional_config', {})['policy_completion_snapshot'] = True
    if os.environ.get('BENCH_SCHEDULE_TRACE_DIR'):
        diagnostic_dir = str(Path(os.environ['BENCH_SCHEDULE_TRACE_DIR']).resolve())
        policy_engine_kwargs.setdefault('additional_config', {})['schedule_diagnostics'] = {
            'base_class': scheduler_cls, 'directory': diagnostic_dir}
        scheduler_cls = 'bench.core.schedule_diagnostics.DiagnosticScheduler'
    if args.async_scheduling is not None:
        if ('async_scheduling' in policy_engine_kwargs
                and policy_engine_kwargs['async_scheduling'] != args.async_scheduling):
            raise ValueError('Requested async scheduling conflicts with the policy engine')
        policy_engine_kwargs['async_scheduling'] = args.async_scheduling
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        block_size=args.block_size,
        enable_prefix_caching=enable_prefix_caching,
        scheduling_policy=scheduling_policy,
        scheduler_cls=scheduler_cls,
        seed=args.seed,
        disable_log_stats=False,
        **policy_engine_kwargs,
    )
    engine_kwargs_for_meta = _engine_kwargs_for_meta(engine_args)

    engines = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    with log.stage(f"Booting {args.num_instances} AsyncLLM instance(s)"):
        with log.capture_stdio(str(output_dir / "engine.log")):
            for inst in range(args.num_instances):
                if args.num_instances > 1:
                    # Engine subprocesses inherit the environment at spawn,
                    # so pinning is done by slicing CUDA_VISIBLE_DEVICES
                    # per instance before each boot.
                    devs = (visible.split(",") if visible else
                            [str(i) for i in range(
                                args.num_instances * args.tensor_parallel_size)])
                    tp = args.tensor_parallel_size
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                        devs[inst * tp:(inst + 1) * tp])
                engines.append(AsyncLLM.from_engine_args(
                    engine_args, stat_loggers=[BenchStatLogger]))
    if visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    # AsyncLLM discovers supported tasks lazily on its first submission. That
    # worker RPC is engine setup, and must complete before workload arrivals.
    from .readiness import await_engine_readiness
    with log.stage("Checking engine readiness"):
        engine_kwargs_for_meta['readiness_instances'] = await await_engine_readiness(engines)
    engine = engines[0]
    from .effective_config import effective_engine_configs
    engine_kwargs_for_meta['effective_instances'] = effective_engine_configs(engines)
    # Engine KV capacity (tokens): what the simulator's npu_mem must reproduce.
    try:
        cc = engine.vllm_config.cache_config
        engine_kwargs_for_meta["kv_cache_tokens"] = int(cc.num_gpu_blocks) * int(cc.block_size)
        engine_kwargs_for_meta["kv_block_size"] = int(cc.block_size)
    except Exception as e:  # pragma: no cover
        log.warning("could not read KV capacity: %s", e)
    # Publish measured limits before workload arrivals, independently of results.
    startup_path = Path(args.output_dir) / 'engine-startup.json'
    startup_tmp = startup_path.with_suffix('.tmp')
    startup_tmp.write_text(json.dumps(dict(model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dataset_hash=_hash_file(Path(args.dataset)),
        engine_kwargs=engine_kwargs_for_meta), indent=2))
    startup_tmp.replace(startup_path)
    log.info('Engine startup limits: %s', json.dumps(engine_kwargs_for_meta['effective_instances']))
    log.info('GPU memory utilization target: %s; startup metadata: %s',
             args.gpu_memory_utilization, startup_path)
    if os.environ.get('BENCH_CAPACITY_ONLY') == '1':
        for e in engines:
            e.shutdown()
        log.info('Capacity probe complete; engine limits saved without replaying the workload.')
        return
    driver = pol.PolicyDriver(tuple_cfg, engines, asyncio.get_event_loop())
    gateway = None
    if tuple_cfg.engine_policy == 'saga':
        gateway = _saga_gateway(policy_engine, engines, driver, SamplingParams, TokensPrompt)
        driver.saga = gateway
        gateway.start(clock=asyncio.get_event_loop().time)
    policy_summary = None
    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    wf_records = None
    prog_records = None
    try:
        if sessions:
            with log.stage(f"Submitting {len(sessions)} agentic sessions"):
                if tuple_cfg.engine_policy == 'infercept':
                    from .session_driver import submit_all_sessions
                    startup_profile = None
                    if os.environ.get('BENCH_PROFILE_STREAM_DRIVER') == '1':
                        import cProfile
                        startup_profile = cProfile.Profile()
                        startup_profile.enable()
                    try:
                        records, prog_records = await submit_all_sessions(
                            engines, sessions, SamplingParams, driver=driver, log=log)
                    finally:
                        if startup_profile is not None:
                            startup_profile.disable()
                            startup_profile.dump_stats(str(output_dir / 'streaming-driver.prof'))
                else:
                    records, prog_records = await _submit_all_chain(
                        engines, sessions, SamplingParams, TokensPrompt,
                        driver=driver, gateway=gateway,
                    )
            policy_summary = await driver.finish()
            if tuple_cfg.engine_policy == 'infercept':
                policy_summary['infercept_engine'] = [
                    await e.engine_core.call_utility_async('policy_metrics')
                    for e in engines
                ]
                from .kv_measurement import attach_turn_measurements
                attach_turn_measurements(records, policy_summary['infercept_engine'])
            if gateway is not None:
                await gateway.stop()
                policy_summary = dict(policy_summary or {})
                policy_summary['saga_gateway'] = dict(gateway.stats)
                policy_summary['saga_coordinator'] = dict(gateway.coordinator.stats)
        elif workflows:
            with log.stage(f"Submitting {len(workflows)} DAG workflows"):
                records, wf_records = await _submit_all_dag(
                    engine, workflows, SamplingParams, TokensPrompt, args.seed
                )
        else:
            with log.stage(f"Submitting {len(requests)} requests"):
                records = await _submit_all(
                    engine, requests, SamplingParams, TokensPrompt
                )
    finally:
        if gateway is not None:
            await gateway.stop()
        if policy_summary is None:
            policy_summary = await driver.finish()
        with log.stage("Shutting AsyncLLM down"):
            for e in engines:
                e.shutdown()

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------
    # Persist outputs.
    # ------------------------------------------------------------------
    recorder.write_meta(
        output_dir,
        model=args.model,
        vllm_version=_vllm_version(),
        engine_kwargs=engine_kwargs_for_meta,
        dataset_path=str(args.dataset),
        dataset_hash=_hash_file(Path(args.dataset)),
        num_requests=len(records),
        started_at=started_at,
        finished_at=finished_at,
        tick_seconds=args.tick_seconds,
        policy=policy_summary or {"tuple": tuple_cfg.as_dict()},
    )
    recorder.write_requests(output_dir, records)
    if wf_records is not None:
        _write_workflows_csv(output_dir, wf_records)
    if prog_records is not None:
        _write_programs_csv(output_dir, prog_records)
        _write_program_summary(output_dir, prog_records, records)
    header, rows = BenchStatLogger.downsample_to_csv_rows(args.tick_seconds)
    recorder.write_timeseries(output_dir, header, rows)
    if prog_records is not None:
        log.success(
            "%d programs (%d turns), %d timeseries rows -> %s",
            len(prog_records), len(records), len(rows), output_dir,
        )
    elif wf_records is not None:
        log.success(
            "%d workflows (%d nodes), %d timeseries rows -> %s",
            len(wf_records), len(records), len(rows), output_dir,
        )
    else:
        log.success(
            "%d requests, %d timeseries rows -> %s",
            len(records), len(rows), output_dir,
        )


async def _submit_all(engine, requests: list[dict], SamplingParams, TokensPrompt) -> list[dict]:
    """Schedule each request at its arrival offset, gather metrics."""
    loop = asyncio.get_event_loop()
    t0_loop = loop.time()
    completed = [0]  # boxed so the inner closure can mutate

    with log.progress("Requests", total=len(requests)) as bar:

        async def _one(idx: int, req: dict) -> dict:
            target = t0_loop + req["arrival_time_ns"] / 1e9
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            # Strict replay: pin output length to whatever the dataset
            # recorded. ``ignore_eos`` blocks early termination; ``min_tokens``
            # blocks vLLM's async-scheduling early-exit (see vllm/v1/engine/
            # async_llm.py:async-scheduling block) so n_out is exactly fixed.
            n_out = int(req["output_toks"])
            sp = SamplingParams(
                min_tokens=n_out,
                max_tokens=n_out,
                ignore_eos=True,
                temperature=0.0,
            )
            prompt = TokensPrompt(prompt_token_ids=list(req["input_tok_ids"]))
            request_id = f"bench-{idx}"

            last_metrics = None
            async for output in engine.generate(prompt, sp, request_id):
                if output.metrics is not None:
                    last_metrics = output.metrics

            completed[0] += 1
            bar.advance()
            return _record_from_metrics(idx, req, last_metrics)

        tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(requests)]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# DAG (multi-agent workflow) driver
# ---------------------------------------------------------------------------

_VOCAB_LO, _VOCAB_HI = 10, 120_000  # avoid special tokens; safe for Llama/Qwen


async def _submit_all_dag(engine, workflows: list[dict], SamplingParams,
                          TokensPrompt, seed: int) -> tuple[list[dict], list[dict]]:
    """Replay multi-agent DAG workflows, honoring dependencies.

    Each workflow starts at its ``arrival_time_ns``; within it, a node is
    submitted only after ALL its parents have finished (their ``last_token``),
    reproducing the fan-in barrier. Per-workflow JCT is computed purely from
    vLLM-internal ``RequestStateStats`` timestamps:

        JCT = max(node.last_token_ts) - min(root.queued_ts)

    NOTE on clocks: ``RequestStateStats.arrival_time`` is a WALL-CLOCK epoch
    (time.time()), while ``queued_ts`` / ``scheduled_ts`` / ``last_token_ts``
    are MONOTONIC (time.monotonic()). We must not mix them — so the workflow
    start is the roots' ``queued_ts`` (monotonic, ~= submit/arrival time,
    includes queue wait under load), matching the sim's arrival->last-token
    window with no client/HTTP overhead. Returns (node_records, wf_records).
    """
    loop = asyncio.get_event_loop()
    t0_loop = loop.time()
    node_records: list[dict] = []
    wf_records: list[dict] = []

    with log.progress("Workflows", total=len(workflows)) as bar:

        async def _run_workflow(widx: int, wf: dict) -> None:
            wf_id = wf.get("workflow_id", f"wf{widx}")
            arrival_ns = int(wf["arrival_time_ns"])
            delay = (t0_loop + arrival_ns / 1e9) - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            nodes = {n.get("node_id", i): n for i, n in enumerate(wf["nodes"])}
            parents = {nid: [] for nid in nodes}
            for e in wf.get("edges", []):
                parents[e["dst"]].append(e["src"])
            roots = {nid for nid in nodes if not parents[nid]}
            done = {nid: asyncio.Event() for nid in nodes}
            metrics: dict = {}
            base_seed = (hash(wf_id) & 0x7FFFFFFF)

            async def _run_node(nid, node_seed: int) -> None:
                for p in parents[nid]:
                    await done[p].wait()
                spec = nodes[nid]
                n_in = max(1, int(spec["input_toks"]))
                n_out = int(spec["output_toks"])
                ids = spec.get("input_tok_ids")
                if not ids:
                    rng = random.Random(node_seed)
                    ids = [rng.randint(_VOCAB_LO, _VOCAB_HI) for _ in range(n_in)]
                sp = SamplingParams(
                    min_tokens=n_out, max_tokens=n_out,
                    ignore_eos=True, temperature=0.0,
                )
                prompt = TokensPrompt(prompt_token_ids=list(ids))
                last_metrics = None
                async for output in engine.generate(prompt, sp, f"{wf_id}-{nid}"):
                    if output.metrics is not None:
                        last_metrics = output.metrics
                metrics[nid] = last_metrics
                node_records.append(_record_from_metrics_dag(wf_id, nid, spec, last_metrics))
                done[nid].set()

            await asyncio.gather(*(
                _run_node(nid, base_seed + i) for i, nid in enumerate(nodes)
            ))

            # queued_ts (monotonic) — NOT arrival_time (wall epoch) — so it
            # shares last_token_ts's clock. Fall back to scheduled_ts.
            arrivals = [getattr(metrics[n], "queued_ts", None)
                        or getattr(metrics[n], "scheduled_ts", None)
                        for n in roots if metrics.get(n) is not None]
            ends = [getattr(metrics[n], "last_token_ts", None)
                    for n in nodes if metrics.get(n) is not None]
            arrivals = [a for a in arrivals if a is not None]
            ends = [e for e in ends if e is not None]
            jct_ns = int((max(ends) - min(arrivals)) * 1e9) if arrivals and ends else None
            wf_records.append({
                "workflow_id": wf_id,
                "arrival_ns": arrival_ns,
                "end_ns": (arrival_ns + jct_ns) if jct_ns is not None else None,
                "jct_ns": jct_ns,
                "num_nodes": len(nodes),
            })
            bar.advance()

        await asyncio.gather(*(
            _run_workflow(i, w) for i, w in enumerate(workflows)
        ))

    wf_records.sort(key=lambda r: r["arrival_ns"])
    return node_records, wf_records


def _record_from_metrics_dag(wf_id: str, node_id, spec: dict, metrics) -> dict:
    """Per-node timing record (flat schema + workflow/node identity)."""
    rec = {
        "request_id": f"{wf_id}-{node_id}",
        "workflow_id": wf_id,
        "node_id": node_id,
        "input_toks": int(spec["input_toks"]),
        "output_toks": int(spec["output_toks"]),
        "arrival_time": getattr(metrics, "arrival_time", None) if metrics else None,
        "queued_ts": getattr(metrics, "queued_ts", None) if metrics else None,
        "scheduled_ts": getattr(metrics, "scheduled_ts", None) if metrics else None,
        "first_token_ts": getattr(metrics, "first_token_ts", None) if metrics else None,
        "last_token_ts": getattr(metrics, "last_token_ts", None) if metrics else None,
    }
    return rec


def _write_workflows_csv(output_dir: Path, wf_records: list[dict]) -> None:
    """Per-workflow JCT CSV — same schema as the simulator's
    ``<output>_workflows.csv`` so ``compare_sim_real.py`` overlays directly."""
    path = output_dir / "workflows.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workflow_id", "arrival_ns", "end_ns", "jct_ns", "num_nodes"])
        for r in wf_records:
            w.writerow([r["workflow_id"], r["arrival_ns"], r["end_ns"],
                        r["jct_ns"], r["num_nodes"]])


# ---------------------------------------------------------------------------
# Agentic session (linear multi-turn chain) driver
# ---------------------------------------------------------------------------

def _saga_gateway(config, engines, driver, SamplingParams, TokensPrompt):
    """Build the SAGA gateway from explicit config over the booted engines."""
    from policies.saga_runtime import SagaPlacement
    from .saga_coordinator import SagaCoordinator
    from .saga_engine import SagaConfig, SagaGateway

    fields = {f.name for f in dataclasses.fields(SagaConfig)}
    saga_cfg = SagaConfig(**{k: v for k, v in config.items() if k in fields})
    placement = SagaPlacement(seed=int(config.get('seed', 0)),
                              affinity_limit=saga_cfg.affinity_limit,
                              idle_s=saga_cfg.idle_s, load_ratio=saga_cfg.load_ratio)

    async def call(instance, method, *arguments):
        return await engines[instance].engine_core.call_utility_async(method, *arguments)

    async def prefetch(instance, program_id, turn, tokens):
        # Recompute the evicted context ahead of the tool result and hold it
        # until that result is due; the successor's kv_tag then hits the cache.
        tag = f"{program_id}:{turn}-prefetch"
        params = SamplingParams(max_tokens=1, temperature=0.0, extra_args={"kv_tag": tag})
        async for _ in engines[instance].generate(
                TokensPrompt(prompt_token_ids=list(tokens)), params, f"{program_id}-prefetch-{turn}"):
            pass
        ttl = driver.retention_exec.policy.predicted_gap_s(gateway.programs[program_id].tool)
        await call(instance, "kv_protect", tag, time.time() + ttl + saga_cfg.prefetch_margin_s)
        return tag

    retention = driver.retention_exec.policy
    tool_ttl = getattr(retention, "predicted_gap_s", None)
    if saga_cfg.prefetch and tool_ttl is None:
        raise ValueError("SAGA prefetch needs --retention saga-tool-ttl for tool duration estimates")
    gateway = SagaGateway(len(engines), saga_cfg, placement, SagaCoordinator(engines, placement),
                          call=call, generate=prefetch if saga_cfg.prefetch else None,
                          tool_ttl_s=tool_ttl)
    return gateway


async def _submit_all_chain(engines, sessions: list[dict], SamplingParams,
                            TokensPrompt, driver=None, gateway=None,
                            ) -> tuple[list[dict], list[dict]]:
    """Replay linear agentic sessions, honoring turn order and tool gaps.

    Each session starts at its ``arrival_time_ns``; its ``sub_requests`` are
    replayed strictly in order — turn ``i+1`` is submitted only after turn
    ``i`` returns and its ``tool_duration_ns`` gap elapses. One LLM call per
    turn (the agent action); no second "collection" call — matches
    ``scripts/validate_agent.py``.

    Per-program JCT is arrival -> last-token, computed purely from
    vLLM-internal ``RequestStateStats``:

        JCT = last_turn.last_token_ts - arrival_ref

    ``arrival_ref = t0_loop + arrival_ns/1e9`` is on the event loop's clock,
    which is ``time.monotonic()`` — the SAME clock as ``last_token_ts`` — so
    the window includes queue wait under load and has no client/HTTP overhead,
    matching the simulator's arrival->last-completion JCT. Returns
    (turn_records, program_records).
    """
    loop = asyncio.get_event_loop()
    t0_loop = loop.time()
    turn_records: list[dict] = []
    prog_records: list[dict] = []
    # Turns currently submitted to an engine (gateway view of the running
    # set for the admission gate).
    inflight = {"n": 0}
    use_admit = driver is not None and getattr(driver, "has_admit", False)
    if os.environ.get("VLLM_ADMISSION_GATE"):
        # The gate runs inside the engine scheduler (vllm/v1/core/sched/
        # admission_gate.py, the simulator's placement); no gateway hold.
        use_admit = False

    with log.progress("Programs", total=len(sessions)) as bar:

        async def _run_session(sidx: int, sess: dict) -> None:
            sid = sess.get("session_id", f"prog{sidx}")
            arrival_ns = int(sess["arrival_time_ns"])
            arrival_ref = t0_loop + arrival_ns / 1e9
            delay = arrival_ref - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            subs = sess["sub_requests"]
            n = len(subs)
            last_end = None
            program_wait_s = 0.0
            if gateway is not None:
                deadline_ns = sess.get("deadline_ns")
                gateway.register(sid, tenant=sess.get("tenant"),
                                 deadline_s=None if deadline_ns is None
                                 else t0_loop + int(deadline_ns) / 1e9)
            for i, sub in enumerate(subs):
                # Strict replay: pin output length. ``min_tokens`` blocks
                # vLLM's async-scheduling early-exit so n_out is exactly fixed.
                n_out = int(sub["output_toks"])
                kv_tag = f"{sid}:{i}"  # retention handle (see policy.py)
                sp = SamplingParams(
                    min_tokens=n_out,
                    max_tokens=n_out,
                    ignore_eos=True,
                    temperature=0.0,
                    extra_args={"kv_tag": kv_tag},
                )
                prompt = TokensPrompt(prompt_token_ids=list(sub["input_tok_ids"]))
                request_id = f"{sid}-{i}"

                # Turn ready: route, release the program's parked KV,
                # stamp the priority (same event order as the simulator's
                # UnifiedPolicyAdapter.on_turn_routed).
                now = loop.time()
                instance, priority = 0, None
                if driver is not None:
                    instance, priority = await driver.turn_ready(
                        sid, i, now, prompt_tokens=len(sub['input_tok_ids']),
                        completed_tool_duration_s=(
                            int(subs[i - 1].get('tool_duration_ns', 0)) / 1e9
                            if i else None))
                    if driver.cfg.engine_policy:
                        sp.extra_args['program_id'] = sid
                        sp.extra_args['program_service_s'] = driver.programs.get(sid).attained_service_s
                        sp.extra_args['program_wait_s'] = program_wait_s
                    rel = policy_mod.take_deferred_release(sid)
                    if rel is not None:  # engine-side release: at arrival, or at first schedule (Continuum)
                        sp.extra_args["kv_release_tag"] = rel
                        rpol = getattr(getattr(driver, "retention_exec", None), "policy", None)
                        sp.extra_args["kv_release_event"] = getattr(rpol, "release_event", "arrival")
                priority = 0 if priority is None else int(priority)
                ready_callback_finished = loop.time()
                if gateway is not None:
                    sp.extra_args['saga_tenant'] = gateway.programs[sid].tenant
                    sp.extra_args['saga_ready_s'] = now
                    if i:
                        gateway.observe_result(sid, max(
                            0, len(sub["input_tok_ids"]) - len(gateway.programs[sid].history)))
                    placed = await gateway.acquire(sid, i, instance, sub["input_tok_ids"], loop.time())
                    if placed != instance and driver is not None:
                        driver.programs.on_memory_pressure(sid, kv_instance=placed)
                    instance = placed

                # Admission gate (gateway-side hold; mirrors the simulator's
                # filter_waiting): ask the policy every tick until it admits.
                # The turn's arrival is unchanged, so the hold is inside JCT.
                holds = 0
                if use_admit:
                    n_prompt = len(sub["input_tok_ids"])
                    while True:
                        if await driver.admit(sid, i, n_prompt, inflight["n"], loop.time()):
                            break
                        holds += 1
                        await asyncio.sleep(driver.ADMIT_TICK_S)

                last_metrics = None
                cached_tokens = None
                kv_snapshot = None
                output_ids = []
                inflight["n"] += 1
                try:
                    async for output in engines[instance].generate(
                            prompt, sp, request_id, priority=priority):
                        if output.metrics is not None:
                            last_metrics = output.metrics
                        if getattr(output, "num_cached_tokens", None) is not None:
                            cached_tokens = output.num_cached_tokens
                        output_ids = list(output.outputs[0].token_ids)
                        if os.environ.get('BENCH_COMPLETION_KV_SNAPSHOT') == '1':
                            kv_snapshot = output.policy_kv_snapshot
                finally:
                    inflight["n"] -= 1
                completion_received_ts = loop.time()
                rec = _record_from_metrics_chain(sid, i, sub, last_metrics)
                rec["cached_tokens"] = cached_tokens  # prefix-cache hit reported by the engine
                rec["admission_holds"] = holds
                rec['completed_tool_duration_s'] = (
                    int(subs[i - 1].get('tool_duration_ns', 0)) / 1e9 if i else None)
                rec['policy_ready_ts'] = now
                rec['ready_callback_finished_ts'] = ready_callback_finished
                rec['completion_received_ts'] = completion_received_ts
                if os.environ.get('BENCH_COMPLETION_KV_SNAPSHOT') == '1':
                    if kv_snapshot is None:
                        raise RuntimeError('Patched engine did not return a completion KV snapshot')
                    rec['completion_kv_snapshot'] = dict(kv_snapshot, instance=instance)
                turn_records.append(rec)
                service_s = 0.0
                if last_metrics is not None:
                    lt = getattr(last_metrics, "last_token_ts", None)
                    st = getattr(last_metrics, "scheduled_ts", None)
                    if lt is not None:
                        last_end = lt
                        if st is not None:
                            service_s = max(0.0, lt - st)
                completion_ts = (rec['last_token_ts'] if rec['last_token_ts'] is not None
                                 else completion_received_ts)
                rec['completion_callback_started_ts'] = loop.time()
                if driver is not None:
                    if driver.cfg.engine_policy == 'autellix':
                        measured = await engines[instance].engine_core.call_utility_async(
                            'policy_metrics', kv_tag)
                        service_s = measured['execution_s']
                        program_wait_s += measured['wait_s']
                        rec['model_execution_s'] = service_s
                        rec['engine_queue_wait_s'] = measured['wait_s']
                    await driver.turn_complete(
                        sid, i, kv_tag, service_s,
                        int(sub["input_toks"]) + n_out, instance,
                        completion_ts,
                        tool_name=sub.get("tool"), kv_snapshot=kv_snapshot)
                if gateway is not None:
                    gateway.release(sid, instance, loop.time(), service_s, sub["input_tok_ids"],
                                    output_ids, tool=sub.get("tool"), last_turn=(i == n - 1))

                rec['completion_callback_finished_ts'] = loop.time()
                # The tool clock starts at completion, concurrently with host
                # bookkeeping. Preserve callback ordering and real lateness;
                # do not sleep the full duration again after callbacks finish.
                if i < n - 1:
                    gap_ns = int(sub.get("tool_duration_ns", 0))
                    rec['tool_ready_target_ts'] = await wait_for_tool(
                        completion_ts, gap_ns, clock=loop.time)
                    rec['tool_wait_finished_ts'] = loop.time()

            jct_ns = int((last_end - arrival_ref) * 1e9) if last_end is not None else None
            prog_records.append({
                "program_id": sid,
                "arrival_ns": arrival_ns,
                "jct_ns": jct_ns,
                "num_turns": n,
            })
            bar.advance()

        await asyncio.gather(*(
            _run_session(i, s) for i, s in enumerate(sessions)
        ))

    prog_records.sort(key=lambda r: r["arrival_ns"])
    return turn_records, prog_records


def _record_from_metrics_chain(sid: str, turn_idx: int, sub: dict, metrics) -> dict:
    """Per-turn timing record (flat schema + program/turn identity)."""
    return {
        "request_id": f"{sid}-{turn_idx}",
        "program_id": sid,
        "turn_idx": turn_idx,
        "input_toks": int(sub["input_toks"]),
        "output_toks": int(sub["output_toks"]),
        "arrival_time": getattr(metrics, "arrival_time", None) if metrics else None,
        "queued_ts": getattr(metrics, "queued_ts", None) if metrics else None,
        "scheduled_ts": getattr(metrics, "scheduled_ts", None) if metrics else None,
        "first_token_ts": getattr(metrics, "first_token_ts", None) if metrics else None,
        "last_token_ts": getattr(metrics, "last_token_ts", None) if metrics else None,
    }


def _pctl(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (matches bench/core/plots.py and the
    simulator's router._percentile)."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return float(sorted_vals[lo])
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _write_programs_csv(output_dir: Path, prog_records: list[dict]) -> None:
    """Per-program JCT CSV — one row per program (the key comparison file)."""
    path = output_dir / "per_program.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["program_id", "arrival_s", "jct_s", "num_turns"])
        for r in prog_records:
            jct_s = r["jct_ns"] / 1e9 if r["jct_ns"] is not None else ""
            w.writerow([r["program_id"], r["arrival_ns"] / 1e9,
                        jct_s, r["num_turns"]])


def _write_program_summary(output_dir: Path, prog_records: list[dict],
                           req_records: list[dict] | None = None) -> None:
    """Bench-style JCT summary: n, mean, p50, p90, p95, p99 (seconds).

    Also reports prefix reuse and the observed tool gaps. Both were only
    recoverable by post-processing requests.jsonl, and a replay whose cache
    behaviour differs from the simulator's looks identical in a JCT-only
    summary -- which is how a 7x divergence in Continuum's pin decisions went
    unnoticed until the logs were joined by hand (2026-09-19).
    """
    jcts = sorted(r["jct_ns"] / 1e9 for r in prog_records if r["jct_ns"] is not None)
    path = output_dir / "summary.txt"
    with path.open("w") as f:
        f.write("Per-Program Job Completion Time (arrival -> last token)\n")
        f.write(f"programs         : {len(prog_records)}\n")
        f.write(f"programs with JCT: {len(jcts)}\n")
        if jcts:
            mean = sum(jcts) / len(jcts)
            f.write(f"mean JCT (s)     : {mean:.3f}\n")
            f.write(f"p50  JCT (s)     : {_pctl(jcts, 50):.3f}\n")
            f.write(f"p90  JCT (s)     : {_pctl(jcts, 90):.3f}\n")
            f.write(f"p95  JCT (s)     : {_pctl(jcts, 95):.3f}\n")
            f.write(f"p99  JCT (s)     : {_pctl(jcts, 99):.3f}\n")
        if req_records:
            reuse_metric = all(r.get('cache_measurement') == 'input_kv_reuse'
                               for r in req_records)
            cache_key = 'kv_reused_tokens' if reuse_metric else 'cached_tokens'
            tin = sum(int(r.get("input_toks") or 0) for r in req_records)
            cold = [r for r in req_records if r.get("turn_idx") == 0]
            warm = [r for r in req_records if r.get("turn_idx") != 0]
            measured = sum(r.get(cache_key) is not None for r in req_records)

            def cache_rate(rows, counts=False):
                if any(r.get(cache_key) is None for r in rows):
                    return "not measured"
                inputs = sum(int(r.get("input_toks") or 0) for r in rows)
                hits = sum(int(r[cache_key]) for r in rows)
                if not inputs:
                    return "n/a"
                rate = f"{hits / inputs * 100:.2f}%"
                return rate + (f"  ({hits}/{inputs})" if counts else "")

            f.write("\nInput KV reuse (GPU-resident or CPU-restored; excludes recomputed tokens)\n"
                    if reuse_metric else "\nPrefix cache (engine-reported cached_tokens)\n")
            f.write(f"requests         : {len(req_records)}\n")
            f.write(f"input tokens     : {tin}\n")
            tc = (str(sum(int(r[cache_key]) for r in req_records))
                  if measured == len(req_records) else "not measured")
            f.write(f"{'reused tokens' if reuse_metric else 'cached tokens':17}: {tc}\n")
            f.write(f"hit rate         : {cache_rate(req_records)}\n")
            if measured != len(req_records):
                f.write(f"cache measurements: {measured}/{len(req_records)} requests\n")
            if cold:
                f.write(f"  turn 0 (cold)  : {cache_rate(cold, counts=True)}\n")
            if warm:
                f.write(f"  turns 1+       : {cache_rate(warm)}\n")
            if reuse_metric:
                for key in ('gpu_reused_tokens', 'cpu_restored_tokens',
                            'prefill_computed_tokens', 'recomputed_context_tokens'):
                    f.write(f"{key}: {sum(r[key] for r in req_records)}\n")
            gaps = sorted(float(r["completed_tool_duration_s"]) for r in req_records
                          if r.get("completed_tool_duration_s") is not None)
            if gaps:
                over = sum(1 for g in gaps if g > 2.0)
                f.write("\nTool gaps fed to the policy (declared, not elapsed)\n")
                f.write(f"observations     : {len(gaps)}\n")
                f.write(f"p50 / p90 (s)    : {_pctl(gaps, 50):.3f} / {_pctl(gaps, 90):.3f}\n")
                f.write(f"over 2.0 s       : {over} ({over / len(gaps) * 100:.1f}%)\n")


def _record_from_metrics(idx: int, req: dict, metrics) -> dict:
    """Project ``RequestStateStats`` onto our flat per-request schema."""
    if metrics is None:
        return {
            "request_id": f"bench-{idx}",
            "input_toks": int(req["input_toks"]),
            "output_toks": int(req["output_toks"]),
            "arrival_time": None,
            "queued_ts": None,
            "scheduled_ts": None,
            "first_token_ts": None,
            "last_token_ts": None,
        }
    return {
        "request_id": f"bench-{idx}",
        "input_toks": int(req["input_toks"]),
        "output_toks": int(req["output_toks"]),
        "arrival_time": getattr(metrics, "arrival_time", None),
        "queued_ts": getattr(metrics, "queued_ts", None),
        "scheduled_ts": getattr(metrics, "scheduled_ts", None),
        "first_token_ts": getattr(metrics, "first_token_ts", None),
        "last_token_ts": getattr(metrics, "last_token_ts", None),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_kwargs_for_meta(engine_args) -> dict:
    fields = (
        "model", "tensor_parallel_size", "data_parallel_size",
        "enable_expert_parallel", "max_num_seqs", "max_num_batched_tokens",
        "max_model_len", "dtype", "kv_cache_dtype", "seed",
        "scheduler_cls", "async_scheduling", "worker_extension_cls", "additional_config",
    )
    return {k: getattr(engine_args, k, None) for k in fields}


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
