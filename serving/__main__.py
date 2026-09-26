"""Simulation entry point: ``python -m serving --cluster-config <...> [...]``.

Parses CLI args, generates ASTRA-Sim input files via ``serving.core.config_builder``,
spawns the ASTRA-Sim subprocess, and runs the iteration loop:
``router.route -> scheduler.schedule -> trace_generator -> graph -> ASTRA-Sim
-> scheduler.add_done`` until every request completes.
"""

import os
import subprocess
import argparse
import json
import shutil
from time import time
from collections import defaultdict

from serving.core.scheduler import *
from serving.core.request import *
from serving.core.utils import *
from serving.core.controller import *
from serving.core.memory_model import *
from serving.core.graph_generator import *
from serving.core.trace_generator import *
from serving.core.pim_model import *
from serving.core.config_builder import *
from serving.core.router import *
from serving.core.power_model import *
from serving.core.logger import *
from serving.core.run_paths import build_run_paths, resolve_run_id
from serving.core.idle_sweep import IdleScheduleSweep
import sys as flush

from pyinstrument import Profiler


def _pad_batch_to_max(batch, max_len):
    """Pad a batch up to ``max_len`` for DP-sync.

    Mirrors vLLM's CUDA-graph DP padding: every DP rank's forward runs at
    ``max(num_tokens_across_dp)``. We bump the high-level counters so
    dense layers, lm_head, and the MoE compute path all reflect the
    padded shape — but we deliberately leave ``decode_k_list`` /
    prefill lists untouched so attention continues to see only the real
    decodes. FlashAttention's varlen kernel gives padded ``seq_len=0``
    entries zero compute in real vLLM, and extending ``decode_k_list``
    with ``kv=1`` dummies would instead collapse ``kv_decode_mean``
    toward 1 and push the attention lookup far outside the profiled
    sweep.

    MoE AG/RS comm size is anchored separately to ``max_total_len`` (no
    ``× group_size``) in the iteration loop — that calibrates the
    bandwidth model against the same ``link_bw`` AllReduce already uses.

    Request-completion accounting (`scheduler.add_done`) reads
    ``batch.requests`` and ``batch.end``, not these mutated token-list
    fields, so it is unaffected.
    """
    pad = max_len - batch.total_len
    if pad <= 0:
        return
    batch.total_len = max_len
    batch.kv_len += pad                  # each dummy contributes kv=1
    batch.num_decode += pad              # counted for lm_head / dense shape


def _runtime_limit(value):
    return float('inf') if value == 0 else value


def _cluster_config_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join("..", path)


def _load_cluster_config_for_overrides(path):
    with open(_cluster_config_path(path), "r") as f:
        return json.load(f)


def _resolve_output_file(path, run_id):
    if path is None:
        return None
    return path.replace("{run_id}", run_id)


def _cleanup_inputs_root(run_paths, logger):
    """Remove generated ASTRA-Sim inputs after a completed simulation."""
    runs_root = os.path.abspath(os.path.join("inputs", "runs"))
    inputs_root = os.path.abspath(run_paths.inputs_root)
    if inputs_root in (os.path.abspath("inputs"), runs_root):
        raise RuntimeError(f"Refusing to remove broad inputs root: {inputs_root}")
    if not inputs_root.startswith(runs_root + os.sep):
        logger.warning(
            "Skipping ASTRA-Sim inputs cleanup because inputs_root is outside %s: %s",
            runs_root, inputs_root,
        )
        return
    shutil.rmtree(inputs_root, ignore_errors=True)
    logger.info("Removed ASTRA-Sim inputs root: %s", inputs_root)


def _prepare_ns3_config(astra_sim, run_paths):
    template = os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/config/config.txt")
    output_dir = os.path.join(run_paths.inputs_root, "ns3", "output")
    config_path = os.path.join(run_paths.inputs_root, "ns3", "config.txt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    replacements = {
        "FLOW_FILE": os.path.join(output_dir, "flow.txt"),
        "TRACE_FILE": os.path.join(output_dir, "trace.txt"),
        "TRACE_OUTPUT_FILE": os.path.join(output_dir, "mix.tr"),
        "FCT_OUTPUT_FILE": os.path.join(output_dir, "fct.txt"),
        "PFC_OUTPUT_FILE": os.path.join(output_dir, "pfc.txt"),
        "QLEN_MON_FILE": os.path.join(output_dir, "qlen.txt"),
    }

    for path in (replacements["FLOW_FILE"], replacements["TRACE_FILE"]):
        open(path, "w").close()

    with open(template, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and parts[0] in replacements:
                f.write(f"{parts[0]} {replacements[parts[0]]}\n")
            else:
                f.write(line)
    return config_path


def _iter_raw_instances(cluster_config):
    for node in cluster_config.get("nodes", []):
        for instance in node.get("instances", []):
            yield instance


def _resolve_instance_dtype(instance, cli_dtype, dtype_to_bits):
    dtype = instance.get("dtype", cli_dtype)
    if dtype is None:
        config = get_config(instance["model_name"])
        torch_dtype = config.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype in dtype_to_bits:
            dtype = torch_dtype
        else:
            dtype = "bfloat16"
    if dtype not in dtype_to_bits:
        raise ValueError(f"Unsupported dtype '{dtype}' for instance {instance.get('instance_id')}")
    return dtype


def _build_instance_runtime_configs(instances, args, dtype_to_bits):
    runtime_configs = []

    for instance_id, instance in enumerate(instances):
        dtype = _resolve_instance_dtype(instance, args.dtype, dtype_to_bits)
        kv_cache_dtype = instance.get("kv_cache_dtype", args.kv_cache_dtype)
        if kv_cache_dtype not in ("auto", "fp8"):
            raise ValueError(f"Unsupported kv_cache_dtype '{kv_cache_dtype}' for instance {instance_id}")

        enable_attn_offloading = instance.get("enable_attn_offloading", args.enable_attn_offloading)
        enable_sub_batch_interleaving = instance.get(
            "enable_sub_batch_interleaving", args.enable_sub_batch_interleaving)
        if enable_sub_batch_interleaving and not enable_attn_offloading:
            raise RuntimeError(
                f"Instance {instance_id} enables sub-batch interleaving without attention offloading")

        runtime_configs.append({
            "max_num_seqs": _runtime_limit(instance.get("max_num_seqs", args.max_num_seqs)),
            "max_num_batched_tokens": _runtime_limit(
                instance.get("max_num_batched_tokens", args.max_num_batched_tokens)),
            "long_prefill_token_threshold": instance.get(
                "long_prefill_token_threshold", args.long_prefill_token_threshold),
            "block_size": instance.get("block_size", args.block_size),
            "dtype": dtype,
            "fp": dtype_to_bits[dtype],
            "kv_cache_dtype": kv_cache_dtype,
            "enable_chunked_prefill": instance.get(
                "enable_chunked_prefill", args.enable_chunked_prefill),
            "enable_prefix_caching": instance.get(
                "enable_prefix_caching", args.enable_prefix_caching),
            "prioritize_prefill": instance.get("prioritize_prefill", args.prioritize_prefill),
            "enable_local_offloading": instance.get(
                "enable_local_offloading", args.enable_local_offloading),
            "enable_attn_offloading": enable_attn_offloading,
            "enable_sub_batch_interleaving": enable_sub_batch_interleaving,
            "enable_block_copy": instance.get("enable_block_copy", args.enable_block_copy),
        })
    return runtime_configs


def _policy_choices(axis):
    """Values `--retention` / `--scheduling` / `--routing` accept, read from
    `policies.VALUES`. Hardcoding them meant adding a policy took three edits
    -- the class, this list, and the adapter's if-chain -- while the registry
    that was supposed to be the single source of truth was read by nothing."""
    try:
        import policies
        return policies.choices_for(axis)
    except Exception:
        return {"kv": ['evict-always', 'cache-lru', 'ttl', 'min-waste',
                       'continuum', 'saga-ttl', 'evolved', 'oracle-ttl'],
                "scheduling": ['fcfs', 'program-fcfs', 'plas', 'continuum',
                               'evolved', 'oracle-srpt'],
                "routing": ['rr', 'least-loaded', 'session-affinity']}[axis]


def _policy_value(axis):
    """argparse type for a policy flag: a registry name, or `module:Class`.

    `choices=` cannot express "one of these, or any importable class", and the
    open half is the point: a policy someone writes should not require an edit
    to this repository to be runnable. `--policy` already spells it
    `module:Class`, so the axis flags spell it the same way.
    """
    known = _policy_choices(axis)

    def parse(value):
        if value in known or ":" in value:
            return value
        raise argparse.ArgumentTypeError(
            f"{value!r} is not one of {known}, and is not module:Class "
            f"(e.g. mypolicy:MyRetention)")
    return parse


def _autellix_queues(args):
    """The MLFQ queue parameters as the driver wants them, or None.

    Parsed here so a malformed list fails at argument time rather than on the
    first scheduling tick, and so the driver can refuse to invent defaults.
    """
    if args.scheduling != "autellix-mlfq":
        return None
    def _list(raw, flag):
        if not raw:
            return None
        try:
            return [float(x) for x in str(raw).split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"{flag} wants comma-separated seconds, got {raw!r}")
    return (_list(args.autellix_service_boundaries, "--autellix-service-boundaries"),
            _list(args.autellix_quanta, "--autellix-quanta"),
            args.autellix_starvation_ratio)


def main():
    # ----------------------------------------------------------------------------------------------
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)

    # -------------------------------------- Argument parsing --------------------------------------
    parser = argparse.ArgumentParser(prog='python -m serving',
                                     description='LLMServingSim') 
    
    parser.add_argument('--cluster-config', type=str, default='configs/cluster/single_node_single_instance.json',
                        help='path to cluster config JSON defining node topology, instance layout, hardware, and memory hierarchy')
    parser.add_argument('--max-num-seqs', type=int, default=128,
                        help='maximum number of sequences in a batch (0 = unlimited)')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048,
                        help='maximum number of tokens processed per iteration across all requests (the total token budget). '
                        'With chunked prefill, long inputs are split across iterations; '
                        'without chunked prefill, this effectively caps max input length')
    parser.add_argument('--long-prefill-token-threshold', type=int, default=0,
                        help='per-request token cap per step for chunked prefill (0 = disabled). '
                        'Limits how many tokens a single prefill request consumes per iteration, '
                        'preventing long prompts from monopolizing the token budget. '
                        'When 0, a single prefill can consume the entire budget')
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32', 'fp8', 'int8'], default=None,
                        help='model weight data type (vLLM-style). When omitted, defaults to the model config\'s '
                        '``torch_dtype`` (falling back to bfloat16). Overrides only take effect if the profiler '
                        'produced matching data under perf/<hw>/<model>/<variant>/tp<N>/')
    parser.add_argument('--request-routing-policy', type=str, choices=['LOAD', 'RR', 'RAND', 'CUSTOM'], default='LOAD',
                        help='request routing policy across instances: LOAD (vLLM-style weighted least-loaded, default), '
                        'RR (round-robin), RAND (random), CUSTOM (user-defined)')
    parser.add_argument('--expert-routing-policy', type=str,
                        choices=['BALANCED', 'RR', 'RAND', 'CUSTOM'],
                        default='BALANCED',
                        help='expert token routing policy for MoE models: '
                        'BALANCED (default; analytical pigeonhole approximation of '
                        'a trained load-balanced learned gate), '
                        'RR (round-robin), RAND (uniform random per token), '
                        'CUSTOM (user-defined)')
    parser.add_argument('--enable-block-copy', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Replay one transformer block\'s trace across every '
                        'layer instead of re-computing the routing per layer — '
                        'cuts trace-generation time roughly num_hidden_layers× '
                        'on MoE models. Safe with BALANCED (deterministic); '
                        'RR/RAND get a small per-layer variance averaged out. '
                        'Disable only for CUSTOM policies that need faithful '
                        'per-layer variance.')
    parser.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=True,
                        help='enable prefix caching via RadixAttention to reuse KV cache across requests '
                        'with shared prefixes (default: enabled). Use --no-enable-prefix-caching to disable')
    parser.add_argument('--enable-chunked-prefill', action=argparse.BooleanOptionalAction, default=True,
                        help='enable chunked prefill to split long prefill requests across multiple iterations, '
                        'matching vLLM v1 behavior (default: enabled). Use --no-enable-chunked-prefill to disable')
    parser.add_argument('--enable-prefix-sharing', action='store_true', default=False,
                        help='enable second-tier prefix cache pooling across instances within a node')
    parser.add_argument('--prefix-storage', type=str, choices=['None', 'CPU', 'CXL'], default='None',
                        help='storage medium for the second-tier prefix cache pool: None (NPU only), CPU, or CXL')
    parser.add_argument('--enable-local-offloading', action='store_true', default=False,
                        help='enable weight offloading to local (NPU) memory. '
                        'Recommended to disable unless weight memory access is not counted in profiling')
    parser.add_argument('--enable-attn-offloading', action='store_true', default=False,
                        help='enable attention computation offloading to PIM (Processing-In-Memory) devices')
    parser.add_argument('--enable-sub-batch-interleaving', action='store_true', default=False,
                        help='enable sub-batch interleaving to overlap XPU and PIM computation. '
                        'Requires --enable-attn-offloading')
    parser.add_argument('--prioritize-prefill', action='store_true', default=False,
                        help='prioritize prefill requests over decode requests in scheduling')
    parser.add_argument('--block-size', type=int, default=16,
                        help='KV cache block size in tokens (number of tokens per block)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='path to .jsonl dataset file with request traces. '
                        'If None, requests must be added manually in serving/__main__.py')
    parser.add_argument('--output', type=str, default=None,
                        help='path for per-request CSV output with latency metrics (TTFT, TPOT, ITL). '
                        'If None, results are printed to stdout only. Supports {run_id} placeholder')
    parser.add_argument('--run-id', type=str, default=None,
                        help='unique id for this simulation run. Intermediate ASTRA-Sim inputs are written under '
                        'astra-sim/inputs/runs/<run-id>. If omitted, a process-unique id is generated')
    parser.add_argument('--inputs-root', type=str, default=None,
                        help='override the root directory for generated ASTRA-Sim inputs. Defaults to '
                        'astra-sim/inputs/runs/<run-id>')
    parser.add_argument('--cleanup-inputs', action=argparse.BooleanOptionalAction, default=True,
                        help='remove generated ASTRA-Sim inputs under astra-sim/inputs/runs/<run-id> '
                        'after a successful simulation (default: enabled). Use --no-cleanup-inputs '
                        'to preserve generated trace files, Chakra workloads, and input configs for debugging')
    parser.add_argument('--skip-prefill', action='store_true', default=False,
                        help='skip the prefill phase, running decode only')
    parser.add_argument('--num-reqs', type=int, default=0,
                        help='number of entries (requests or sessions) to load from the dataset. '
                        'For agentic datasets, each entry is a session with multiple sub-requests. '
                        '0 = load all entries')
    parser.add_argument('--log-interval', type=float, default=1.0,
                        help='interval in seconds between throughput/memory usage log messages')
    parser.add_argument('--log-level', type=str, choices=['WARNING', 'INFO', 'DEBUG'], default='WARNING',
                        help='logging verbosity: WARNING (minimal), INFO (per-iteration details), DEBUG (per-layer memory)')
    parser.add_argument('--kv-cache-dtype', type=str, choices=['auto', 'fp8'], default='auto',
                        help='KV cache data type: auto (use default profile.csv) or fp8 (use profile_fp8.csv, halves KV cache memory)')
    parser.add_argument('--network-backend', type=str, choices=['analytical', 'ns3'], default='analytical',
                        help='network simulation backend: analytical (fast, default) or ns3 (detailed, WIP)')
    # Unified serving policy (agentservesim harness mirror). Setting any
    # of the three knobs activates the adapter; unset knobs keep stock
    # behavior. See serving/core/unified_policy_adapter.py.
    parser.add_argument('--paper', default=None,
                        help='One published system by name (stock, continuum, '
                             'saga, autellix, infercept): every plane it '
                             'decides, none it does not. --planes program only. '
                             'Equivalent to the matching --retention/'
                             '--scheduling/--routing values, plus the axes '
                             'those flags cannot name together.')
    parser.add_argument('--policy', default=None,
                        help='One policy object spanning any subset of the '
                             'three axes, as module:Class (e.g. '
                             'harness.my_policy:MyPolicy). It is used for every '
                             'axis whose interface it implements; the rest keep '
                             'the engine default. --planes program only. '
                             'Mutually exclusive with --retention/--scheduling/'
                             '--routing, which name one published value each.')
    parser.add_argument('--planes', choices=['request', 'program'],
                        default='request',
                        help="Which decision planes to run. 'request' is the "
                             "stock request-scoped scheduler and memory model. "
                             "'program' is the program-aware orchestrator, KV "
                             "manager and batch scheduler. Default request: the "
                             "stock path stays the reference implementation.")
    parser.add_argument('--kv-pool-tokens', type=int, default=None,
                        help='KV pool size in TOKENS. Required with '
                             '--planes program: the program-aware KV manager '
                             'takes the pool as an input rather than deriving '
                             'it from a model-weight estimate, because tokens '
                             'are the quantity two different hosts can be held '
                             'equal on (measure it from vLLM kv_cache_tokens).')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='Reject requests whose prompt plus output exceeds this context limit')
    parser.add_argument('--saga-stealing', action=argparse.BooleanOptionalAction,
                        default=True, help='Allow SAGA placement to steal waiting work')
    parser.add_argument('--retention', default=None,
                        type=_policy_value('kv'),
                        help='retention knob: evict-always (run with --no-enable-prefix-caching), '
                             'cache-lru (stock), ttl, min-waste, gate (the evolved champion, policies/gate.py), '
                             'evolved (EvolvedRetention from --harness-root), '
                             'oracle-ttl (clairvoyant per-turn tau from the trace; headroom probe)')
    parser.add_argument('--retention-tau', type=float, default=None,
                        help='TTL retention: protection deadline in seconds after '
                             'gap start. Unset lets each policy use its own value '
                             '(Continuum pins 2 s, its released FIXED_THRESHOLD). '
                             'Required for ttl and saga-ttl, which have no intrinsic '
                             'window. Was 60.0, a number with no source -- no paper '
                             'or measurement specifies it -- which doubled as the '
                             '"unset" marker, so --retention-tau 60 was silently '
                             'ignored by Continuum and any TTL policy run without '
                             'the flag got 60 s (that is how SAGA ran at 60 against '
                             'a real leg at 2).')
    parser.add_argument('--retention-gap-default', type=float, default=1.0,
                        help='min-waste retention: fallback gap prediction in seconds')
    parser.add_argument('--saga-prefetch', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='SAGA: recompute a paused context just before its tool '
                             'result is due so the successor hits a warm prefix. Needs '
                             '--retention saga-tool-ttl for the learned gap estimate.')
    parser.add_argument('--saga-prefetch-margin', type=float, default=0.5,
                        help='SAGA prefetch: seconds before the predicted tool return to '
                             'start the recompute')
    parser.add_argument('--saga-fairness', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="SAGA: order the queue by the paper's adaptive fair share. "
                             'Requires tenant and deadline_ns on every session in the '
                             'dataset; neither can be guessed, so a trace without them '
                             'is refused.')
    parser.add_argument('--saga-fairness-slack', type=float, default=1.0,
                        help='SAGA fairness: minimum slack in seconds for a program past '
                             'its deadline. The paper leaves this unspecified, so it is a '
                             'recorded port parameter.')
    parser.add_argument('--saga-eviction-order', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="SAGA: reclaim by the paper's workflow-aware LRU (recency, "
                             'estimated reuse and size) instead of the prefix cache LRU. '
                             'Reuse is estimated from observed turn history only, and the '
                             'LRU stands until a successor turn has been seen.')
    parser.add_argument('--autellix-service-boundaries', type=str, default=None,
                        help='autellix-mlfq: comma-separated attained-service boundaries in '
                             'seconds that separate the MLFQ queues')
    parser.add_argument('--autellix-quanta', type=str, default=None,
                        help='autellix-mlfq: comma-separated per-queue quanta in seconds, one '
                             'more entry than --autellix-service-boundaries')
    parser.add_argument('--autellix-starvation-ratio', type=float, default=None,
                        help='autellix-mlfq: promote a call to queue 0 once its wait reaches '
                             'this multiple of its attained service')
    parser.add_argument('--autellix-swap', action=argparse.BooleanOptionalAction, default=False,
                        help="autellix-mlfq: preempt by copying the victim's KV to host memory "
                             'instead of recomputing it, so its next turn restores. Needs '
                             '--autellix-swap-bw and --autellix-swap-profile.')
    parser.add_argument('--autellix-swap-bw', type=float, default=None,
                        help='autellix-mlfq: measured per-rank host link GB/s for --autellix-swap')
    parser.add_argument('--autellix-swap-profile', type=str, default=None,
                        help='autellix-mlfq: measured forward-time profile JSON the transfer '
                             'budget is sized from')
    parser.add_argument('--autellix-overprovision', type=int, default=0,
                        help='autellix-mlfq: keep this many calls past the first non-fitting '
                             'one queued so they start the moment a selected call finishes')
    parser.add_argument('--min-waste-swap', action=argparse.BooleanOptionalAction, default=False,
                        help='experimental causal host copies for min-waste; serialized transfers '
                             'with radix ownership, not native InferCept swap/compute overlap')
    parser.add_argument('--min-waste-fcfs-restore', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='min-waste: admit whole-context restores in FCFS order; '
                             'an oversized head transfers alone with its full serialized '
                             'cost (not native InferCept chunk restoration)')
    parser.add_argument('--min-waste-swap-bw', type=float, default=None,
                        help='required with --min-waste-swap: measured per-rank host link GB/s '
                             '(slowest participating rank/direction); sizes and times both directions')
    parser.add_argument('--min-waste-profile', type=str, default=None,
                        help='min-waste retention: path to the measured InferCept profile JSON')
    parser.add_argument('--scheduling', default=None,
                        type=_policy_value('scheduling'),
                        help='scheduling knob: fcfs (stock), program-fcfs, plas, continuum, gate, '
                             'evolved (EvolvedScheduling from --harness-root), '
                             'oracle-srpt (clairvoyant remaining-work priority; headroom probe)')
    parser.add_argument('--routing', default=None,
                        type=_policy_value('routing'),
                        help='routing knob (overrides --request-routing-policy when set)')
    parser.add_argument('--long-prompt-tokens', type=int, default=2048,
                        help='autellix-route: prompts longer than this go back to the '
                             "program's established home; shorter ones go to the least "
                             'loaded instance')
    parser.add_argument('--routing-capacity-limit', type=int, default=None,
                        help='session-affinity routing: in-flight turns per instance before fallback')
    parser.add_argument('--decision-log-dir', type=str, default=None,
                        help='directory for the per-knob JSONL decision logs (parity input)')
    parser.add_argument('--harness-root', type=str, default=None,
                        help='directory holding your custom policy '
                             '(custom_retention.py / custom_scheduling.py), or '
                             'the AgentServingSim checkout. Default: this repo')
    parser.add_argument('--policy-root', dest='harness_root', type=str,
                        default=None,
                        help='alias for --harness-root; the name the policy '
                             'loader is actually about')

    args = parser.parse_args()
    if args.min_waste_swap and args.planes == 'program':
        parser.error('--min-waste-swap currently requires the request plane')
    
    args.run_id = resolve_run_id(args.run_id)
    run_paths = build_run_paths(astra_sim, args.run_id, args.inputs_root)
    args.inputs_root = run_paths.inputs_root
    args.output = _resolve_output_file(args.output, args.run_id)

    configure_logger(level=args.log_level)
    logger = get_logger("Main")
    print_banner()
    print_input_config(args=args)
    print_markup("[sim.heading]▶ Starting simulation...[/]\n")
    flush.stdout.flush()
    
    _dtype_to_bits = {'float16': 16, 'bfloat16': 16, 'float32': 32, 'fp8': 8, 'int8': 8}
    request_routing_policy=args.request_routing_policy
    expert_routing_policy=args.expert_routing_policy
    enable_prefix_sharing=args.enable_prefix_sharing
    prefix_storage=args.prefix_storage
    dataset=args.dataset
    output_file=args.output
    is_init = not args.skip_prefill
    num_req=args.num_reqs
    log_interval=args.log_interval
    network_backend = args.network_backend
    raw_cluster_config = _load_cluster_config_for_overrides(args.cluster_config)
    raw_instances = list(_iter_raw_instances(raw_cluster_config))
    # Exposed-host-overhead calibration (mixed prefill+decode steps). This is a
    # hardware/serving characteristic, so it lives in the cluster config as the
    # per-instance field `host_overhead_mix_per_seq_ns`. The trace generator
    # reads it via the MIX_OVERHEAD_PER_SEQ_NS env var; an explicit env value
    # (e.g. for calibration sweeps) takes precedence over the config field.
    if "MIX_OVERHEAD_PER_SEQ_NS" not in os.environ:
        _mix_ns = next((inst["host_overhead_mix_per_seq_ns"] for inst in raw_instances
                        if inst.get("host_overhead_mix_per_seq_ns")), None)
        if _mix_ns:
            os.environ["MIX_OVERHEAD_PER_SEQ_NS"] = str(int(_mix_ns))
    build_enable_local_offloading = args.enable_local_offloading or any(
        inst.get("enable_local_offloading", False) for inst in raw_instances)
    build_enable_attn_offloading = args.enable_attn_offloading or any(
        inst.get("enable_attn_offloading", False) for inst in raw_instances)
    # ---------------------------------- Extract cluster config -----------------------------------
    cluster = build_cluster_config(
        astra_sim, args.cluster_config, build_enable_local_offloading, build_enable_attn_offloading,
        inputs_root=run_paths.inputs_root)
    num_nodes = cluster["num_nodes"]
    num_instances = cluster["num_instances"]
    instances = cluster["instances"]
    inst2node_mapping = cluster["inst2node_mapping"]
    inst2npu_mapping = cluster["inst2npu_mapping"]
    npu2inst_mapping = cluster["npu2inst_mapping"]
    prefill_instance = cluster["prefill_instance"]
    decode_instance = cluster["decode_instance"]
    start_npu_ids = cluster["start_npu_ids"]
    end_npu_ids = cluster["end_npu_ids"]
    placement = cluster["placement"]
    block_mode_on = cluster["block_mode_on"]
    total_npu = cluster["total_npu"]
    cpu_mem_size = cluster["cpu_mem_size"]
    cpu_mem_bw = cluster["cpu_mem_bw"]
    power_modeling = cluster["power_modeling"]
    power_configs = cluster["power_configs"]
    pim_models = cluster["pim_models"]
    instance_runtime_configs = _build_instance_runtime_configs(instances, args, _dtype_to_bits)
    any_prefix_caching = any(cfg["enable_prefix_caching"] for cfg in instance_runtime_configs)
    # ----------------------------------------- Set config -----------------------------------------
    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config.py and each json file
    if network_backend == 'analytical':
        network=run_paths.network_config
        binary=os.path.join(astra_sim, "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra")
    elif network_backend == 'ns3':
        network=_prepare_ns3_config(astra_sim, run_paths)
        binary=os.path.join(astra_sim, "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    else:
        raise NotImplementedError("Only analytical and ns3 network backend are supported")
    memory=run_paths.memory_config
    system=run_paths.system_config
    # ------------------------------------- Prepare simulation -------------------------------------
    # Need to extract each instance's memory accessability 
    node2inst_mapping = defaultdict(list)
    for inst_id, node_id in inst2node_mapping.items():
        node2inst_mapping[node_id].append(inst_id)
    node2inst_mapping = dict(node2inst_mapping)

    prefix_pool_inst_mapping = {}
    for i in range(num_instances):
        prefix_pool_inst_mapping[i] = None

    pool_device = None

    if prefix_storage == "CPU":
        pool_device = Device.CPU
    elif prefix_storage == "CXL":
        pool_device = Device.CXL

    if any_prefix_caching and enable_prefix_sharing and prefix_storage != 'None':
        num_prefix_pool = num_nodes
        # make prefix pool objects based on num_prefix_pool
        prefix_pools = []

        def _pool_kv_bytes_per_token(inst_ids):
            """KV bytes per token for a shared pool."""
            kv_shapes = {
                (
                    instances[i]["model_name"],
                    instance_runtime_configs[i]["fp"],
                    instance_runtime_configs[i]["kv_cache_dtype"],
                )
                for i in inst_ids
            }
            if len(kv_shapes) > 1:
                raise RuntimeError(
                    "Shared prefix pool requires instances to share model, "
                    f"dtype, and kv_cache_dtype; got {kv_shapes}"
                )
            model = instances[inst_ids[0]]['model_name']
            cfg = instance_runtime_configs[inst_ids[0]]
            return full_cluster_kv_bytes_per_token(model, cfg["fp"], cfg["kv_cache_dtype"])

        if prefix_storage == 'CPU':
            for i in range(num_prefix_pool):
                if cpu_mem_size[i] > 0:
                    new_prefix_pool = RadixCache(
                                                node_id=0,
                                                device=prefix_storage,
                                                page_size=256,
                                                capacity = cpu_mem_size[i] * GB_TO_BYTE,
                                                kv_size=_pool_kv_bytes_per_token(node2inst_mapping[i]),
                                                enable_kv_cache_events=True)
                    prefix_pools.append(new_prefix_pool)
                else:
                    raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
            # This means one node shares one prefix pool
            prefix_pool_inst_mapping = inst2node_mapping

        elif prefix_storage == 'CXL':
            if cluster["cxl_mem_size"] > 0:
                new_prefix_pool = RadixCache(
                                            node_id=None,
                                            device=prefix_storage,
                                            page_size=1,
                                            capacity = cluster["cxl_mem_size"] * GB_TO_BYTE,
                                            kv_size=_pool_kv_bytes_per_token(list(range(num_instances))),
                                            enable_kv_cache_events=True)
                prefix_pools.append(new_prefix_pool)
                # This means every instance shares the same universal prefix pool (maybe fixed later)
                prefix_pool_inst_mapping = [0 for _ in range(num_instances)]
            else:
                raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
        else:
            raise NotImplementedError(f"Prefix storage type {prefix_storage} is not supported or memory size is invalid")

    schedulers = []
    # One orchestrator for the whole cluster when --planes program: a program is
    # one entity wherever its turns land, while a KV manager is per-instance
    # because pressure is per-instance.
    _program_orchestrator = [None]
    for instance_id, instance in enumerate(instances):
        prefix_pool_index = prefix_pool_inst_mapping[instance_id]
        prefix_pool = None
        if prefix_pool_index != None:
            prefix_pool = prefix_pools[prefix_pool_index]
        cxl_mem = 0
        if cluster["cxl_mem_size"] > 0:
            cxl_mem = cluster["cxl_mem_size"]        
        
        # Make scheduler for each instance

        inst_cfg = instance_runtime_configs[instance_id]

        if args.planes == 'program':
            # Program-aware planes. One orchestrator for the CLUSTER (a program
            # is one entity wherever its turns land) and one KV manager per
            # instance (pressure is per-instance).
            from .core.program_kv import ProgramKVManager
            from .core.program_orchestrator import ProgramOrchestrator
            from .core.program_scheduler import ProgramBatchScheduler
            if args.kv_pool_tokens is None:
                raise ValueError(
                    "--planes program requires --kv-pool-tokens: the KV pool is "
                    "an input to the program-aware manager, not something it "
                    "derives from a weight estimate. Measure it from the real "
                    "engine (vLLM reports kv_cache_tokens).")
            if _program_orchestrator[0] is None:
                _program_orchestrator[0] = ProgramOrchestrator()
            _orch = _program_orchestrator[0]
            schedulers.append(ProgramBatchScheduler(
                instance_id, _orch,
                ProgramKVManager(instance_id, args.kv_pool_tokens, _orch,
                                 block_size=inst_cfg["block_size"]),
                model=instance["model_name"],
                max_num_batched_tokens=inst_cfg["max_num_batched_tokens"],
                max_num_seqs=inst_cfg["max_num_seqs"],
                long_prefill_token_threshold=inst_cfg["long_prefill_token_threshold"],
                start_npu=inst2npu_mapping[instance_id],
                num_npus=instance["num_npus"],
                pd_type=instance["pd_type"],
                pp_size=instance["pp_size"],
                enable_prefix_caching=inst_cfg["enable_prefix_caching"],
            ))
            schedulers[-1].max_model_len = args.max_model_len
            continue

        schedulers.append(Scheduler(
            instance["model_name"], instance["node_id"], instance_id,
            inst_cfg["max_num_seqs"], inst_cfg["max_num_batched_tokens"],
            instance["num_npus"], instance["tp_size"], instance["pp_size"],
            instance["npu_mem"]["mem_size"], cpu_mem_size[instance["node_id"]],
            inst2npu_mapping[instance_id], instance["pd_type"],
            inst_cfg["fp"], inst_cfg["block_size"], num_req,
            inst_cfg["prioritize_prefill"], inst_cfg["enable_prefix_caching"],
            enable_prefix_sharing, prefix_pool, pool_device, inst_cfg["enable_chunked_prefill"],
            inst_cfg["long_prefill_token_threshold"],
            cxl_mem,
            ep_size=instance.get("ep_total", 1),
            kv_cache_dtype=inst_cfg["kv_cache_dtype"],
        ))
        schedulers[-1].max_model_len = args.max_model_len
        pool_tokens = instance.get('kv_pool_tokens', args.kv_pool_tokens)
        if pool_tokens is not None:
            schedulers[-1].memory.set_kv_capacity_tokens(pool_tokens)

    # Controller for astra-sim process communication
    controller = Controller(total_npu)

    # Unified serving policy adapter (retention, scheduling, routing);
    # active only when at least one knob is set.
    for sched in schedulers:
        sched.cleanup_et = bool(args.cleanup_inputs)
    policy_adapter = None
    program_policy = None
    # The InferCept waste model is a fit of T_fwd on ONE hardware, model and
    # tensor-parallel width; another platform's numbers price every swap
    # decision wrong. The cluster config already states all three, so derive it
    # rather than requiring a flag that must agree with the config and will one
    # day disagree with it. Explicit --min-waste-profile still wins.
    #
    # Before this, `--retention min-waste` without the flag died inside a JSON
    # loader with "expected str, bytes or os.PathLike object, not NoneType",
    # and the arena runner never emitted the flag at all -- so InferCept could
    # not run there.
    if args.min_waste_profile is None and (
            args.retention == "min-waste" or args.paper == "infercept"
            or args.policy):
        try:
            from policies.utils.waste_model import resolve_profile
            _i0 = instances[0]
            args.min_waste_profile = resolve_profile(
                _i0["hardware"], _i0["model_name"], _i0["tp_size"])
        except Exception as _e:
            if args.retention == "min-waste" or args.paper == "infercept":
                raise            # it is required; fail with the real reason
            pass                 # --policy may not use the waste model at all
    if args.paper and args.planes != 'program':
        raise ValueError(
            "--paper needs --planes program: the request-plane adapter takes "
            "one value per axis and cannot express a paper that decides "
            "several planes at once.")
    if args.paper and (args.policy or args.retention or args.scheduling
                       or args.routing):
        raise ValueError(
            "--paper names a whole published system; passing it alongside "
            "--policy or an axis flag leaves it ambiguous which decides an "
            "axis they both name.")
    if args.policy and args.planes != 'program':
        raise ValueError(
            "--policy needs --planes program: the request-plane adapter takes "
            "one published value per axis and has no way to express one object "
            "that decides several.")
    if args.policy and (args.retention or args.scheduling or args.routing):
        raise ValueError(
            "--policy replaces --retention/--scheduling/--routing rather than "
            "adding to them: passing both leaves it ambiguous which decides "
            "an axis they both implement.")
    if args.planes == 'program' and (args.retention or args.scheduling
                                     or args.routing or args.policy
                                     or args.paper):
        # The SAME policy classes the real GPU harness runs, reading a record
        # projected from the orchestrator. No second program table: the whole
        # reason these planes exist is that the old engine needed a shadow one.
        from .core.program_policy_adapter import ProgramPolicyAdapter
        from .core.unified_policy_adapter import import_harness
        if args.retention == 'evict-always' and any_prefix_caching:
            raise ValueError("--retention evict-always requires "
                             "--no-enable-prefix-caching (APC off is the mechanism)")
        mods = import_harness(args.harness_root)
        oracle_tbl = None
        if args.retention == 'oracle-ttl' or args.scheduling == 'oracle-srpt':
            from policies.oracle import OracleTable
            _p = args.dataset
            if _p and not os.path.isabs(_p):
                _p = f"../{_p}"
            oracle_tbl = OracleTable(_p)
        program_policy = ProgramPolicyAdapter(
            _program_orchestrator[0], mods,
            retention=args.retention, scheduling=args.scheduling,
            routing=args.routing,
            num_instances=sum(1 for s in schedulers if s.pd_type != "decode"),
            tau_s=args.retention_tau,
            default_gap_s=args.retention_gap_default,
            min_waste_profile=args.min_waste_profile,
            capacity_limit=args.routing_capacity_limit,
            oracle_table=oracle_tbl,
            unified=args.policy,
            paper=args.paper,
            log_dir=args.decision_log_dir)
        for sched in schedulers:
            sched.policy = program_policy
            # None when the policy does not OVERRIDE the hook, not merely when
            # no policy is configured. The plane gates its expensive per-step
            # work on these being set -- `sync_footprints` alone is three full
            # tree walks per resident program, per step -- and a bound method
            # that immediately returns `None` is still truthy, so wiring them
            # unconditionally made every baseline pay a policy's price to call
            # functions with no opinion. `--paper stock` overrides nothing at
            # all, which is the point of a baseline: it must cost what no
            # policy costs.
            sched.priority_fn = (program_policy.priority_fn
                                 if program_policy.wants_priority else None)
            sched.admit_fn = (program_policy.admit_fn
                              if program_policy.wants_admit else None)
            sched.victim_fn = (program_policy.victim_fn
                               if program_policy.wants_victim else None)
    if args.planes != 'program' and (args.retention or args.scheduling
                                     or args.routing):
        from .core.unified_policy_adapter import UnifiedPolicyAdapter
        if args.retention == 'evict-always' and any_prefix_caching:
            raise ValueError("--retention evict-always requires "
                             "--no-enable-prefix-caching (APC off is the mechanism)")
        prefill_count = sum(1 for s in schedulers if s.pd_type != "decode")
        policy_adapter = UnifiedPolicyAdapter(
            args.retention, args.scheduling, args.routing,
            num_instances=prefill_count,
            saga_stealing=args.saga_stealing,
            block_size=schedulers[0].memory.block_size,   # old planes only
            tau_s=args.retention_tau,
            default_gap_s=args.retention_gap_default,
            min_waste_profile=args.min_waste_profile,
            min_waste_swap=args.min_waste_swap,
            autellix_queues=_autellix_queues(args),
            autellix_overprovision=args.autellix_overprovision,
            saga_eviction_order=args.saga_eviction_order,
            autellix_swap=args.autellix_swap,
            long_prompt_tokens=args.long_prompt_tokens,
            min_waste_fcfs_restore=args.min_waste_fcfs_restore,
            saga_fairness=args.saga_fairness,
            saga_fairness_slack=args.saga_fairness_slack,
            saga_prefetch=args.saga_prefetch,
            saga_prefetch_margin_s=args.saga_prefetch_margin,
            capacity_limit=args.routing_capacity_limit,
            log_dir=args.decision_log_dir,
            harness_root=args.harness_root,
            oracle_table_path=args.dataset,
        )
        policy_adapter.attach_schedulers(schedulers)
        for sched in schedulers:
            sched.memory.kv_protection = policy_adapter
            sched.policy_hooks = policy_adapter
            if policy_adapter.priority_scheduling:
                sched.scheduling_policy = "priority"
            if args.retention == 'min-waste' and args.min_waste_swap:
                policy_adapter.configure_host_swap(sched.memory, args.min_waste_swap_bw)
            if args.saga_eviction_order and sched.memory.enable_prefix_caching:
                policy_adapter.saga_evictor(sched.memory)
            if args.autellix_swap:
                from policies.utils.waste_model import WasteProfile
                from serving.core.host_swap import PreemptSwapPolicy
                if not args.autellix_swap_profile:
                    parser.error('--autellix-swap needs --autellix-swap-profile')
                policy_adapter.configure_host_swap(
                    sched.memory, args.autellix_swap_bw,
                    profile=WasteProfile.from_json(args.autellix_swap_profile),
                    policy=PreemptSwapPolicy(), flag='--autellix-swap')

    # Global Request Router.
    #
    # Two planes answer the loop's questions below, and each question is asked
    # of whichever plane owns the fact. Rather than branch at every call site,
    # the branch is taken once here and the loop calls names that mean the same
    # thing on both paths. The program path deliberately does NOT wrap the old
    # router: a wrapper would keep one interface over two different models of
    # what a program is, which is the arrangement this design exists to end.
    router = None
    orchestrator = None
    program_router = None
    if args.planes == 'program':
        from .core.program_router import ProgramRouter
        from .core.program_router import dispatch as _plane_dispatch
        from .core.program_router import transfer_prefill as _plane_transfer
        from .core.program_workload import load_programs as _plane_load
        orchestrator = _program_orchestrator[0]
        program_router = ProgramRouter(
            orchestrator, num_instances, schedulers,
            policy=request_routing_policy,
            capacity_limit=args.routing_capacity_limit,
            route_fn=(program_policy.route_fn
                      if program_policy is not None
                      and program_policy.axes["routing"] else None),
            on_placed=(program_policy.on_placed
                       if program_policy is not None
                       and program_policy.axes["routing"] else None))

        def plane_load(path):
            _plane_load(path, orchestrator, any_prefix_caching)

        def plane_first_arrival():
            return orchestrator.first_arrival_ts()

        def plane_dispatch(now):
            _plane_dispatch(program_router, schedulers, now)

        def plane_transfer_prefill(reqs):
            _plane_transfer(program_router, reqs, schedulers)

        def plane_turn_complete(instance_id, reqs, now):
            # Nothing: `ProgramBatchScheduler.add_done` already told the
            # orchestrator, because on this plane a finished request IS a
            # finished turn and no third party knows better.
            pass

        def plane_has_pending():
            return orchestrator.has_pending()

        def plane_next_arrival():
            return orchestrator.next_arrival_ts()

        def plane_has_workflow_metrics():
            return orchestrator.has_workflow_metrics()

        def plane_workflow_metrics_summary():
            return orchestrator.workflow_metrics_summary()

        def plane_save_workflow_metrics(path):
            return orchestrator.save_workflow_metrics(path)
    else:
        router = Router(num_instances, schedulers, num_req,
                        request_routing_policy, policy_adapter=policy_adapter)

        def plane_load(path):
            router.load_requests(path, enable_prefix_caching=any_prefix_caching,
                                 is_init=is_init)

        def plane_first_arrival():
            return router.get_first_arrival_time()

        def plane_dispatch(now):
            router.route_arrived_requests(now)

        def plane_transfer_prefill(reqs):
            router.transfer_prefill_request(reqs)

        def plane_turn_complete(instance_id, reqs, now):
            for req in reqs:
                # `memory` feeds the old retention adapter's release hook.
                router.notify_request_completed(
                    req.id, now, req_obj=req,
                    memory=schedulers[instance_id].memory)

        def plane_has_pending():
            return router.has_pending_requests() or router.has_deferred_sessions()

        def plane_next_arrival():
            return router.get_next_pending_arrival()

        def plane_has_workflow_metrics():
            return router.has_workflow_metrics()

        def plane_workflow_metrics_summary():
            return router.workflow_metrics_summary()

        def plane_save_workflow_metrics(path):
            return router.save_workflow_metrics(path)
    # Power Modeling if enabled
    if power_modeling:
        power_model = PowerModel(power_configs)
    else:
        power_model = None
    # Load requests into router (routed in real-time during simulation)
    if dataset != None:
        plane_load(dataset)
    else:
        # Manually adding request (legacy: route all upfront)
        for i in range(16):
            for sched in schedulers:
                sched.add_request([i, sched.model, 64, 128, 0, i % num_instances])

    # Simulator start
    current = 0 # current tick of the system
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request
    is_prefill_done = False # flag to check if prefill is done
    done_instance = [] # list of done instances
    done_inst_npus = [[] for _ in range(num_instances)]
    start_time = time()
    last_end_time = [0 for _ in range(num_instances)]
    last_calc_time = [0 for _ in range(num_instances)]
    waiting_request = [False for _ in range(num_instances)]

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000_000_000 # 1 GHz (1e9 Hz)
    INTERVAL = log_interval*FREQ
    RATIO = FREQ//INTERVAL
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    req_cnt = 0

    # Set Event Handler that loop with INTERVAL time until first request arrive (for all instances)
    first_arival_time = plane_first_arrival()
    if INTERVAL > first_arival_time:
        event_time = first_arival_time
    else:
        event_time = INTERVAL
    generate_event(int(event_time), inputs_root=run_paths.inputs_root)
    # Make Chakra Grapth
    generate_graph(None, None, total_npu, event=True, inputs_root=run_paths.inputs_root,
                   cleanup_trace=args.cleanup_inputs)
    # set first workload file
    workload = get_workload(None, None, event=True, inputs_root=run_paths.inputs_root)
    # run subprocess
    astra_args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--memory-configuration="+memory]
    if start_npu_ids != "":
        astra_args.append("--start-npu-ids="+start_npu_ids)
    if end_npu_ids != "":
        astra_args.append("--end-npu-ids="+end_npu_ids)
    if network_backend == 'ns3':
        astra_args.append("--logical-topology-configuration="+astra_sim+"/inputs/logical_topology/logical_8nodes_1D.json")
    # The analytical backend writes its parsed memory configs to a CWD-relative
    # scratch dir (network_frontend/analytical/congestion_unaware/main.cc:
    # save_json_to_tmp -> tmp__mem/{local,remote,cxl}_mem.json). With the
    # shared astra-sim/ checkout as CWD, two runs starting concurrently on
    # any nodes overwrite/remove each other's files: "Unable to open file:
    # tmp__mem/remote_mem.json" or a SIGABRT from the JSON parser at start.
    # Every argument is absolute, so run it inside this run's inputs root.
    p = subprocess.Popen(astra_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True, cwd=run_paths.inputs_root)

    # DP group synchronization: defer trace generation until all members have scheduled
    # dp_groups maps dp_group_name -> list of instance_ids
    dp_groups = {}
    for inst in instances:
        dg = inst.get("dp_group")
        if dg is not None:
            dp_groups.setdefault(dg, []).append(inst["instance_id"])
    # Reverse lookup: instance_id -> dp_group_name
    inst_dp_group = {}
    for dg, members in dp_groups.items():
        for inst_id in members:
            inst_dp_group[inst_id] = dg
    # Pending batches per DP group (waiting for all members to schedule)
    dp_pending = {dg: {} for dg in dp_groups}  # dp_group -> {instance_id: (new_req, sys)}
    # Pre-generated workloads ready to submit on next "Waiting"
    dp_ready_workloads = {}  # instance_id -> workload_path

    # Agentic idle-gap handling. When every instance is drained but a deferred
    # sub-request will only arrive in the future (a tool call is in flight),
    # ASTRA-Sim has no event to advance: poking it with "pass" deadlocks (it
    # blocks reading our stdin while we block reading its stdout). Instead we
    # fast-forward wall-clock time in Python (skip_read: do not read the backend
    # that iteration) and remember how much idle time we skipped (idle_offset),
    # because ASTRA-Sim's cycle counter only advances on actual compute.
    skip_read = False
    idle_offset = 0
    idle_sweep = IdleScheduleSweep()
    _fp_last = -10**18
    _fp_rows = []

    # ----------------------------------- Start simulation loop ------------------------------------
    # Starting simulation, one while loop processes one iteration
    while True:

        if skip_read:
            # We fast-forwarded through an idle gap last iteration; ASTRA-Sim has
            # no pending event to report, so do NOT read it. Re-use the carried
            # sys/id/current and go straight to (re)scheduling the arrival.
            skip_read = False
        else:
            out = controller.read_wait(p)
            out_dict = controller.parse_output(out[-2])

            if out_dict != None:
                sys = out_dict['sys']
                id = out_dict['id']
                # ASTRA-Sim's cycle counts compute only; add the idle time we
                # skipped so 'current' tracks true wall-clock.
                current = out_dict['cycle'] + idle_offset

        # Route newly arrived requests to instances based on current load
        # A previous failed scheduling attempt is obsolete after work completes
        # or a new arrival becomes dispatchable. Completion is handled below;
        # reset while its batch is still present so non-start NPU completions
        # invalidate the sweep as well.
        dispatch_due = plane_next_arrival()
        if (any(sc.inflight for sc in schedulers)
                or (dispatch_due is not None and dispatch_due <= current)):
            idle_sweep.reset()
        if dataset is not None:
            plane_dispatch(current)

        instance_id = npu2inst_mapping[sys]  # get instance id from NPU id
        node_id = inst2node_mapping[instance_id] # get node id from instance id

        # add stanby energy consumption for power modeling
        if power_modeling and sys == inst2npu_mapping[instance_id] and waiting_request[instance_id]:
            power_model.add_npu_standby_energy_consumption(instances[instance_id]["hardware"], node_id, current,
                        last_end_time[instance_id], last_calc_time[instance_id], num_npus=instances[instance_id]["num_npus"])
            last_calc_time[instance_id] = current

        # mark latest end time of the first NPU in the instance
        # An instance can span multiple NPUs. Only update end-time when sys is the first NPU of the instance.
        # waiting_request[instance_id] = True means the instance has no batch to run (idle).
        if sys == inst2npu_mapping[instance_id] and not waiting_request[instance_id]:
            last_end_time[instance_id] = current
            waiting_request[instance_id] = True

        # check request is done
        prompt_t, gen_t, finished_reqs = schedulers[instance_id].add_done(id, sys, current)
        # A SAGA prefetch is a recompute the policy issued, not a program turn.
        # Its GPU cost has already been paid inside the batch above; counting
        # it as a completion would inflate the request count and hand the
        # router a chain step that does not exist.
        prefetched = [r for r in finished_reqs if r.prefetch_of is not None]
        if prefetched:
            finished_reqs = [r for r in finished_reqs if r.prefetch_of is None]
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        # count only finished requests
        req_cnt += len(finished_reqs) if instances[instance_id]["pd_type"] != "prefill" else 0

        # Release the next turn in each completed program's chain.
        if instances[instance_id]["pd_type"] != "prefill":
            plane_turn_complete(instance_id, finished_reqs, current)

        # Add prefill ended requests to decode instance
        if instances[instance_id]["pd_type"] == "prefill" and len(finished_reqs) > 0:
            plane_transfer_prefill(finished_reqs)

        # schedule requests
        new_req = schedulers[instance_id].schedule(current, sys, id)
        responded = False  # track whether we already sent a response to ASTRA-Sim

        # Check if a pre-generated workload is ready for this instance (from DP sync)
        if new_req is None and instance_id in dp_ready_workloads:
            controller.write_flush(p, dp_ready_workloads.pop(instance_id))
            responded = True
        # DP group: truly idle instance (no inflight batch) — create dummy batch so ALLTOALL syncs
        elif new_req is None and instance_id in inst_dp_group and sys == inst2npu_mapping[instance_id] and len(schedulers[instance_id].inflight) == 0:
            dg = inst_dp_group[instance_id]
            if dp_pending[dg]:
                # Emit a 1-token dummy; the uniform pad-to-max pass below
                # brings it (and any undersized real peers) up to the
                # group's max_total_len, matching vLLM's CUDA-graph DP padding.
                logger.debug(f"Instance {instance_id} is idle but DP group {dg} has pending batches. Creating dummy batch for synchronization.")
                dummy = Batch(schedulers[instance_id].get_batch_id(), instances[instance_id]["model_name"],
                              1, 1, [1], [], 0, 1, [], [], [1], current, 0)
                dummy.fired.append(sys)
                dp_pending[dg][instance_id] = (dummy, inst2node_mapping[instance_id])

                if len(dp_pending[dg]) == len(dp_groups[dg]):
                    # All DP members accounted for — pad every batch to the
                    # group's max (vLLM CUDA-graph DP padding) and generate.
                    config = get_config(instances[instance_id]["model_name"])
                    max_total_len = max(b.total_len for b, _ in dp_pending[dg].values())
                    for b, _ in dp_pending[dg].values():
                        _pad_batch_to_max(b, max_total_len)
                    # MoE AG/RS comm size is anchored to ``max_total_len``
                    # (not ``max × group_size``). The trace generator divides
                    # this by ep_total internally for the per-rank AG chunk
                    # and uses the same value for the RS pre-scatter buffer.
                    # Empirically this matches real NCCL AG/RS bandwidth on
                    # PCIe 5.0 at the same ``link_bw`` that already calibrates
                    # AllReduce — i.e. ASTRA-Sim's Ring half-duplex model
                    # ends up correct for AR but 2× over real AG/RS, and the
                    # "× group_size" we used previously stacked the two errors.
                    sum_total_len = max_total_len

                    # Shared workload folder for all DP members
                    first_inst_id = dp_groups[dg][0]
                    first_batch = dp_pending[dg][first_inst_id][0]
                    dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                    for inst_id in dp_groups[dg]:
                        batch, nid = dp_pending[dg][inst_id]
                        inst = instances[inst_id]
                        inst_cfg = instance_runtime_configs[inst_id]
                        generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                       inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                       nid, inst_id,
                                       inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                       placement[inst_id], block_mode_on[inst_id],
                                       expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                       inst_cfg["enable_attn_offloading"],
                                       power_model, pim_models[nid],
                                       inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                       dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                       tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                       dp_sum_total_len=sum_total_len,
                                       enable_block_copy=inst_cfg["enable_block_copy"],
                                       inputs_root=run_paths.inputs_root)
                        generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                       inst_id, inst2npu_mapping[inst_id],
                                       inst_cfg["enable_local_offloading"],
                                       workload_name=dp_workload_name,
                                       inputs_root=run_paths.inputs_root,
                                       cleanup_trace=args.cleanup_inputs)
                        if inst_id != instance_id:
                            dp_ready_workloads[inst_id] = get_workload(batch, inst["hardware"], inst_id,
                                                                    workload_name=dp_workload_name,
                                                                    inputs_root=run_paths.inputs_root)

                    dp_pending[dg].clear()
                    workload = get_workload(dummy, instances[instance_id]["hardware"], instance_id,
                                            workload_name=dp_workload_name,
                                            inputs_root=run_paths.inputs_root)
                    controller.write_flush(p, workload)
                    responded = True
                else:
                    controller.write_flush(p, "pass")
                    responded = True
        # runnable batch exists
        elif new_req is not None:
            if sys == inst2npu_mapping[instance_id]:  # first NPU of the instance
                waiting_request[instance_id] = False
                instance = instances[instance_id]
                dg = inst_dp_group.get(instance_id)

                if dg is not None:
                    # DP group: defer trace generation until all members scheduled
                    dp_pending[dg][instance_id] = (new_req, node_id)

                    if len(dp_pending[dg]) == len(dp_groups[dg]):
                        # All DP members have scheduled — pad every batch to
                        # the group's max (vLLM CUDA-graph DP padding) so
                        # smaller batches gain dummy decodes that all layers
                        # still compute over.
                        config = get_config(instance["model_name"])
                        max_total_len = max(b.total_len for b, _ in dp_pending[dg].values())
                        for b, _ in dp_pending[dg].values():
                            _pad_batch_to_max(b, max_total_len)
                        # See twin block above: anchor MoE comm to max_total_len
                        # (no group-size multiplier).
                        sum_total_len = max_total_len

                        # Shared workload folder for all DP members
                        first_inst_id = dp_groups[dg][0]
                        first_batch = dp_pending[dg][first_inst_id][0]
                        dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                        for inst_id in dp_groups[dg]:
                            batch, nid = dp_pending[dg][inst_id]
                            inst = instances[inst_id]
                            inst_cfg = instance_runtime_configs[inst_id]
                            generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                           inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                           nid, inst_id,
                                           inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                           placement[inst_id], block_mode_on[inst_id],
                                           expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                           inst_cfg["enable_attn_offloading"],
                                           power_model, pim_models[nid],
                                           inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                           dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                           tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                           dp_sum_total_len=sum_total_len,
                                           enable_block_copy=inst_cfg["enable_block_copy"],
                                           inputs_root=run_paths.inputs_root)
                            generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                           inst_id, inst2npu_mapping[inst_id],
                                           inst_cfg["enable_local_offloading"],
                                           workload_name=dp_workload_name,
                                           inputs_root=run_paths.inputs_root,
                                           cleanup_trace=args.cleanup_inputs)
                            if inst_id != instance_id:
                                dp_ready_workloads[inst_id] = get_workload(batch, inst["hardware"], inst_id,
                                                                        workload_name=dp_workload_name,
                                                                        inputs_root=run_paths.inputs_root)

                        dp_pending[dg].clear()
                        workload = get_workload(new_req, instance["hardware"], instance_id,
                                                workload_name=dp_workload_name,
                                                inputs_root=run_paths.inputs_root)
                        controller.write_flush(p, workload)
                    else:
                        # Waiting for other DP members — send pass
                        controller.write_flush(p, "pass")
                        responded = True
                else:
                    # Independent instance: generate trace immediately
                    inst_cfg = instance_runtime_configs[instance_id]
                    generate_trace(new_req, instance["hardware"], instance["tp_size"], instance["pp_size"],
                                   instance["local_ep"], instance["ep_total"],
                                   instance["pd_type"],
                                   node_id, instance_id,
                                   inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                   placement[instance_id], block_mode_on[instance_id],
                                   expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                   inst_cfg["enable_attn_offloading"], power_model, pim_models[node_id],
                                   inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                   dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                   tp_dim=instance["tp_dim"], ep_dim=instance["ep_dim"],
                                   enable_block_copy=inst_cfg["enable_block_copy"],
                                   inputs_root=run_paths.inputs_root)
                    generate_graph(new_req, instance["hardware"], instance["num_npus"], node_id,
                                   instance_id, inst2npu_mapping[instance_id],
                                   inst_cfg["enable_local_offloading"],
                                   inputs_root=run_paths.inputs_root,
                                   cleanup_trace=args.cleanup_inputs)
                    workload = get_workload(new_req, instance["hardware"], instance_id,
                                            inputs_root=run_paths.inputs_root)
                    controller.write_flush(p, workload)
            elif new_req is not None:
                # Non-first NPU: pick up existing batch workload
                workload = get_workload(new_req, instances[instance_id]["hardware"], instance_id,
                                        inputs_root=run_paths.inputs_root)
                controller.write_flush(p, workload)

        # check time to store throughput (only print on start NPU to avoid transient states)
        # Per-sim-second footprint series (referenced = contexts of running/
        # inflight requests; cached = prefix-cache tokens), comparable with
        # the engine's kv_cache_usage in timeseries.csv.
        if output_file and current > _fp_last + 1_000_000_000:
            _fp_last = current
            for _i, _sc in enumerate(schedulers):
                _r = _sc.mem_report()
                _fp_rows.append((_i, current / 1e9, _r["waiting"], _r["batched"],
                                 _r["referenced_tokens"], _r["cache_total"],
                                 _r["cache_evictable"], _r["cache_protected"],
                                 _r["kv_util"], _r["reserved_tokens"]))
        if current > last_log + INTERVAL and sys == inst2npu_mapping[instance_id]:
            # store the prompt
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            log_time_str = f"[{last_log / FREQ:.1f}s]"
            log_time_len = len(log_time_str)
            log_indent = ' ' * log_time_len + '  '
            tree_indent = '├─'
            # Heartbeat timestamp stays in the terminal's default
            # colour — bright enough to scan, not so dim that it
            # disappears. (The per-log-record [HH:MM:SS.mmm] stays
            # dim via sim.time because it appears every other line.)
            print_markup(
                f"{log_time_str} "
                f"[blue]Avg prompt throughput: {prompt_th * RATIO:.1f} tokens/s,[/] "
                f"[blue]Avg generation throughput: {gen_th * RATIO:.1f} tokens/s[/]"
            )
            prompt_th = 0
            gen_th = 0

            ######### Per Instance Metrics #########

            for inst_id in range(num_instances):
                running_reqs = sum(len(batch.requests) for batch in schedulers[inst_id].inflight)
                waiting_reqs = len([req for req in schedulers[inst_id].request if req.arrival <= current])

                _r = schedulers[inst_id].mem_report()
                npu_util = (_r["npu_used"] / _r["npu_mem"] * 100.0) if _r["npu_mem"] else 0.0

                line = (
                    f"{log_indent+tree_indent}Running Instance\\[{inst_id}]: "
                    f"{running_reqs} reqs, Waiting: {waiting_reqs} reqs, "
                    f"Total # {schedulers[inst_id].num_npus} NPUs, "
                )
                # The two planes account in different units and the same line
                # was printing both as bytes: the program plane's `npu_used` is
                # KV TOKENS, so it rendered as "0.11 MB" -- a number with no
                # meaning, on the line being used to diagnose it. The old
                # plane's byte figure also folds in model weights, so its
                # percentage starts at 78.96% with an empty cache and is not
                # comparable to the program plane's without saying so.
                if hasattr(schedulers[inst_id], "memory"):
                    line += (f"Each NPU Memory Usage "
                             f"{_r['npu_used'] / MB_TO_BYTE:.2f} MB "
                             f"({npu_util:.3f} % Used, weights included)")
                else:
                    line += (f"KV {_r['npu_used']:,}/{_r['npu_mem']:,} tok "
                             f"({npu_util:.3f} % Used; "
                             f"locked {_r['cache_protected']:,} "
                             f"cached {_r['cache_evictable']:,} "
                             f"reserved {_r['reserved_tokens']:,})")
                if schedulers[inst_id].enable_prefix_caching:
                    if hasattr(schedulers[inst_id], "memory"):
                        line += schedulers[inst_id].memory.npu_prefix_cache.format_prefix_info()
                print_markup(line)

            ######### Per Node Metrics #########
            if node2inst_mapping:
                num_nodes = len(node2inst_mapping)
                for i, (node_id, inst_ids) in enumerate(node2inst_mapping.items()):
                    node_cpu_usage = 0
                    inst_usage = []
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        node_cpu_usage = prefix_pools[node_id].total_size() * prefix_pools[node_id].kv_size
                    else:
                        for inst_id in inst_ids:
                            inst_cpu_usage = schedulers[inst_id].mem_report()["cpu_used"]
                            node_cpu_usage += inst_cpu_usage
                            inst_usage.append(inst_cpu_usage)

                    cpu_capacity = cpu_mem_size[node_id] * GB_TO_BYTE
                    cpu_util = (node_cpu_usage / cpu_capacity) * 100 if cpu_capacity else 0.0
                    if prefix_storage != "CXL" and not power_modeling and i == num_nodes - 1:
                        tree_indent = '└─'
                    line = (
                        f"{log_indent+tree_indent}Node\\[{node_id}]: "
                        f"Total CPU Memory Usage {node_cpu_usage/MB_TO_BYTE:.2f} MB, "
                        f"{cpu_util:.3f} % Used "
                    )
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        line += prefix_pools[node_id].format_prefix_info()

                    if (any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU") or (len(inst_ids) == 1):
                        print_markup(line)
                    else:
                        parts = []
                        for j, inst_cpu_usage in enumerate(inst_usage):
                            inst_cpu_util = (inst_cpu_usage / node_cpu_usage)*100 if node_cpu_usage else 0
                            parts.append(f"Instance\\[{inst_ids[j]}]: {inst_cpu_util:.2f} %")
                        print_markup(line + "(" + ", ".join(parts) + ")")

            ######### Per CXL Metrics #########
            if any_prefix_caching and prefix_storage == "CXL":
                if enable_prefix_sharing:
                    num_prefix_pool = len(prefix_pools)
                    for cxl_id, cxl_pool in enumerate(prefix_pools):
                        cxl_usage = cxl_pool.total_size() * cxl_pool.kv_size
                        cxl_util = cxl_usage / cxl_pool.capacity
                        if not power_modeling and cxl_id == num_prefix_pool - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[{cxl_id}]: "
                            f"Total CXL Device Memory Usage "
                            f"{cxl_usage/MB_TO_BYTE:.2f}MB, {cxl_util:.3f} % Used"
                        )
                else:
                    enabled_inst_ids = [
                        inst_id for inst_id, sched in enumerate(schedulers)
                        if sched.enable_prefix_caching
                    ]
                    for pos, inst_id in enumerate(enabled_inst_ids):
                        second_tier = getattr(
                            getattr(schedulers[inst_id], "memory", None),
                            "second_tier_prefix_cache", None)
                        if second_tier is None:
                            continue
                        cxl_usage = second_tier.total_size() * second_tier.kv_size
                        cxl_util = cxl_usage / second_tier.capacity
                        if not power_modeling and pos == len(enabled_inst_ids) - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[0]/Instance\\[{inst_id}]: "
                            f"Total CXL Device Memory Usage {cxl_usage / MB_TO_BYTE:.2f} MB, "
                            f"{cxl_util:.3f} % Used"
                        )

            ######### Power Modeling #########
            if power_modeling:
                tree_indent = '└─'
                print_markup(
                    f"{log_indent+tree_indent}"
                    f"Avg power consumption: {power_model.get_current_power(current)} W"
                )
        # check if all requests are done for current instance#
        # NOTE: 'instance_id' could occur in duplicate, because 'npu2inst_mapping[sys]' is not one-to-one mapping
        if (instance_id not in decode_instance or is_prefill_done) and instance_id not in done_instance and schedulers[instance_id].is_request_empty() and not plane_has_pending():
            # For DP groups: only mark done when ALL members of the group are empty
            dg = inst_dp_group.get(instance_id)
            if dg is not None:
                all_dp_empty = all(
                    schedulers[inst_id].is_request_empty() and len(schedulers[inst_id].inflight) == 0
                    for inst_id in dp_groups[dg]
                )
                if not all_dp_empty:
                    # Other DP members still have work — keep this instance alive for dummy waves
                    if not responded:
                        controller.write_flush(p, "pass")
                    flush.stdout.flush()
                    continue

            if sys not in done_inst_npus[instance_id]:
                done_inst_npus[instance_id].append(sys)
            if len(done_inst_npus[instance_id]) == (1 if instances[instance_id]["num_npus"] == 1 else 2):
                done_instance.append(instance_id)

            # check if all prefill instances are done
            if len(done_instance) == len(prefill_instance):
                is_prefill_done = True

            # check if all instances are done
            if len(done_instance) == num_instances:
                if policy_adapter is not None:
                    # Unpark protected prefix chains before the cache is
                    # freed, else locked nodes survive and read as a leak.
                    policy_adapter.retention_exec.finish()
                for inst_idx in range(num_instances):
                    # teardown() frees and reports whether everything went back;
                    # the planes answer it from their own structure.
                    schedulers[inst_idx].teardown()

                print_rule()
                print_markup("[sim.heading]▶ Exiting simulation...[/]\n")
                controller.write_flush(p, "exit")
                break
            controller.write_flush(p, "done") # make done instances to sleep
        elif new_req == None and not responded:
            # This instance has no runnable batch. If the WHOLE system is idle
            # (every instance drained, nothing in flight) but a pending request
            # will arrive in the future (an agentic tool call is still running),
            # fast-forward wall-clock time in PYTHON rather than driving
            # ASTRA-Sim: the backend has no event to advance, so a "pass" would
            # deadlock (mutual pipe_read). Jump to the next arrival, bank the
            # skipped time into idle_offset, pull the arrival in, and re-schedule
            # WITHOUT reading the backend (skip_read) — the next pass through the
            # loop hands ASTRA-Sim a real workload.
            # Only the instance's start NPU may fast-forward: a new batch is
            # created only when schedule() is called with sys == start NPU,
            # so fast-forwarding on a non-start NPU (tp/pp > 1) would find no
            # batch on the re-scheduling pass and jump again, arrival after
            # arrival, until the pending list is exhausted (the first batch
            # then formed only once the LAST program had arrived). ASTRA-Sim
            # polls end NPUs before start NPUs each round and re-polls a
            # passed NPU next round, so "pass" here is safe: the start NPU's
            # report follows, fast-forwards, and forms the batch.
            system_idle = all(len(schedulers[i].inflight) == 0 for i in range(num_instances))
            next_arrival = plane_next_arrival()
            swap_wakeups = []
            # Queued host copies retain GPU sources. If no batch fits, the
            # min-waste decision can change at a future time even without a
            # new arrival. Include that event rather than declaring deadlock
            # before the policy has a chance to release the source.
            if system_idle and args.planes != 'program':
                for sc in schedulers:
                    swap = sc.memory.host_swap
                    if swap is None:
                        continue
                    wake = swap.next_reconsider_ns(
                        current, [r for r in sc.request if r.arrival <= current])
                    if wake is not None:
                        swap_wakeups.append((swap, wake))
                        next_arrival = wake if next_arrival is None else min(next_arrival, wake)
            idle_sweep_complete = False
            if system_idle and sys == inst2npu_mapping[instance_id]:
                idle_sweep_complete = idle_sweep.record_failure(
                    instance_id,
                    [i for i, sc in enumerate(schedulers) if not sc.is_request_empty()])
            if (system_idle and idle_sweep_complete
                    and next_arrival is not None and next_arrival > current
                    and sys == inst2npu_mapping[instance_id]):
                idle_sweep.reset()
                for swap, wake in swap_wakeups:
                    if wake == next_arrival:
                        swap.stats['idle_reconsider_wakeups'] = swap.stats.get('idle_reconsider_wakeups', 0) + 1
                idle_offset += next_arrival - current
                current = next_arrival
                if dataset is not None:
                    plane_dispatch(current)
                skip_read = True
                flush.stdout.flush()
                continue
            if (system_idle and idle_sweep_complete and next_arrival is None
                    and sys == inst2npu_mapping[instance_id]
                    and any(not schedulers[i].is_request_empty() for i in range(num_instances))):
                # Every pending instance has failed a start-NPU scheduling
                # attempt since the last progress event. One idle instance
                # alone cannot establish this: other instances may not have
                # received their next backend poll yet.
                lines = []
                for i in range(num_instances):
                    sc = schedulers[i]
                    if args.planes == 'program':
                        # The program planes have no MemoryModel, and skipping
                        # them here meant a deadlock on this path raised with no
                        # state at all -- 64 minutes of board cell to learn only
                        # that it stopped. Report the same facts from the plane
                        # that owns them.
                        occ = sc.kv.occupancy()
                        lines.append(
                            f"instance {i}: {len(sc.waiting)} waiting, "
                            f"{len(sc.running)} running, inflight {len(sc.inflight)}; "
                            f"pool {sc.kv.capacity_tokens} tok "
                            f"locked {occ['locked']} pinned {occ['pinned']} "
                            f"cached {occ['cached']} free {occ['free']}; "
                            f"reserved {sc.kv.inflight_tokens()} "
                            f"lock_holders {len(sc.kv._lock_holder)}")
                        lines.append(f"  counters: {sc.counters}")
                        lines.append(f"  last_none_reason: {sc._none_reason}")
                        for r in (sc.running + sc.waiting)[:8]:
                            take = 1 if not r.is_prefill() else 16
                            key = sc._key(r, r.num_computed_tokens + take)
                            lines.append(
                                f"  req {r.id}: input={r.original_input} "
                                f"computed={r.num_computed_tokens} out={r.output} "
                                f"prefill={r.is_prefill()} hit={r.npu_cache_hit} "
                                f"locked={r._prefix_locked} admit_seq={r.admit_seq} "
                                f"n_preempted={r.n_preempted} arrival={r.arrival}")
                            # Reservations are sized by the STEP now, not by
                            # `len(key) - probe(key)`: `needed()` is gone, and
                            # this line still called it, so the diagnostic that
                            # exists to explain a stall raised AttributeError
                            # instead and reported nothing at all.
                            lines.append(
                                f"      step wants {take} tok, "
                                f"{sc.kv.probe(key)}/{len(key)} of its context "
                                f"cached, reclaimable {sc.kv.reclaim_target(take)}, "
                                f"can_fit={sc.kv.can_fit(take)}")
                        continue
                    if not hasattr(sc, "memory"):
                        continue
                    mm = sc.memory; c = mm.npu_prefix_cache
                    kp = mm.kv_protection
                    real_parked = (sum(e.tokens for e in kp._parked.values() if e.memory is mm)
                                   if kp is not None and hasattr(kp, "_parked") else 0)
                    lines.append(
                        f"instance {i}: {len(sc.request)} pending, inflight {len(sc.inflight)}; "
                        f"npu used {mm.npu_used / 2**20:.0f}MB of {mm.npu_mem / 2**20:.0f}MB (weights {mm.weight / 2**20:.0f}MB), "
                        f"reserved {mm.npu_reserved / 2**20:.0f}MB, cache total {c.total_size()} tok "
                        f"evictable {c.evictable_size()} locked {c.protected_size()} "
                        f"parked_adm {mm.parked_size(Device.NPU) // max(1, mm._bytes_per_token)} "
                        f"parked_real {real_parked} tok")
                    lines.append(f"  last_none_reason: {sc._none_reason}")
                    for r in sc.request[:8]:
                        lines.append(f"  req{r.id}: input={r.original_input} computed={r.num_computed_tokens} out={r.output} "
                                     f"hit={r.npu_cache_hit} locked={r._prefix_locked} evict={r.evict} "
                                     f"admit_seq={r.admit_seq} preempt_seq={r.preempt_seq} arrival={r.arrival}")
                raise RuntimeError("scheduler stuck: no runnable batch, nothing in flight, no future arrival, "
                                   "but requests pending\n" + "\n".join(lines))
            # Partial idle (other instances still have work in flight): a "pass"
            # is safe here because ASTRA-Sim still has events to report.
            controller.write_flush(p, "pass")
        
        # flush
        flush.stdout.flush()

    # calculate simulation time
    end_time = time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # check all scheduled requests in astra-sim are well done
    controller.check_end(p)

    print_markup("Recompute preemptions (all instances):                              "
                 f"{sum(sc.num_preemptions for sc in schedulers)}")
    if output_file and any(sc.preemption_log for sc in schedulers):
        _pp = (output_file[:-4] if output_file.endswith(".csv") else output_file) + "_preemptions.csv"
        # cwd is astra-sim/ at this point; repo-relative outputs live one level up
        # (same resolution as Scheduler.save_output).
        if not os.path.isabs(_pp) and not os.path.isdir(os.path.dirname(_pp)) and os.path.isdir(os.path.join("..", os.path.dirname(_pp))):
            _pp = os.path.join("..", _pp)
        with open(_pp, "w") as _f:
            _f.write("instance,sim_time_s,victim_id,computed_tokens,generated_tokens,kv_needed_bytes,avail_bytes,evictable_bytes,victim_admit_seq,admit_counter,batch_len,temp_len,victim_input,victim_cache_hit\n")
            for _i, sc in enumerate(schedulers):
                for _row in sc.preemption_log:
                    (t, rid, comp, gen, need, av, ev,
                     aseq, actr, blen, tlen, vin, vhit) = _row
                    _f.write(f"{_i},{t / 1e9:.3f},{rid},{comp},{gen},{need},{av},{ev},"
                             f"{aseq},{actr},{blen},{tlen},{vin},{vhit}\n")
        print(f"Saving preemption log to output file: {_pp}")
    if output_file and _fp_rows:
        _fpp = (output_file[:-4] if output_file.endswith(".csv") else output_file) + "_footprint.csv"
        if not os.path.isabs(_fpp) and not os.path.isdir(os.path.dirname(_fpp)) and os.path.isdir(os.path.join("..", os.path.dirname(_fpp))):
            _fpp = os.path.join("..", _fpp)
        with open(_fpp, "w") as _f:
            _f.write("instance,sim_time_s,waiting,inflight,referenced_tokens,cache_tokens,evictable_tokens,locked_tokens,kv_used_frac,reserved_tokens\n")
            for row in _fp_rows:
                _f.write(",".join(str(x) for x in row) + "\n")
        print(f"Saving footprint series to output file: {_fpp}")
    # calcuate prefix caching metrics
    total_requested_tokens = 0
    total_npu_hit_tokens = 0
    total_cpu_hit_tokens = 0
    if any_prefix_caching:
        for i in range(num_instances):
            if not schedulers[i].enable_prefix_caching:
                continue
            (temp_npu_a, temp_npu_b), (temp_cpu_a, temp_cpu_b) = schedulers[i].return_prefix_info()
            if (not enable_prefix_sharing) and (prefix_storage != "None") and (temp_npu_a != temp_cpu_a):
                raise RuntimeError(f"Instance[{i}] prefix caching requested tokens mismatch between NPU ({temp_npu_a}) and CPU ({temp_cpu_a})")
            total_requested_tokens += temp_npu_a
            total_npu_hit_tokens += temp_npu_b
            if not enable_prefix_sharing:
                total_cpu_hit_tokens += temp_cpu_b
        
        if enable_prefix_sharing:
            for pool in prefix_pools:
                _, temp_cpu_b = pool.return_prefix_info()
                total_cpu_hit_tokens += temp_cpu_b
    
    # This is total system's throughput
    total_latency = current/FREQ
    print_rule()
    print_markup("[sim.heading]▶ Simulation results...[/]\n")
    print_markup(f"Total simulation time: {int(hours)}h {int(minutes)}m {seconds:.3f}s")
    print_rule("[sim.tagline]Throughput Results[/]")
    print_markup(f"Total requests:                                                     {req_cnt}")
    print_markup(f"Total clocks (ns):                                                  {current}")
    print_markup(f"Total latency (s):                                                  {total_latency:.3f}")
    print_markup(f"Total input tokens:                                                 {total_prompt}")
    print_markup(f"Total generated tokens:                                             {total_gen}")
    print_markup(f"Request throughput (req/s):                                         {req_cnt/total_latency:.2f}")
    print_markup(f"Average prompt throughput (tok/s):                                  {total_prompt/total_latency:.2f}")
    print_markup(f"Average generation throughput (tok/s):                              {total_gen/total_latency:.2f}")
    print_markup(f"Total token throughput (tok/s):                                     {(total_prompt + total_gen)/total_latency:.2f}")
    print_markup(f"Throughput per {1/RATIO} sec (\\[prompt_throughput], \\[gen_throughput]): {throughput}")
    print_rule()
    if any_prefix_caching:
        print_rule("[sim.tagline]Prefix Caching Results[/]")
        print_markup(f"Total requested prompt tokens:                                      {total_requested_tokens}")
        print_markup(f"NPU prefix hit prompt tokens:                                       {total_npu_hit_tokens}")
        if total_requested_tokens > 0:
            print_markup(f"NPU prefix hit ratio (%):                                           {(total_npu_hit_tokens/total_requested_tokens)*100:.2f}")
            if prefix_storage != "None":
                print_markup(f"{prefix_storage} prefix hit prompt tokens:                                       {total_cpu_hit_tokens}")
                print_markup(f"{prefix_storage} prefix hit ratio (%):                                           {(total_cpu_hit_tokens/total_requested_tokens)*100:.2f}")
            print_markup(f"Total prefix hit ratio (%):                                         {((total_npu_hit_tokens+total_cpu_hit_tokens)/total_requested_tokens)*100:.2f}")
        else:
            print_markup("NPU prefix hit ratio (%):                                           N/A (no requests tracked)")
        print_rule()
    if power_modeling:
        print_rule("[sim.tagline]Power Modeling Results[/]")
        total_energy = power_model.get_final_energy(current)
        print_markup(f"Total energy consumption (kJ):                                      {total_energy/1000:.2f}")
        # Each node results
        power_model.print_power_summary()
        print_markup(f"Power per {1/RATIO} sec (W): {power_model.power_time_series}")
        print_rule()
    # Each instacne results
    for i in range(num_instances):
        print_rule(f"[sim.tagline]Instance \\[{i}][/]")
        schedulers[i].print_result()
        print_rule()

    # Multi-agent workflow-level metrics: JCT (mean + tail) and workflow
    # throughput, aggregated across every completed DAG workflow. This is the
    # deliverable for the multi-agent simulator; it is empty for flat/chain
    # workloads.
    if plane_has_workflow_metrics():
        wf = plane_workflow_metrics_summary()
        print_rule("[sim.tagline]Multi-Agent Workflow Results[/]")
        print_markup(f"Total workflows:                                                    {wf['num_workflows']}")
        print_markup(f"Workflow JCT mean (s):                                              {wf['jct_mean_ns']/FREQ:.3f}")
        print_markup(f"Workflow JCT P50 (s):                                               {wf['jct_p50_ns']/FREQ:.3f}")
        print_markup(f"Workflow JCT P90 (s):                                               {wf['jct_p90_ns']/FREQ:.3f}")
        print_markup(f"Workflow JCT P99 (s):                                               {wf['jct_p99_ns']/FREQ:.3f}")
        print_markup(f"Workflow JCT min / max (s):                                         {wf['jct_min_ns']/FREQ:.3f} / {wf['jct_max_ns']/FREQ:.3f}")
        print_markup(f"Workflow throughput (workflows/s):                                  {wf['workflow_throughput_per_s']:.3f}")
        print_rule()

    # Important informations about metrics
    # The TTFT (Time to First Token) in our simulator differs from vllm. 
    # While vllm measures TTFT as the time when the client receives the first token,
    # Our simulator measures it as the time when the computation of the first token is completed.
    # Therefore, vllm gets much more higher TTFT.
    # (Ref: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com#interval-calculations-vs-preemptions)

    if output_file != None:
        print(f"Saving each request's information to output file: {output_file}")
        for i in range(num_instances):
            schedulers[i].save_output(output_file, is_append=False if i == 0 else True)
        if plane_has_workflow_metrics():
            wf_path = (output_file[:-4] if output_file.endswith(".csv") else output_file) + "_workflows.csv"
            plane_save_workflow_metrics(wf_path)
            print(f"Saving per-workflow metrics to output file: {wf_path}")

    if program_policy is not None:
        path = program_policy.finish()
        print_rule("[sim.tagline]Program-Aware Policy[/]")
        for k, v in program_policy.snapshot().items():
            print_markup(f"{k:<68s}{v}")
        if path:
            print(f"Saving policy decisions to {path}")
        print_rule()
    if policy_adapter is not None:
        # Release still-parked protections and flush the decision logs.
        policy_adapter.finish()
        stats = policy_adapter.kv_stats()
        print_rule("[sim.tagline]Unified Serving Policy[/]")
        print_markup(f"Retention decisions logged:                                         {len(policy_adapter.retention_exec.decisions)}")
        print_markup(f"KV protection stats:                                                {stats}")
        if policy_adapter.routing_exec is not None:
            print_markup(f"Routing decisions logged:                                           {len(policy_adapter.routing_exec.decisions)}")
        print_markup(f"Priority stamps logged:                                             {len(policy_adapter.scheduling_exec.stamps)}")
        print_rule()

    if args.cleanup_inputs:
        _cleanup_inputs_root(run_paths, logger)
    

if __name__ == "__main__": 
    # For simulation time breakdown
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
