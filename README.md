# AgentServeSim

Agent-serving simulation, profiling, policy search, and reference replay with patched vLLM v0.19.0.

## Layout

| Directory                                 | Contents                              |
| ----------------------------------------- | ------------------------------------- |
| `serving/`                              | Simulator                             |
| `bench/`                                | Reference replay and measurement      |
| `profiler/`                             | Profiling code and performance tables |
| `policies/`, `harness/`, `runtime/` | Policies and shared execution support |
| `evolve/`                               | Policy search                         |
| `configs/`                              | Model and cluster configurations      |
| `astra-sim/`, `vllm/`                 | Included backend sources              |

Commands below run from the repository root. The example is **B200 / Phi-3.5-MoE / TP1 / Continuum / BFCL150 at 0.8 programs/s**. 

The workload is included in `workloads/bfcl_phi_jps0.8_n150.jsonl.gz`: **14.5 MiB compressed, 120 MiB extracted**, containing 150 programs and 1,370 turns. Extract it once before simulation or reference serving:

```sh
gzip -dk workloads/bfcl_phi_jps0.8_n150.jsonl.gz
```

## Profiling

On one B200, install the patched vLLM environment and profile:

```sh
bash scripts/install-vllm.sh
source .venv/bin/activate
bash scripts/profile-example.sh
```

## Simulation

Build and enter the CPU simulator container, then run:

```sh
bash scripts/docker-sim.sh
# Inside the container:
bash scripts/compile.sh
bash scripts/simulate-example.sh
```

Outputs go to `outputs/example-sim/`.

## Reference Serving

On one B200, using the patched vLLM environment installed above:

```sh
source .venv/bin/activate
bash scripts/reference-example.sh
```

Outputs go to `outputs/example-real/`. 

## Policy Search

Configure the LLM endpoint in `evolve/config_local.yaml` or `evolve/config_openrouter.yaml`. Inside the simulator container:

```sh
EVOLVE_DATASET=/path/to/search-trace.jsonl \
EVOLVE_CLUSTER_CONFIG=/path/to/search-cluster.json \
  bash scripts/search-example.sh
```
