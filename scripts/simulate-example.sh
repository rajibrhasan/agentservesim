#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
DATASET="${DATASET:-$ROOT/workloads/bfcl_phi_jps0.8_n150.jsonl}"
[[ -f "$DATASET" ]] || { echo "Missing workload: $DATASET" >&2; exit 1; }
OUTPUT="${OUTPUT:-$ROOT/outputs/example-sim/requests.csv}"
mkdir -p "$(dirname "$OUTPUT")"
export SIM_KV_INVARIANT=0 SIM_KV_UTIL_SEMANTICS=vllm
export SIM_ADMIT_VALVE=1 SIM_DERIVE_OUTPUT_IDS=0
# Keep generated traces/graphs on node-local scratch, as in the recorded run.
INPUTS_ROOT="$(mktemp -d "${TMPDIR:-/dev/shm}/continuum-sim.XXXXXX")"
cmd=("$PYTHON" -m serving --inputs-root "$INPUTS_ROOT"
  --cluster-config "${CLUSTER_CONFIG:-$ROOT/configs/cluster/example_b200_phi_continuum.json}"
  --dataset "$DATASET" --output "$OUTPUT" --planes request
  --dtype bfloat16 --kv-cache-dtype auto --block-size 16
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --max-num-seqs "${MAX_NUM_SEQS:-128}"
  --max-num-batched-tokens "${MAX_BATCH_TOKENS:-16384}"
  --enable-prefix-caching --enable-chunked-prefill
  --retention "${RETENTION:-continuum}" --scheduling "${SCHEDULING:-continuum}"
  --retention-tau 2 --harness-root "$ROOT" --num-reqs 0 --log-level WARNING "$@")
printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
exec "${cmd[@]}"
