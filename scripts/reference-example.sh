#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
DATASET="${DATASET:-$ROOT/workloads/bfcl_phi_jps0.8_n150.jsonl}"
[[ -f "$DATASET" ]] || { echo "Missing workload: $DATASET" >&2; exit 1; }
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/example-real}"
mkdir -p "$OUTPUT_DIR"
# Reject an accidentally installed stock engine before allocating GPU resources.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "$PYTHON" - "$ROOT" <<'CHECK'
import sys
from pathlib import Path
import vllm
expected = Path(sys.argv[1]) / 'vllm' / 'vllm'
if Path(vllm.__file__).resolve().parent != expected.resolve():
    raise SystemExit('Install the included patched engine with scripts/install-vllm.sh; '
                     f'currently importing {vllm.__file__}')
CHECK
fi
export VLLM_SERVICE_RETENTION_DURING_EXECUTION=1
export VLLM_KV_PROTECTION=1 VLLM_KV_PIN_AT_FREE_TTL=2
export VLLM_KV_RELEASE_AT_ARRIVAL=1
cmd=("$PYTHON" -m bench run --decision-log-dir "$OUTPUT_DIR/decisions"
  --model "${MODEL:-microsoft/Phi-3.5-MoE-instruct}"
  --dataset "$DATASET" --output-dir "$OUTPUT_DIR"
  --tensor-parallel-size "${TP:-1}" --num-instances 1
  --dtype bfloat16 --kv-cache-dtype auto --block-size 16
  --num-gpu-blocks-override 41065 --gpu-memory-utilization 0.9
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --max-num-seqs "${MAX_NUM_SEQS:-128}"
  --max-num-batched-tokens "${MAX_BATCH_TOKENS:-16384}"
  --enable-prefix-caching --no-async-scheduling
  --retention "${RETENTION:-continuum}" --scheduling "${SCHEDULING:-continuum}"
  --seed 42 --retention-tau 2 --harness-root "$ROOT" --num-reqs 0 --log-level WARNING "$@")
printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
exec "${cmd[@]}"
