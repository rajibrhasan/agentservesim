#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
# One physical GPU; TP shapes are emulated on it. Existing shots resume.
cmd=("$PYTHON" -m profiler profile "${MODEL:-microsoft/Phi-3.5-MoE-instruct}"
  --hardware "${HARDWARE:-B200}" --tp "${TP_DEGREES:-1}"
  --dtype bfloat16 --kv-cache-dtype auto
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --max-num-seqs "${MAX_NUM_SEQS:-128}"
  --max-num-batched-tokens "${MAX_BATCH_TOKENS:-16384}"
  --attention-max-kv "${ATTENTION_MAX_KV:-131072}"
  --measurement-iterations 3
  --out-root "${OUT_ROOT:-$ROOT/profiler/perf}" "$@")
printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
exec "${cmd[@]}"
