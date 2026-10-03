#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SIM_CONTAINER="${SIM_CONTAINER:-none}"
: "${EVOLVE_DATASET:?Set EVOLVE_DATASET to the search workload path}"
: "${EVOLVE_CLUSTER_CONFIG:?Set EVOLVE_CLUSTER_CONFIG to the cluster configuration path}"
export EVOLVE_DATASET EVOLVE_CLUSTER_CONFIG
export EVOLVE_SCRATCH="${EVOLVE_SCRATCH:-${TMPDIR:-/tmp}/policy-search}"
export EVOLVE_SEED_PATH="$ROOT/evolve/seed_policy.py"
mkdir -p "$EVOLVE_SCRATCH"
PYTHON="${PYTHON:-python3}"
"$PYTHON" evolve/simrun.py "$EVOLVE_DATASET:$EVOLVE_CLUSTER_CONFIG"
exec "$PYTHON" -m openevolve.cli evolve/seed_policy.py evolve/evaluate.py \
  --config "${SEARCH_CONFIG:-$ROOT/evolve/config_local.yaml}" \
  --output "${OUTPUT_DIR:-$ROOT/outputs/search}" "$@"
