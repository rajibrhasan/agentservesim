

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

uv venv --python 3.12

# vLLM brings in torch, pydantic, pyyaml, and rich as transitive
# dependencies — no need to list them separately.
# The source is vendored without upstream Git metadata: pin its version and kernels.
export VLLM_VERSION_OVERRIDE=0.19.0
export VLLM_PRECOMPILED_WHEEL_COMMIT=2a69949bdadf0e8942b7a1619b229cb475beef20
VLLM_USE_PRECOMPILED=1 uv pip install --editable "$REPO_ROOT/vllm" --verbose

# Extra deps for workloads.generators (HF dataset loading) and
# bench.core.plots (matplotlib).
uv pip install datasets matplotlib
