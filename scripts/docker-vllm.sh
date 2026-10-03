

set -euo pipefail

# Resolve the repo root regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim

docker run --name vllm_docker \
  --gpus all \
  -it \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$REPO_ROOT":/workspace \
  --volume "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --shm-size=16g \
  -w /workspace \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.19.0 \
  -c "pip install datasets matplotlib && exec bash"
