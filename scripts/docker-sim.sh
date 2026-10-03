

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${SIM_IMAGE:-agentservesim:latest}"

if [[ -n "${REBUILD:-}" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE ..."
  docker build -t "$IMAGE" "$REPO_ROOT"
fi

exec docker run --rm -it \
  -v "$REPO_ROOT":/app/LLMServingSim \
  -w /app/LLMServingSim \
  "$IMAGE" bash
