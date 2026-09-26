#!/bin/bash

# Build (once) and enter the simulator image: ASTRA-Sim's toolchain plus the
# Python packages the engine imports.
#
# One image, two uses: the shell you work in, and the container each
# policy-search evaluation runs in. It used to pip-install the dependencies at
# every container start, which is tolerable for a shell and not for a search
# that starts a container per candidate.
#
#   ./scripts/docker-sim.sh            # shell in the image, repo mounted
#   SIM_IMAGE=other:tag ./scripts/docker-sim.sh
#   REBUILD=1 ./scripts/docker-sim.sh  # rebuild even if the image exists

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
