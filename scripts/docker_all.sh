#!/usr/bin/env bash
# Same as: docker compose up --build
# Pass flags through, e.g. bash scripts/docker_all.sh -d  (detached)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec docker compose up --build "$@"
