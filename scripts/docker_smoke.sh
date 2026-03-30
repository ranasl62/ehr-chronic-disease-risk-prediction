#!/usr/bin/env bash
# Build API image and smoke-test HTTP endpoints (requires Docker).
# Optional: train model.pkl first so /v1/ready returns 200.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-ehr-risk-smoke}"
PORT="${PORT:-9876}"

docker build -f "$ROOT/deployment/Dockerfile" -t "$IMAGE" "$ROOT"

EXTRA_ARGS=()
if [[ -f "$ROOT/model.pkl" ]]; then
  EXTRA_ARGS+=( -v "$ROOT/model.pkl:/app/model.pkl:ro" )
fi

CID="$(docker run -d "${EXTRA_ARGS[@]}" -p "${PORT}:8000" "$IMAGE")"
cleanup() { docker rm -f "$CID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

sleep 2
curl -sf "http://127.0.0.1:${PORT}/health" | grep -q .
if [[ -f "$ROOT/model.pkl" ]]; then
  curl -sf "http://127.0.0.1:${PORT}/v1/ready" | grep -q .
  if [[ -f "$ROOT/reports/evaluation_report.json" ]]; then
    curl -sf "http://127.0.0.1:${PORT}/v1/model/metrics" | grep -q .
  fi
fi
echo "docker_smoke OK (port ${PORT})"
