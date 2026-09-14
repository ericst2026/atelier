#!/usr/bin/env bash
# Local development without docker: SQLite + a local redis + the Vite dev server.
#   ./scripts/dev.sh            start api, worker and frontend
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

export ATELIER_ROOT="${ATELIER_ROOT:-$ROOT/.dev}"
export ATELIER_EXPERIMENTS_DIR="${ATELIER_EXPERIMENTS_DIR:-$ROOT/experiments}"
export ATELIER_MATERIALS_DIR="${ATELIER_MATERIALS_DIR:-$ROOT/.dev/materials}"
export ATELIER_DATA_DIR="${ATELIER_DATA_DIR:-$ROOT/.dev/data}"
export ATELIER_REDIS_URL="${ATELIER_REDIS_URL:-redis://localhost:6379/0}"
export ATELIER_GPU_COUNT="${ATELIER_GPU_COUNT:-0}"
export PYTHONPATH="$ROOT/backend"
mkdir -p "$ATELIER_DATA_DIR" "$ATELIER_MATERIALS_DIR"

if ! redis-cli -u "$ATELIER_REDIS_URL" ping >/dev/null 2>&1; then
  echo "redis is not reachable at $ATELIER_REDIS_URL — start one (docker run -p 6379:6379 redis:7-alpine)" >&2
  exit 1
fi

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT

( cd backend && python -m uvicorn atelier.main:app --reload --port 8000 ) & pids+=($!)
( cd backend && python -m atelier.worker ) & pids+=($!)
( cd frontend && npm run dev ) & pids+=($!)

echo "api      http://localhost:8000/api/docs"
echo "frontend http://localhost:5173"
wait
