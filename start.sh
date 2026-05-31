#!/usr/bin/env bash
#
# Strategy Builder launcher.
#
#   ./start.sh          # dev: FastAPI (8000, --reload) + Vite (3000, proxies /api) -> open http://localhost:3000
#   ./start.sh prod     # build frontend, then serve UI + API from FastAPI on 8000 -> open http://localhost:8000
#
# Ctrl-C stops everything.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/venv/bin/python"
UVICORN="$ROOT/venv/bin/uvicorn"
MODE="${1:-dev}"

if [[ ! -x "$UVICORN" ]]; then
  echo "ERROR: $UVICORN not found. Create the venv and install deps first:"
  echo "  python3 -m venv venv && venv/bin/pip install -e \".[dev]\""
  exit 1
fi

# Install frontend deps on first run.
if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo ">> installing frontend dependencies (first run)..."
  (cd "$ROOT/frontend" && npm install)
fi

if [[ "$MODE" == "prod" ]]; then
  echo ">> building frontend..."
  (cd "$ROOT/frontend" && npm run build)
  echo ">> serving UI + API on http://localhost:8000"
  exec "$UVICORN" strategy_api.main:app --port 8000
fi

# --- dev mode: backend + frontend, clean shutdown on exit ---
PIDS=()
cleanup() {
  echo
  echo ">> stopping..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo ">> backend  -> http://localhost:8000 (FastAPI, --reload)"
"$UVICORN" strategy_api.main:app --reload --port 8000 &
PIDS+=($!)

echo ">> frontend -> http://localhost:3000 (Vite, proxies /api)"
(cd "$ROOT/frontend" && npm run dev) &
PIDS+=($!)

echo
echo ">> open http://localhost:3000   (Ctrl-C to stop both)"
wait
