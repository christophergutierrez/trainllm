#!/usr/bin/env bash
# Start both backend and frontend dev servers.
# Usage: ./web/run_dev.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DIR")"

cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
}
trap cleanup EXIT

echo "Starting backend (port 8080)..."
cd "$ROOT"
~/.unsloth/studio/unsloth_studio/bin/uvicorn web.backend.main:app \
    --host 0.0.0.0 --port 8080 --reload &
BACKEND_PID=$!

echo "Starting frontend (port 5173)..."
cd "$DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Dashboard: http://localhost:5173"
echo "API:       http://localhost:8080/api/health"
echo ""
echo "Press Ctrl+C to stop both servers."

wait
