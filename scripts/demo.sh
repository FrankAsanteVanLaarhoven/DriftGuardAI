#!/usr/bin/env bash
# End-to-end local proof of DriftGuard. Run: make demo
set -uo pipefail

# Keep the venv hermetic: a sourced ROS setup leaks Python 3.10 site-packages onto PYTHONPATH,
# which breaks this project's uv/venv. The Makefile unexports it; do the same for direct uv calls.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Serve on a free port by default (override with `PORT=8010 make demo`). Hardcoding 8000 clashes
# with anything already bound there and makes the health probe hit a foreign service (spurious 404).
pick_port() {
  uv run python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}
PORT="${PORT:-$(pick_port)}"
BASE="http://127.0.0.1:${PORT}"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
restore_pointer() { printf 'artifacts/primary.joblib' > models/primary_pointer; }

# Always tear down the server AND restore the primary pointer on exit — even if a step fails, so a
# failed run never leaves the repo on a deleted pointer (which would break the latency-breach test).
cleanup() { [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null; restore_pointer; }
trap cleanup EXIT

say "1/6 Install + train (real ag_news, seeded)"
make install
[ -f artifacts/primary.joblib ] || make train
restore_pointer   # ensure the pointer exists before we serve

say "2/6 Test suite (unit + integration + FALLBACK chaos test)"
make test

say "3/6 Start the service on port $PORT"
uv run uvicorn driftguard.api.main:app --host 127.0.0.1 --port "$PORT" >/tmp/driftguard-demo.log 2>&1 &
SERVER_PID=$!
up=""
for _ in $(seq 1 40); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: uvicorn exited before becoming healthy (port $PORT in use?). Log:"
    tail -20 /tmp/driftguard-demo.log; exit 1
  fi
  curl -sf "$BASE/health" >/dev/null 2>&1 && { up=1; break; }
  sleep 1
done
[ -n "$up" ] || { echo "FAIL: server never healthy on $BASE"; tail -20 /tmp/driftguard-demo.log; exit 1; }

say "4/6 Predict with the PRIMARY model"
curl -s -X POST "$BASE/predict" -H 'content-type: application/json' \
  -d '{"text":"New GPU sets an on-device AI record."}'; echo

say "5/6 Remove the primary — service must stay up on the BASELINE"
rm -f models/primary_pointer
code=$(curl -s -o /tmp/driftguard-after.json -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'content-type: application/json' -d '{"text":"New GPU sets an on-device AI record."}')
echo "HTTP $code -> $(cat /tmp/driftguard-after.json)"
[ "$code" = "200" ] || { echo "FAIL: expected 200 during fallback"; exit 1; }
grep -q '"served_by":"baseline"' /tmp/driftguard-after.json \
  && echo "OK: served by baseline while degraded" || { echo "FAIL: not baseline"; exit 1; }
restore_pointer   # explicit restore (the EXIT trap guarantees it too)

say "6/6 Drift detection (PSI, non-zero exit on drift)"
echo "-- stable sample:"; uv run python -m driftguard.drift artifacts/current_baseline.json \
  && echo "exit 0 (no drift)"
echo "-- shifted sample:"; uv run python -m driftguard.drift artifacts/current_shifted.json \
  || echo "exit non-zero (DRIFT flagged) ✓"

say "Demo complete — service never 5xx'd, baseline covered the primary outage."
