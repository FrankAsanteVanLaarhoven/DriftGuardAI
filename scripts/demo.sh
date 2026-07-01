#!/usr/bin/env bash
# End-to-end local proof of DriftGuard. Run: make demo
set -uo pipefail

PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "1/6 Install + train (real ag_news, seeded)"
make install
[ -f artifacts/primary.joblib ] || make train

say "2/6 Test suite (unit + integration + FALLBACK chaos test)"
make test

say "3/6 Start the service"
uv run uvicorn driftguard.api.main:app --host 127.0.0.1 --port "$PORT" >/tmp/driftguard-demo.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done

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
printf 'artifacts/primary.joblib' > models/primary_pointer   # restore pointer

say "6/6 Drift detection (PSI, non-zero exit on drift)"
echo "-- stable sample:"; uv run python -m driftguard.drift artifacts/current_baseline.json \
  && echo "exit 0 (no drift)"
echo "-- shifted sample:"; uv run python -m driftguard.drift artifacts/current_shifted.json \
  || echo "exit non-zero (DRIFT flagged) ✓"

say "Demo complete — service never 5xx'd, baseline covered the primary outage."
