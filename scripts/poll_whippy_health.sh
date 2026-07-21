#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:4000}"
HEALTH_URL="${BASE_URL}/api/health"
POLL_SECONDS="${2:-180}"
INTERVAL_SECONDS=2
API_LOG="/tmp/whippy-api.log"

echo "Waiting for ${BASE_URL} to listen..."
deadline=$((SECONDS + 120))
while (( SECONDS < deadline )); do
  if ss -ltnp | grep -q ':4000'; then
    echo "Port 4000 is listening."
    break
  fi
  sleep 2
done

if ! ss -ltnp | grep -q ':4000'; then
  echo "Port 4000 never started listening." >&2
  echo "Latest API logs:" >&2
  tail -80 "$API_LOG" >&2 || true
  exit 1
fi

echo "Polling ${HEALTH_URL} for up to ${POLL_SECONDS}s"
echo "Status meanings: timeout/connection failure=not ready, 422=booting, 200=ready"
echo

poll_deadline=$((SECONDS + POLL_SECONDS))
last_status="none"

while (( SECONDS < poll_deadline )); do
  if ! response=$(curl -s -o /dev/null -m 90 -w '%{http_code}' "$HEALTH_URL" 2>/dev/null); then
    last_status="timeout"
    echo "[$(date +%H:%M:%S)] not ready (connection failure or timeout)"
  else
    last_status="$response"
    echo "[$(date +%H:%M:%S)] HTTP ${response}"
    if [[ "$response" == "200" ]]; then
      echo
      echo "API ready."
      exit 0
    fi
    if [[ "$response" == "422" ]]; then
      echo "  -> listening but still booting"
    fi
  fi
  sleep "$INTERVAL_SECONDS"
done

echo
echo "API did not reach HTTP 200 within ${POLL_SECONDS}s (last status: ${last_status})"
echo
echo "Latest API logs:" >&2
tail -80 "$API_LOG" >&2 || true
exit 1
