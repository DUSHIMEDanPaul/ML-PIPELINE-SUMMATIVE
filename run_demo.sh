#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command local demo: builds and starts the whole stack via Docker Compose.
#
#   ./run_demo.sh            # 1 API replica + nginx + UI
#   ./run_demo.sh 4          # 4 API replicas behind the nginx load balancer
#
# API / Swagger : http://localhost:8000/docs
# Streamlit UI  : http://localhost:8501
# ---------------------------------------------------------------------------
set -euo pipefail

SCALE="${1:-1}"

# Pick "docker compose" (v2) or fall back to "docker-compose" (v1).
if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DC="docker-compose"
else
  echo "ERROR: Docker Compose not found. Install Docker Desktop / Compose first." >&2
  exit 1
fi

echo ">> Building and starting stack with api scaled to ${SCALE} ..."
$DC up --build -d --scale "api=${SCALE}"

echo
echo ">> Waiting for the API to become healthy ..."
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    echo ">> API is up."
    break
  fi
  sleep 3
  if [ "$i" -eq 60 ]; then
    echo "WARNING: API did not report healthy within ~3 min. Check: $DC logs api" >&2
  fi
done

echo
echo "============================================================"
echo "  API   (Swagger)  ->  http://localhost:8000/docs"
echo "  UI    (Streamlit) ->  http://localhost:8501"
echo "  API replicas       :  ${SCALE}"
echo "------------------------------------------------------------"
echo "  Logs   : $DC logs -f"
echo "  Stop   : $DC down"
echo "  Test   : ./test_api.sh"
echo "============================================================"
