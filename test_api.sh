#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Manual API walkthrough — verifies every endpoint in order before recording
# the demo. Run the stack first (./run_demo.sh), then: ./test_api.sh
#
# Override the target: BASE=http://your-host:8000 ./test_api.sh
# Use your own lesion image for /predict: IMG=/path/to/lesion.jpg ./test_api.sh
#
# JSON is parsed with Python (no `jq` dependency).
# ---------------------------------------------------------------------------
set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

hr() { echo "------------------------------------------------------------"; }
jq_get() { python -c "import sys,json;print(json.load(sys.stdin)$1)"; }

# --- make sample images if the user didn't supply one ----------------------
python - "$TMP" <<'PY'
import sys, numpy as np
from PIL import Image
d = sys.argv[1]
rng = np.random.default_rng(0)
def save(path, tint):
    a = rng.integers(60, 200, size=(128,128,3), dtype=np.uint8)
    a[..., tint] = np.clip(a[..., tint].astype(int)+50, 0, 255)
    Image.fromarray(a, "RGB").save(path, "JPEG", quality=90)
# one standalone predict image
save(f"{d}/sample.jpg", 0)
# a small labelled batch for /upload (0_=benign, 1_=malignant filename prefixes)
import os; os.makedirs(f"{d}/batch", exist_ok=True)
for i in range(4): save(f"{d}/batch/0_benign_{i}.jpg", 2)
for i in range(4): save(f"{d}/batch/1_malig_{i}.jpg", 0)
print("sample images created")
PY

IMG="${IMG:-$TMP/sample.jpg}"

echo "Target API: $BASE"
echo

# === 1. HEALTH =============================================================
hr; echo "1) GET /health"; hr
echo "\$ curl -s $BASE/health"
curl -s "$BASE/health" | tee "$TMP/health.json" | python -m json.tool
echo

# === 2. METRICS ============================================================
hr; echo "2) GET /metrics"; hr
echo "\$ curl -s $BASE/metrics"
curl -s "$BASE/metrics" | python -m json.tool
echo

# === 3. PREDICT ============================================================
hr; echo "3) POST /predict  (single image)"; hr
echo "\$ curl -s -F 'file=@$IMG' $BASE/predict"
curl -s -F "file=@$IMG" "$BASE/predict" | python -m json.tool
echo

# === 4. UPLOAD (bulk) ======================================================
hr; echo "4) POST /upload  (bulk labelled batch)"; hr
echo "\$ curl -s -F 'files=@0_benign_0.jpg' -F 'files=@1_malig_0.jpg' ... $BASE/upload"
UP_ARGS=()
for f in "$TMP"/batch/*.jpg; do UP_ARGS+=(-F "files=@$f"); done
UPLOAD_JSON="$(curl -s "${UP_ARGS[@]}" "$BASE/upload")"
echo "$UPLOAD_JSON" | python -m json.tool
BATCH_ID="$(echo "$UPLOAD_JSON" | jq_get "['batch_id']" 2>/dev/null || true)"
echo
echo ">> batch_id = $BATCH_ID"
echo

if [ -z "${BATCH_ID:-}" ] || [ "$BATCH_ID" = "None" ]; then
  echo "Upload did not return a batch_id; stopping before retrain." >&2
  exit 1
fi

# === 5. RETRAIN (async) ====================================================
hr; echo "5) POST /retrain  (returns a job_id immediately)"; hr
echo "\$ curl -s -H 'Content-Type: application/json' -d '{\"batch_id\":\"$BATCH_ID\",\"epochs\":1,\"subsample\":200}' $BASE/retrain"
RETRAIN_JSON="$(curl -s -H 'Content-Type: application/json' \
  -d "{\"batch_id\":\"$BATCH_ID\",\"epochs\":1,\"subsample\":200}" "$BASE/retrain")"
echo "$RETRAIN_JSON" | python -m json.tool
JOB_ID="$(echo "$RETRAIN_JSON" | jq_get "['job_id']" 2>/dev/null || true)"
echo
echo ">> job_id = $JOB_ID"
echo

# === 6. STATUS (poll) ======================================================
hr; echo "6) GET /status/$JOB_ID  (poll until done/failed)"; hr
if [ -n "${JOB_ID:-}" ] && [ "$JOB_ID" != "None" ]; then
  for i in $(seq 1 120); do
    ST_JSON="$(curl -s "$BASE/status/$JOB_ID")"
    STATUS="$(echo "$ST_JSON" | jq_get "['status']" 2>/dev/null || echo '?')"
    PROG="$(echo "$ST_JSON" | jq_get ".get('progress',0)" 2>/dev/null || echo 0)"
    echo "   [$i] status=$STATUS progress=${PROG}%"
    if [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ]; then
      echo; echo "Final job state:"; echo "$ST_JSON" | python -m json.tool
      break
    fi
    sleep 3
  done
else
  echo "No job_id; skipping status poll." >&2
fi

echo
hr; echo "7) GET /metrics  (after retrain — note request count grew)"; hr
curl -s "$BASE/metrics" | python -m json.tool

echo
echo "Walkthrough complete."
