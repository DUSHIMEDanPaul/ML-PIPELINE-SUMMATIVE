# ---------------------------------------------------------------------------
# Manual API walkthrough for Windows (PowerShell). Verifies every endpoint in
# order before recording the demo. Run the stack first (.\run_demo.ps1), then:
#
#   .\test_api.ps1
#   .\test_api.ps1 -Base http://your-host:8000 -Img C:\path\to\lesion.jpg
# ---------------------------------------------------------------------------
param(
    [string]$Base = "http://localhost:8000",
    [string]$Img  = ""
)
$ErrorActionPreference = "Stop"

$tmp = Join-Path $env:TEMP ("ham_test_" + [guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Force -Path $tmp, (Join-Path $tmp "batch") | Out-Null

function Hr { Write-Host "------------------------------------------------------------" }

# --- generate sample images (unless -Img supplied) -------------------------
python - $tmp @'
import sys, os, numpy as np
from PIL import Image
d = sys.argv[1]
rng = np.random.default_rng(0)
def save(p, tint):
    a = rng.integers(60,200,size=(128,128,3),dtype=np.uint8)
    a[...,tint] = np.clip(a[...,tint].astype(int)+50,0,255)
    Image.fromarray(a,"RGB").save(p,"JPEG",quality=90)
save(os.path.join(d,"sample.jpg"),0)
for i in range(4): save(os.path.join(d,"batch",f"0_benign_{i}.jpg"),2)
for i in range(4): save(os.path.join(d,"batch",f"1_malig_{i}.jpg"),0)
print("sample images created")
'@

if (-not $Img) { $Img = Join-Path $tmp "sample.jpg" }
Write-Host "Target API: $Base`n"

# === 1. HEALTH =============================================================
Hr; Write-Host "1) GET /health"; Hr
Invoke-RestMethod "$Base/health" | ConvertTo-Json -Depth 6
Write-Host ""

# === 2. METRICS ============================================================
Hr; Write-Host "2) GET /metrics"; Hr
Invoke-RestMethod "$Base/metrics" | ConvertTo-Json -Depth 6
Write-Host ""

# === 3. PREDICT ============================================================
Hr; Write-Host "3) POST /predict (single image)"; Hr
$form = @{ file = Get-Item $Img }
Invoke-RestMethod -Uri "$Base/predict" -Method Post -Form $form | ConvertTo-Json -Depth 6
Write-Host ""

# === 4. UPLOAD (bulk) ======================================================
Hr; Write-Host "4) POST /upload (bulk labelled batch)"; Hr
$files = Get-ChildItem (Join-Path $tmp "batch") -Filter *.jpg | ForEach-Object { Get-Item $_.FullName }
$upload = Invoke-RestMethod -Uri "$Base/upload" -Method Post -Form @{ files = $files }
$upload | ConvertTo-Json -Depth 6
$batchId = $upload.batch_id
Write-Host "`n>> batch_id = $batchId`n"

# === 5. RETRAIN (async) ====================================================
Hr; Write-Host "5) POST /retrain (returns job_id immediately)"; Hr
$body = @{ batch_id = $batchId; epochs = 1; subsample = 200 } | ConvertTo-Json
$retrain = Invoke-RestMethod -Uri "$Base/retrain" -Method Post -Body $body -ContentType "application/json"
$retrain | ConvertTo-Json -Depth 6
$jobId = $retrain.job_id
Write-Host "`n>> job_id = $jobId`n"

# === 6. STATUS (poll) ======================================================
Hr; Write-Host "6) GET /status/$jobId (poll until done/failed)"; Hr
for ($i = 1; $i -le 120; $i++) {
    $st = Invoke-RestMethod "$Base/status/$jobId"
    Write-Host ("   [{0}] status={1} progress={2}%" -f $i, $st.status, $st.progress)
    if ($st.status -in @("done", "failed")) {
        Write-Host "`nFinal job state:"; $st | ConvertTo-Json -Depth 8; break
    }
    Start-Sleep -Seconds 3
}

Write-Host ""
Hr; Write-Host "7) GET /metrics (after retrain)"; Hr
Invoke-RestMethod "$Base/metrics" | ConvertTo-Json -Depth 6

Remove-Item -Recurse -Force $tmp
Write-Host "`nWalkthrough complete."
