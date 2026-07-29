# ---------------------------------------------------------------------------
# One-command local demo for Windows (PowerShell).
#
#   .\run_demo.ps1           # 1 API replica + nginx + UI
#   .\run_demo.ps1 -Scale 4  # 4 API replicas behind the nginx load balancer
#
# API / Swagger : http://localhost:8000/docs
# Streamlit UI  : http://localhost:8501
# ---------------------------------------------------------------------------
param([int]$Scale = 1)

$ErrorActionPreference = "Stop"

# Prefer "docker compose" (v2); fall back to "docker-compose".
docker compose version *> $null
if ($LASTEXITCODE -eq 0) { $DC = @("docker", "compose") }
elseif (Get-Command docker-compose -ErrorAction SilentlyContinue) { $DC = @("docker-compose") }
else { Write-Error "Docker Compose not found. Install Docker Desktop first."; exit 1 }

Write-Host ">> Building and starting stack with api scaled to $Scale ..."
& $DC[0] $DC[1..($DC.Count-1)] up --build -d --scale "api=$Scale"

Write-Host "`n>> Waiting for the API to become healthy ..."
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 | Out-Null
        $ok = $true; break
    } catch { Start-Sleep -Seconds 3 }
}
if ($ok) { Write-Host ">> API is up." }
else { Write-Warning "API did not report healthy within ~3 min. Check: docker compose logs api" }

Write-Host "`n============================================================"
Write-Host "  API   (Swagger)   ->  http://localhost:8000/docs"
Write-Host "  UI    (Streamlit) ->  http://localhost:8501"
Write-Host "  API replicas       :  $Scale"
Write-Host "------------------------------------------------------------"
Write-Host "  Logs : docker compose logs -f"
Write-Host "  Stop : docker compose down"
Write-Host "  Test : .\test_api.ps1"
Write-Host "============================================================"
