# =============================================================================
#  run_j3_stack.ps1 — démarre et TESTE toute la stack Docker du J3
#  (api + PostgreSQL + Prometheus + Grafana), pour Windows PowerShell.
#  Usage (PowerShell) :  Unblock-File .\run_j3_stack.ps1
#                        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#                        .\run_j3_stack.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot   # se placer dans le dossier du dépôt

# 1) Fichier .env (identifiants de DEV) : on le crée s'il manque.
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "[.env créé depuis .env.example]" }

# 2) Docker est-il démarré ?
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker n'est pas démarré. Lance Docker Desktop, attends la baleine, puis relance."
    exit 1
}

# 3) Construire les images + lancer les 4 services en arrière-plan.
Write-Host "== docker compose up --build ==" -ForegroundColor Cyan
docker compose up -d --build

# 4) Attendre que l'API réponde sur /health (jusqu'à ~2 min).
Write-Host "== attente de l'API (/health) ==" -ForegroundColor Cyan
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod http://localhost:8000/health -TimeoutSec 2 | Out-Null; $ok = $true; break }
    catch { Start-Sleep -Seconds 2 }
}
if (-not $ok) { docker compose logs --tail=40 api; Write-Error "API indisponible (timeout)."; exit 1 }
Write-Host "API OK" -ForegroundColor Green

# 5) Test réel d'une prédiction (avec la clé d'API et le payload d'exemple).
Write-Host "== test POST /predict-tabular ==" -ForegroundColor Cyan
Invoke-RestMethod -Uri http://localhost:8000/predict-tabular -Method Post `
    -ContentType application/json -Headers @{ "X-API-Key" = "dev-key" } -InFile payload.json

# 6) Récapitulatif des accès.
Write-Host ""
Write-Host "Stack J3 prête :" -ForegroundColor Green
Write-Host "  API (Swagger) : http://localhost:8000/docs"
Write-Host "  Prometheus    : http://localhost:9090"
Write-Host "  Grafana       : http://localhost:3000   (admin / admin - datasource Prometheus deja branchee)"
Write-Host ""
Write-Host "Commandes utiles : docker compose ps | docker compose logs -f | docker compose down (-v pour effacer la base)"
