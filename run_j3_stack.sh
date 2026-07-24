#!/usr/bin/env bash
# =============================================================================
#  run_j3_stack.sh — démarre et TESTE toute la stack Docker du J3
#  (api + PostgreSQL + Prometheus + Grafana), pour macOS / Linux / WSL / Git Bash.
#  Usage :  chmod +x run_j3_stack.sh  &&  ./run_j3_stack.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"   # se placer dans le dossier du dépôt (où vit ce script)

# 1) Fichier .env (identifiants de DEV) : on le crée s'il manque.
if [ ! -f .env ]; then cp .env.example .env; echo "[.env créé depuis .env.example]"; fi

# 2) Docker est-il démarré ?
if ! docker info >/dev/null 2>&1; then
  echo "ERREUR : Docker n'est pas démarré. Lance Docker Desktop, attends la baleine, puis relance." >&2
  exit 1
fi

# 3) Construire les images + lancer les 4 services en arrière-plan (-d).
echo "== docker compose up --build =="
docker compose up -d --build

# 4) Attendre que l'API réponde sur /health (jusqu'à ~2 min).
printf "== attente de l'API (/health) "
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then echo " OK"; break; fi
  printf "."; sleep 2
  if [ "$i" -eq 60 ]; then echo " TIMEOUT"; docker compose logs --tail=40 api; exit 1; fi
done

# 5) Test réel d'une prédiction (avec la clé d'API et le payload d'exemple).
echo "== test POST /predict-tabular =="
curl -fsS -X POST http://localhost:8000/predict-tabular \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  --data @payload.json; echo

# 6) Récapitulatif des accès.
cat <<'URLS'

Stack J3 prête :
  API (Swagger) : http://localhost:8000/docs
  Prometheus    : http://localhost:9090
  Grafana       : http://localhost:3000   (admin / admin — datasource Prometheus déjà branchée)

Commandes utiles :
  docker compose ps        # état des services
  docker compose logs -f   # logs en direct
  docker compose down      # arrêter (ajouter -v pour effacer la base)
URLS
