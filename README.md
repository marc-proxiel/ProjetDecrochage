# Decrochage Sprint 3 Starter

Point de depart propre pour demarrer le Sprint 3 CISIA.

Objectif : repartir tous du meme etat, meme si les sprints precedents ont ete
heterogenes. Ce package fournit :

- un vrai package Python en layout `src/` ;
- des donnees brutes Decrochage dans `data/raw/` ;
- un dataset gold dans `data/gold/gold_dataset.csv` ;
- des fonctions de chargement et de normalisation ;
- des features temporelles sans fuite de donnees ;
- une CLI `decrochage` ;
- des tests `pytest` ;
- une configuration `ruff`, `black`, `pre-commit`.

## Verification rapide

### Windows PowerShell

```powershell
cd "C:\chemin\vers\decrochage-sprint3-starter"
uv venv --python 3.13
uv sync --extra dev
uv run python --version
uv run pytest -q
uv run ruff check .
uv run black --check .
uv run decrochage --help
```

La commande `uv run python --version` doit afficher Python 3.13.x. C'est cette
version qui fait foi, pas le `python --version` global de la machine.

## Commandes utiles

```bash
uv run decrochage check-data
uv run decrochage build-gold
uv run decrochage train
uv run decrochage predict
```

## Demarrer l'API 

```bash
uv run uvicorn decrochage.api.main:app --reload
```

Puis ouvrir la doc interactive : http://localhost:8000/docs

- `GET /health` : 200 (le serveur est vivant) ;
- `GET /ready`  : 200 si le modele est charge, sinon 503 ;
- `POST /predict-tabular` : probabilite de decrochage d'un etudiant (cle API requise).

Exemple d'appel (la cle de demonstration est `dev-key`) :

```bash
curl -X POST http://localhost:8000/predict-tabular \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  --data @payload.json
```

> Astuce Windows : depuis **PowerShell**, utilise `curl.exe` (pas l'alias `curl`) et remplace l'antislash de fin de ligne par un accent grave ; ou lance la commande depuis **Git Bash / WSL**.

### Appeler l'API depuis Windows PowerShell

L'API doit tourner dans une AUTRE fenetre (commande `uvicorn` ci-dessus). On n'ecrit PAS le JSON a la main : on envoie le fichier `payload.json` fourni.

```powershell
# Option 1 — Invoke-RestMethod (natif PowerShell)
Invoke-RestMethod -Uri http://localhost:8000/predict-tabular -Method Post `
  -ContentType application/json -Headers @{ "X-API-Key" = "dev-key" } -InFile payload.json

# Option 2 — curl.exe (le vrai curl, pas l'alias PowerShell), en UNE seule ligne
curl.exe -X POST http://localhost:8000/predict-tabular -H "X-API-Key: dev-key" -H "Content-Type: application/json" --data "@payload.json"
```

Le plus simple, sans aucune syntaxe : ouvrir http://localhost:8000/docs , deplier `POST /predict-tabular`, cliquer **Try it out**, renseigner `X-API-Key = dev-key`, coller le contenu de `payload.json` et **Execute**.

Reponse attendue : `{"proba_abandon":0.0556,"decision":"ok","model_version":"0.1.0","threshold":0.5}`

Codes attendus : sans cle -> 401, corps trop gros -> 413, trop de requetes -> 429, donnees invalides -> 422.

## Demarrer toute la stack avec Docker 

Prerequis : Docker installe (scripts `install_docker_windows.ps1` / `install_docker_macos.sh`) et Docker Desktop demarre.

Le plus simple — un seul script qui construit, lance et TESTE les 4 services (api + PostgreSQL + Prometheus + Grafana) :

```bash
# macOS / Linux / WSL / Git Bash
chmod +x run_j3_stack.sh && ./run_j3_stack.sh
```
```powershell
# Windows PowerShell
.\run_j3_stack.ps1
```

A la main, etape par etape :

```bash
cp .env.example .env          # identifiants de DEV (cle d'API + mot de passe Postgres)
docker compose up -d --build  # construit l'image et lance les 4 services
docker compose ps             # verifier l'etat (db "healthy")
```

Acces une fois lance : API http://localhost:8000/docs - Prometheus http://localhost:9090 - Grafana http://localhost:3000 (admin / admin, datasource deja branchee).
Arret : `docker compose down` (ajouter `-v` pour effacer aussi la base). Raccourcis : `make up`, `make ps`, `make logs`, `make down`, `make smoke`.

## Structure

```text
src/decrochage/
  config.py
  cli.py
  models/tabular.py
  api/              # Sprint 3 / J2 : l'API FastAPI
    main.py         #   routes : /health /ready /predict-tabular /predict-image /metrics
    schemas.py      #   contrat d'entree-sortie (Pydantic)
    security.py     #   cle API (401), anti-flood (429), taille max (413)
    model_store.py  #   chargement du modele au demarrage
tests/              # dont test_api.py et test_security.py (J2)
Dockerfile          # J3 : image multi-stage, non-root
docker-compose.yml  # J3 : api + PostgreSQL + Prometheus + Grafana
monitoring/         # J6 : configuration Prometheus
data/raw/
data/silver/
data/gold/
artifacts/models/
```
