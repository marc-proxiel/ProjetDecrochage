# Decrochage — detection precoce du decrochage etudiant

Projet de certification "Concevoir et implementer une solution d'intelligence
artificielle" : detecter, a mi-semestre, les etudiants a risque de decrochage
(`abandon`) et estimer leur note finale (`moyenne_finale`), a partir de donnees
administratives, d'engagement LMS et du catalogue des formations.

Le projet couvre l'ensemble du cycle : preparation des donnees (bronze/silver/gold),
comparaison et choix de modeles (classification + regression), explicabilite (SHAP),
suivi des experimentations (MLflow), orchestration (Prefect), API de scoring
(FastAPI), stockage batch (PostgreSQL) et supervision (Prometheus/Grafana).

- **Notebook d'analyse complet** : [`EtudeDecrochageMVA.ipynb`](EtudeDecrochageMVA.ipynb)
  — EDA, preparation des donnees, choix de modele, SHAP, MLflow, architecture cible,
  conclusions/recommandations.
- **Journal de bord** : [`JournalDeBord-MVA.ipynb`](JournalDeBord-MVA.ipynb).
- **Code source** : https://github.com/marc-proxiel/ProjetDecrochage

## Verification rapide

### Windows PowerShell

```powershell
cd "C:\chemin\vers\ProjetDecrochage"
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

`uv run pytest -q` execute les 49 tests du projet (`tests/`) : package, config,
pipeline tabulaire, CLI, API, securite (cle API/anti-flood) et stockage (SQLite en
memoire, sans PostgreSQL requis).

## Pipeline de donnees et entrainement (CLI)

```bash
uv run decrochage check-data    # charge les 2 CSV bruts, affiche shape + valeurs manquantes
uv run decrochage build-gold    # bronze -> silver -> gold, ecrit data/gold/gold-dataset.csv
uv run decrochage train         # entraine ET sauvegarde les 2 modeles (voir ci-dessous)
uv run decrochage predict FICHIER.csv   # score un CSV (memes colonnes que le gold)
```

`decrochage train` entraine et sauvegarde ensemble, sur le meme split train/test :

- un **classifieur** (regression logistique, `abandon`) -> `artifacts/models/logistic.joblib` ;
- un **regresseur** (Gradient Boosting, `moyenne_finale`) -> `artifacts/models/regressor.joblib` ;
- l'**imputer** entraine sur le train -> `artifacts/models/imputer.joblib` ;
- les metadonnees (metriques, features, seed, version) -> `artifacts/models/model_metadata.json`.

Reglages surchargeables via `.env` (prefixe `DECROCHAGE_`, voir
`src/decrochage/.env.sample`) : `DATA_DIR`, `GOLD_DIR`, `MODEL_DIR`, `RANDOM_SEED`,
`TARGET_COL`, `API_KEY`, `DECISION_THRESHOLD`, `DB_URL`.

## Suivi des experimentations avec MLflow

Chaque run (baseline puis modele tune, pour les 3 familles comparees en
classification et en regression) est journalise dans MLflow local
(`sqlite:///mlflow.db`), sur 2 experiences separees :

- `decrochage-abandon-classification`
- `decrochage-moyenne-finale-regression`

Les 2 modeles retenus sont enregistres au Model Registry
(`decrochage-abandon-logistique`, `decrochage-moyenne-finale-gb`).

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir http://localhost:5000.

## Orchestration avec Prefect

`flows/pipeline.py` enchaine, en un seul flow (`pipeline_decrochage`) : construction
du gold -> entrainement des 2 modeles -> scoring d'un echantillon -> rapport -> (si
`DECROCHAGE_DB_URL` est configure) ecriture des scores en base PostgreSQL.

```bash
uv run prefect server start --host 127.0.0.1 --port 4200   # dans un terminal a part
uv run python flows/pipeline.py
```

Dashboard des runs : http://127.0.0.1:4200. Voir aussi
[`COMMANDES_J4_PREFECT.md`](COMMANDES_J4_PREFECT.md) pour le detail des commandes
(idempotence, deploiement planifie via `.serve()`).

## Demarrer l'API

```bash
uv run uvicorn decrochage.api.main:app --reload
```

Puis ouvrir la doc interactive : http://localhost:8000/docs

- `GET /health` : 200 (le serveur est vivant) ;
- `GET /ready`  : 200 si les modeles sont charges, sinon 503 ;
- `POST /predict-tabular` : proba d'abandon **et** note finale predite pour un
  etudiant (cle API requise) ;
- `POST /predict-image` : stub generique de validation d'image (pas utilise pour le
  decrochage, a brancher plus tard si besoin) ;
- `GET /metrics` : metriques Prometheus (HTTP generiques + metier, voir plus bas).

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

Reponse attendue : `{"proba_abandon":0.0556,"decision":"ok","moyenne_predite":16.85,"model_version":"0.1.0","threshold":0.5}`

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

### Monitoring et alertes (Prometheus / Grafana)

L'API expose des metriques metier en plus des metriques HTTP generiques :
`decrochage_model_loaded`, `decrochage_predictions_total{decision}`,
`decrochage_proba_abandon`, `decrochage_moyenne_predite`.

7 regles d'alerte sont provisionnees automatiquement dans Grafana au demarrage
(`monitoring/grafana/provisioning/alerting/rules.yml`) : API down, modele non
charge, taux d'erreur 5xx eleve, latence p95 elevee, et 3 alertes de derive
(taux d'abandon predit trop haut/trop bas, note predite anormale) comparees a la
baseline observee a l'entrainement.

### Stockage batch des scores (PostgreSQL)

`src/decrochage/storage.py` (SQLAlchemy Core, portable PostgreSQL/SQLite) ecrit les
scores produits par le flow Prefect dans la table `predictions` (toujours en
INSERT, jamais d'ecrasement -> historique). Configuration via `DECROCHAGE_DB_URL`
(pointe par defaut vers le service `db` de `docker-compose.yml`, port 5432 publie
pour un acces depuis l'hote en dev).

## Notebook autonome (sans installation du projet)

`standalone/EtudeDecrochageMVA_autonome.ipynb` est une version du notebook principal
executable seule, sans le package `src/decrochage`, sans Docker, sans Prefect et sans
PostgreSQL. Il suffit d'avoir, dans le meme dossier que ce notebook :

- `dataset decrochage_etudiants_complet_V5.csv`
- `dataset catalogue_formations_V5.csv`

(les deux sont deja fournis dans `standalone/`).

```bash
pip install pandas numpy matplotlib scikit-learn shap mlflow joblib
```

Puis ouvrir le notebook et faire "Executer tout" (Run All). Il cree lui-meme ses
dossiers intermediaires (`data/silver/`, `data/gold/`, `artifacts/models/`) et son
suivi MLflow local (`mlflow.db`) dans son propre dossier.

Differences avec le notebook complet du depot :

- les chapitres API (FastAPI), Docker/docker-compose, alertes Grafana et stockage
  PostgreSQL decrivent l'architecture de production reellement testee dans le depot
  complet, mais leur code est presente en lecture seule (blocs non executes) puisqu'il
  depend de services et d'un package externes a ce notebook ;
- le chapitre "Mesure de performance et impact" reutilise directement les modeles deja
  entraines dans ce notebook, au lieu de recharger des fichiers `.joblib` produits par
  la CLI du projet.

## Documents complementaires

- [`COMMANDES_J4_PREFECT.md`](COMMANDES_J4_PREFECT.md) — commandes Prefect detaillees
  (idempotence, deploiement planifie).
- [`COMMANDES_PC_MAC.md`](COMMANDES_PC_MAC.md) — check-list d'environnement
  Windows/macOS (Docker, WSL, Python, uv).
- [`README_COMMENTAIRES.md`](README_COMMENTAIRES.md) — copie pedagogique commentee
  ligne a ligne d'un repo de reference (support de formation, sans lien fonctionnel
  avec ce depot).

## Structure

```text
src/decrochage/
  config.py         # reglages (pydantic-settings), prefixe DECROCHAGE_
  cli.py             # commandes check-data / build-gold / train / predict
  models/tabular.py  # pipeline de features + entrainement classifieur/regresseur
  storage.py         # ecriture/lecture des scores dans PostgreSQL (SQLAlchemy Core)
  api/
    main.py         # routes : /health /ready /predict-tabular /predict-image /metrics
    schemas.py       # contrat d'entree-sortie (Pydantic), documente pour Swagger
    security.py      # cle API (401), anti-flood (429), taille max (413)
    model_store.py   # chargement des modeles (+ imputer, catalogue) au demarrage
flows/pipeline.py    # orchestration Prefect (gold -> train -> score -> stockage)
tests/               # 49 tests pytest (package, config, tabular, cli, api, security, storage)
monitoring/
  prometheus.yml
  grafana/provisioning/  # datasource + 7 regles d'alerte provisionnees
Dockerfile           # image multi-stage, non-root
docker-compose.yml   # api + PostgreSQL + Prometheus + Grafana
data/raw/            # 2 CSV bruts (etudiants, catalogue des formations)
data/silver/
data/gold/
artifacts/models/    # logistic.joblib, regressor.joblib, imputer.joblib, model_metadata.json
standalone/          # notebook autonome + ses 2 CSV (voir section dediee)
EtudeDecrochageMVA.ipynb   # notebook d'analyse complet (deliverable principal)
JournalDeBord-MVA.ipynb    # journal de bord de la certification
```