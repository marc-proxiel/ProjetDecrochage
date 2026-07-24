# Commandes J4 — Prefect : idempotence & dashboard (modules 29-30)

Toutes les commandes se lancent **depuis la racine du repo** dans un terminal
VS Code (PowerShell sous Windows, zsh/bash sous macOS). Elles sont identiques
sur les deux OS grâce à `uv`.

## 0. Préflight (5 min, début de journée)

```bash
git pull                        # récupérer la dernière version du starter
uv sync --frozen                # installe l'env exact du lock (Prefect 3.7.6, Python 3.13)
uv run python -c "import prefect; print(prefect.__version__)"   # -> 3.7.6
```

## 1. Démo idempotence (script filet de sécurité, module 30)

Le flow `ingest -> predict -> store` écrit dans `predictions.db` (SQLite local,
aucun Docker requis) avec un **UPSERT** sur la clé `(machine, prediction_ts)`.

```bash
# 1) Base vide -> 1er run : 12 lignes créées (4 machines x 3 moments)
uv run python scripts/demo_prefect_idempotence.py --reset

# 2) REJOUER à l'identique -> "IDEMPOTENT (0 nouvelle ligne)" : aucun doublon
uv run python scripts/demo_prefect_idempotence.py

# 3) AJOUTER des données : +1 moment/machine -> exactement 4 nouvelles lignes (16 au total)
uv run python scripts/demo_prefect_idempotence.py --new 1

# 4) Rejouer avec le même --new 1 -> encore idempotent (16 -> 16)
uv run python scripts/demo_prefect_idempotence.py --new 1

# 5) Encore un cran : --new 2 -> +4 lignes (20 au total)
uv run python scripts/demo_prefect_idempotence.py --new 2

# 6) Panne transitoire simulée -> voir Prefect RÉESSAYER (retries=2) puis réussir
uv run python scripts/demo_prefect_idempotence.py --flaky

# Utilitaires
uv run python scripts/demo_prefect_idempotence.py --show    # affiche la base et quitte
uv run python scripts/demo_prefect_idempotence.py --html    # ouvre predictions.html (auto-refresh 3 s)
```

Astuce démo : ouvre `predictions.html` dans le navigateur **avant** de lancer
les runs — la page se rafraîchit toute seule, on voit le tableau évoluer.

Vérification SQL « aucun doublon » (exercice du module 30) :

```bash
uv run python -c "import sqlite3; print(sqlite3.connect('predictions.db').execute('SELECT machine, prediction_ts, COUNT(*) AS c FROM predictions GROUP BY machine, prediction_ts HAVING c > 1').fetchall())"
# -> []  (liste vide = zéro doublon)
```

## 2. Dashboard / back office Prefect (UI locale, pas de Cloud)

**Terminal 1** (laisser tourner toute la séance) :

```bash
uv run prefect server start
# UI : http://127.0.0.1:4200
```

**Terminal 2** (nouveau terminal VS Code : bouton « + » ou Ctrl+Shift+ù) :

```bash
# pointer les runs vers le serveur local (1 seule fois, persiste dans le profil)
uv run prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

# lancer des flows -> ils apparaissent EN DIRECT dans l'UI
uv run python scripts/demo_prefect_idempotence.py --reset
uv run python scripts/demo_prefect_idempotence.py --new 1
uv run python scripts/demo_prefect_idempotence.py --flaky   # le retry est visible dans l'UI
```

Dans l'UI (http://127.0.0.1:4200) : **Runs** = historique + états ;
cliquer un run = graphe des tâches `ingest -> predict_batch -> store`, logs,
et la tentative échouée + retry pour le run `--flaky`.

Notes :

- Sans serveur démarré, les runs passent par un serveur temporaire mais sont
  quand même historisés dans `~/.prefect/prefect.db` : ils apparaissent dans
  l'UI au prochain `prefect server start`.
- Fin de séance / retour au mode autonome :

```bash
uv run prefect config unset PREFECT_API_URL
```

## 3. Dépannage express

| Symptôme | Remède |
|---|---|
| `Port 4200 already in use` | un serveur tourne déjà -> réutiliser, ou `uv run prefect server stop` |
| Les runs n'apparaissent pas dans l'UI | vérifier `uv run prefect config view` -> `PREFECT_API_URL` doit valoir `http://127.0.0.1:4200/api` |
| Erreur de connexion API au lancement d'un flow | le serveur du Terminal 1 n'est plus lancé -> le relancer, ou `unset PREFECT_API_URL` |
| Base de démo incohérente | `uv run python scripts/demo_prefect_idempotence.py --reset` |
| `git` bloqué sur `index.lock` | fermer les opérations git en cours puis supprimer `.git/index.lock` |

> Rappel périmètre : l'UI Prefect (4200) est indépendante de la stack Docker
> J3 (API 8000 / Prometheus 9090 / Grafana 3000). Pas besoin de Docker pour J4.
