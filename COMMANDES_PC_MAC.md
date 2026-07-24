# Commandes de deploiement et de test environnement

Ce fichier sert de check-list formateur et apprenant.

## 1. Preflight Windows PowerShell

```powershell
python --version
py -0p
uv --version
git --version
wsl -l -v
docker --version
```

Notes :

- si `python --version` affiche 3.14, ce n'est pas bloquant ;
- ce qui compte est `uv run python --version` dans le projet ;
- Docker peut etre absent le J1, mais il doit etre pret avant le J3.

Installation minimale si besoin :

```powershell
winget install --id Astral.Uv -e
winget install --id Git.Git -e
winget install --id Microsoft.VisualStudioCode -e
uv python install 3.13
```

Verifier Python 3.13 :

```powershell
uv python list
uv venv --python 3.13
uv run python --version
```

## 2. Preflight macOS Terminal

```bash
python3 --version
uv --version
git --version
docker --version
```

Installation minimale si besoin :

```bash
brew install uv git
uv python install 3.13
```

Si Homebrew n'est pas installe :

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 3. Installation du package apprenant

### Windows

```powershell
cd "C:\chemin\vers\decrochage-sprint3-starter"
uv venv --python 3.13
uv sync --extra dev
uv run python --version
```

### macOS

```bash
cd /chemin/vers/decrochage-sprint3-starter
uv venv --python 3.13
uv sync --extra dev
uv run python --version
```

Attendu : Python 3.13.x.

## 4. Tests de validation du poste

```bash
uv run python -c "import decrochage; print(decrochage.__file__)"
uv run pytest -q
uv run ruff check .
uv run black --check .
uv run decrochage --help
uv run decrochage check-data
uv run decrochage build-gold
```

Attendu :

- import du package OK ;
- tests verts ;
- ruff propre ;
- black propre ;
- CLI repond ;
- donnees sample chargees.
- dataset gold regenerable.

## 5. Commandes Sprint 3 J1

Matin module 23 :

```bash
uv sync --extra dev
uv run pytest tests/test_temporal.py -q
uv run pytest tests/test_loaders.py -q
uv run ruff check .
uv run decrochage --help
```

Apres-midi module 24 :

```bash
uv run pre-commit install
uv run pre-commit run --all-files
uv add dvc mlflow
uv lock
```

## 6. Si uv n'est pas trouve

Windows :

```powershell
winget install --id Astral.Uv -e
```

macOS :

```bash
brew install uv
```

Fermer et rouvrir le terminal apres installation.
