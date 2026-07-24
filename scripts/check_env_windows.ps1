# =============================================================================
# scripts/check_env_windows.ps1 — Verification de l'environnement (Windows)
# -----------------------------------------------------------------------------
# ROLE : equivalent Windows du script de verification. Ce script PowerShell
# affiche les versions des outils systeme, puis prepare l'environnement du
# projet et lance les premieres verifications (tests, lint). A executer une
# fois au demarrage pour valider son installation sous Windows.
#
# Lancement (dans un terminal PowerShell) :
#   powershell -ExecutionPolicy Bypass -File scripts\check_env_windows.ps1
# Format du fichier : script PowerShell (commentaires avec # en debut de ligne).
# Note : `n dans les chaines PowerShell represente un saut de ligne.
# =============================================================================

# Politique d'erreur : "Continue" => le script ne s'arrete PAS au premier echec
# (utile ici pour afficher TOUTES les versions, meme si un outil manque).
$ErrorActionPreference = "Continue"

# --- Verification des outils SYSTEME (versions installees) ------------------
Write-Host "== System tools =="
# Version de Python.
python --version
# Liste tous les Python installes et leurs chemins (option -0p du lanceur "py").
py -0p
# Version de uv (gestionnaire de paquets/venv).
uv --version
# Version de git.
git --version
# Liste les distributions WSL installees et leur version (1 ou 2).
wsl -l -v
# Version de Docker (conteneurs).
docker --version

# --- Verifications PROPRES AU PROJET ----------------------------------------
# (`n produit une ligne vide avant le titre, pour la lisibilite.)
Write-Host "`n== Project checks =="
# Cree un environnement virtuel base sur Python 3.13.
uv venv --python 3.13
# Installe les dependances du projet + le groupe "dev".
uv sync --extra dev
# Affiche la version de Python utilisee DANS l'environnement (controle).
uv run python --version
# Verifie que le paquet "decrochage" s'importe et affiche son emplacement.
uv run python -c "import decrochage; print(decrochage.__file__)"
# Lance la suite de tests (-q = sortie concise).
uv run pytest -q
# Verifie la qualite du code avec ruff.
uv run ruff check .
