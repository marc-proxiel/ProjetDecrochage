# =============================================================================
#  src/decrochage/api/__init__.py  —  marqueur de SOUS-PACKAGE « api »
# -----------------------------------------------------------------------------
#  Place dans le projet : Sprint 3, module API (n°25) + sécurité (n°26).
#
#  À QUOI SERT CE FICHIER ?
#  La SEULE présence d'un fichier nommé `__init__.py` transforme le dossier
#  `api/` en « package » (= boîte) Python. Sans lui, Python ne saurait pas que
#  `api` est un module importable, et les lignes comme
#      from decrochage.api.main import app
#      from decrochage.api.schemas import PredictionResponse
#  échoueraient avec une erreur « ModuleNotFoundError ».
#
#  POURQUOI EST-IL VIDE ?
#  Un `__init__.py` n'a AUCUNE obligation de contenir du code. Ici on n'a rien
#  de particulier à exécuter au moment où le package est importé (pas de version
#  à déclarer, pas de raccourci d'import à exposer). On le laisse donc vide :
#  son simple rôle est d'exister pour « activer » le package. C'est un usage
#  tout à fait normal et courant en Python.
#
#  ORGANISATION DU SOUS-PACKAGE `api/` (les 4 autres fichiers) :
#    - schemas.py     : le « contrat » des données entrantes/sortantes (Pydantic).
#    - security.py    : les garde-fous (taille du corps de requête, anti-flood).
#    - model_store.py : le chargement et le stockage du modèle de ML en mémoire.
#    - main.py        : l'application FastAPI (les routes /health, /predict-...).
# =============================================================================

# (Intentionnellement vide : voir le bloc d'explication ci-dessus.)
