# =============================================================================
#  src/decrochage/__init__.py
# -----------------------------------------------------------------------------
#  La SEULE présence de ce fichier transforme le dossier `decrochage/` en
#  « package » Python : on pourra écrire `import decrochage` ou
#  `from decrochage.models.tabular import train_model`.
#
#  On y déclare aussi le NUMÉRO DE VERSION du package, à un seul endroit
#  (principe « source unique de vérité ») : il est réutilisé par la CLI
#  (métadonnées du modèle) et vérifié par le test `tests/test_package.py`.
# =============================================================================

"""Decrochage Sprint 3 starter package."""  # docstring = description courte du package

# Version du package. On la centralise ici pour ne pas la dupliquer.
# Convention SemVer : MAJEUR.MINEUR.CORRECTIF (ici 0.1.0 = première ébauche).
__version__ = "0.1.0"
