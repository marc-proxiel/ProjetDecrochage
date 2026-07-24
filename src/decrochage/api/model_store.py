# =============================================================================
#  src/decrochage/api/model_store.py  —  STOCKAGE du modèle de ML en mémoire
# -----------------------------------------------------------------------------
#  Place dans le projet : Sprint 3, module API (n°25).
#
#  RÔLE DU FICHIER
#  Charger le modèle de Machine Learning UNE SEULE FOIS (au démarrage de l'API)
#  et le garder en mémoire pour que chaque prédiction soit RAPIDE.
#
#  POURQUOI « UNE SEULE FOIS » ?
#  Lire un modèle entraîné depuis le disque (le fichier `rf.joblib`) est une
#  opération LENTE (plusieurs dizaines de Mo à désérialiser). Si on le rechargeait
#  à chaque requête /predict, l'API serait inutilisable. On le charge donc au
#  démarrage, on le range dans une variable globale, et toutes les requêtes
#  réutilisent ce même objet déjà prêt. C'est un patron classique en MLOps :
#  séparer le COÛT du chargement (payé une fois) du COÛT de l'inférence.
#
#  NOTION DE « BUNDLE »
#  Un modèle seul ne suffit pas pour décider : il faut aussi savoir sa VERSION,
#  le SEUIL de décision à appliquer, et le nom de la colonne CIBLE. On regroupe
#  donc tout cela dans un petit objet appelé « bundle » (= paquet). Ainsi, une
#  prédiction dispose en un seul endroit de tout ce dont elle a besoin.
#
#  INJECTION DE DÉPENDANCES
#  La fonction `get_model_bundle()` est faite pour être branchée dans FastAPI
#  via `Depends(get_model_bundle)` (voir main.py). FastAPI l'appellera tout seul
#  et « injectera » le résultat dans les routes qui en ont besoin. Ce mécanisme
#  (« injection de dépendances ») évite que chaque route aille chercher le modèle
#  elle-même : elle le reçoit en argument, proprement.
# =============================================================================

# Annotations de type modernes (voir explication dans les autres fichiers).
from __future__ import annotations

# `json` : module standard pour lire/écrire du JSON. Ici on l'utilise pour lire
# le fichier de métadonnées du modèle (`model_metadata.json`).
import json

# `dataclass` : un décorateur qui transforme une classe en simple « structure de
# données ». Il génère automatiquement le constructeur `__init__`, l'affichage,
# l'égalité, etc. Idéal pour regrouper quelques champs sans écrire de code répétitif.
from dataclasses import dataclass

# `Path` : représentation portable d'un chemin de fichier (Windows/macOS/Linux).
# Permet d'écrire `model_dir / "rf.joblib"` au lieu de bricoler des chaînes.
from pathlib import Path

# `Any` : type « n'importe quoi ». On l'emploie pour le modèle car son type exact
# (un RandomForest scikit-learn) n'a pas besoin d'être connu/contraint ici ; ce
# qui compte, c'est qu'il sache faire `.predict_proba(...)`.
from typing import Any

# On importe la fonction `load_model` du module « tabular » du projet. C'est elle
# qui sait désérialiser le fichier `rf.joblib` (via joblib) et reconstruire
# l'objet modèle utilisable. On réutilise cette fonction plutôt que de recoder
# la lecture ici : chaque responsabilité reste à sa place.
from decrochage.models.tabular import load_model


# -----------------------------------------------------------------------------
#  La structure « ModelBundle » : le modèle + ses informations indispensables.
# -----------------------------------------------------------------------------
@dataclass
class ModelBundle:
    # `@dataclass` ci-dessus génère automatiquement le constructeur. On déclare
    # juste les CHAMPS (nom + type) ; on pourra ensuite créer un bundle avec
    # ModelBundle(model=..., version=..., threshold=..., target_col=...).

    # `model` : l'objet modèle entraîné lui-même (le RandomForest). Type `Any`
    # car on ne veut pas se contraindre ici (cf. explication de l'import `Any`).
    model: Any

    # `version` : la version du modèle (texte), tirée des métadonnées. Sert à la
    # TRAÇABILITÉ : on saura toujours quel modèle a produit telle prédiction.
    version: str

    # `threshold` : le SEUIL de décision (nombre entre 0 et 1). Si la probabilité
    # de panne dépasse ce seuil -> décision "alerte", sinon "ok". On le stocke
    # avec le modèle pour que la règle de décision soit cohérente et centralisée.
    threshold: float

    # `target_col` : le nom de la colonne CIBLE (ce que le modèle prédit, ex.
    # "panne"). Utile au moment de préparer les features : on doit retirer cette
    # colonne des entrées du modèle (on ne donne pas la réponse en question !).
    target_col: str


# -----------------------------------------------------------------------------
#  L'état global : LE bundle chargé, partagé par toute l'application.
# -----------------------------------------------------------------------------
# `_BUNDLE` contiendra le modèle une fois chargé. Au démarrage du module, il vaut
# `None` (« rien pour l'instant ») : le modèle n'est pas encore chargé. C'est le
# fichier main.py qui, au démarrage de l'API (via `lifespan`), appellera
# `load_bundle(...)` et rangera le résultat dans `store._BUNDLE`.
#   - Type `ModelBundle | None` : soit un bundle, soit rien.
#   - Le préfixe « _ » indique un détail interne au module.
#   - Tant que `_BUNDLE` vaut None, la route /ready renverra 503 (pas prêt) : on
#     refuse de prédire tant que le modèle n'est pas réellement disponible.
_BUNDLE: ModelBundle | None = None


# -----------------------------------------------------------------------------
#  Fonction de CHARGEMENT : lit le modèle + ses métadonnées depuis le disque.
# -----------------------------------------------------------------------------
def load_bundle(model_dir: Path, threshold: float) -> ModelBundle:
    # Paramètres :
    #   - `model_dir` : le DOSSIER qui contient les fichiers du modèle.
    #   - `threshold` : le seuil de décision à mémoriser dans le bundle (il vient
    #     de la configuration du projet, pas du fichier modèle).
    # Renvoie : un `ModelBundle` tout prêt à l'emploi.

    # On lit le fichier de MÉTADONNÉES JSON situé dans le dossier du modèle.
    #   - `model_dir / "model_metadata.json"` construit le chemin complet.
    #   - `.read_text()` lit tout le fichier sous forme de texte.
    #   - `json.loads(...)` transforme ce texte JSON en dictionnaire Python.
    # Ce fichier contient des infos écrites au moment de l'entraînement
    # (version du package, nom de la colonne cible, etc.).
    meta = json.loads((model_dir / "model_metadata.json").read_text())

    # On construit et on RENVOIE le bundle, en remplissant ses quatre champs :
    return ModelBundle(
        # On charge le modèle entraîné depuis `rf.joblib` (rf = random forest).
        # `load_model` gère la désérialisation ; on lui passe le chemin du fichier.
        model=load_model(model_dir / "rf.joblib"),
        # Version du modèle, lue dans les métadonnées. `meta.get("clé", défaut)`
        # renvoie la valeur si la clé existe, sinon la valeur par défaut ("0").
        # On entoure de `str(...)` pour être SÛR d'avoir bien du texte.
        version=str(meta.get("package_version", "0")),
        # Seuil de décision : on prend simplement celui reçu en paramètre.
        threshold=threshold,
        # Nom de la colonne cible : lu dans les métadonnées, défaut "panne".
        target_col=str(meta.get("target_col", "panne")),
    )


# -----------------------------------------------------------------------------
#  Accès au bundle : la fonction « injectée » dans les routes via Depends.
# -----------------------------------------------------------------------------
def get_model_bundle() -> ModelBundle | None:
    # Cette fonction se contente de RENVOYER l'état global `_BUNDLE`.
    #   - Si le modèle est chargé -> elle renvoie le bundle.
    #   - Sinon -> elle renvoie None, et les routes sauront répondre « pas prêt »
    #     (code 503) plutôt que de planter.
    # C'est volontairement une fonction (et non un simple accès à la variable) :
    # cela permet à FastAPI de l'appeler via `Depends(get_model_bundle)` et
    # d'« injecter » proprement le bundle dans chaque route qui le demande.
    # Astuce : on relit `_BUNDLE` à chaque appel (et non une copie capturée),
    # donc dès que main.py l'a rempli au démarrage, toutes les routes le voient.
    return _BUNDLE
