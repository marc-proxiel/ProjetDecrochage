# =============================================================================
#  src/decrochage/models/tabular.py  —  LE MODÈLE (données tabulaires)
# -----------------------------------------------------------------------------
#  Cinq petites fonctions, chacune avec UNE responsabilité (principe « single
#  responsibility ») :
#    • select_features : isoler les colonnes d'entrée X (≠ cible, ≠ identifiants)
#    • train_model     : entraîner un RandomForest
#    • predict_proba   : sortir la PROBABILITÉ de panne (entre 0 et 1)
#    • save_model      : écrire le modèle sur disque (.joblib)
#    • load_model      : relire un modèle sauvegardé
#
#  « Tabulaire » = données en tableau (lignes × colonnes), par opposition aux
#  images ou au texte. Ici on prédit `panne` (0/1) à partir des features capteurs.
# =============================================================================
from __future__ import annotations  # annotations de type modernes

from pathlib import Path  # chemins de fichiers portables
from typing import Any  # « Any » = type volontairement non précisé (un modèle sklearn)

import joblib  # sérialisation efficace d'objets Python (idéale pour les modèles scikit-learn)
import numpy as np  # tableaux numériques (le vecteur de probabilités renvoyé)
import pandas as pd  # DataFrame (les features X)
from sklearn.ensemble import RandomForestClassifier  # l'algorithme : forêt d'arbres de décision

# Colonnes qui NE SONT PAS des features : un identifiant (machine) et une date
# (timestamp) ne doivent jamais entrer dans le modèle comme variables explicatives.
NON_FEATURE_COLUMNS: tuple[str, ...] = ("machine", "timestamp")


def select_features(
    df: pd.DataFrame,  # tableau complet (features + cible + identifiants)
    target_col: str,  # nom de la colonne cible (ex. "panne")
    exclude: tuple[str, ...] = NON_FEATURE_COLUMNS,  # colonnes à retirer en plus de la cible
) -> pd.DataFrame:  # renvoie X = uniquement les features
    # On construit la liste des colonnes à enlever : les identifiants + la cible,
    # mais SEULEMENT si elles existent réellement dans le tableau (`if col in df.columns`).
    cols = [col for col in (*exclude, target_col) if col in df.columns]
    # `drop` renvoie une COPIE sans ces colonnes → il ne reste que les features (X).
    return df.drop(columns=cols)


def train_model(
    x: pd.DataFrame,  # X = les features (une ligne par observation)
    y: pd.Series,  # y = la cible 0/1 alignée sur X
    n_estimators: int = 200,  # nombre d'arbres dans la forêt (plus = plus stable, plus lent)
    random_state: int = 42,  # graine = entraînement REPRODUCTIBLE (mêmes arbres à chaque fois)
) -> RandomForestClassifier:  # renvoie le modèle entraîné
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        # class_weight="balanced" : les pannes sont RARES (~10 %). Sans pondération, le
        # modèle aurait tendance à toujours prédire « pas de panne ». « balanced » donne
        # plus de poids à la classe minoritaire pour qu'elle compte autant que l'autre.
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(x, y)  # ENTRAÎNEMENT : le modèle apprend la relation features → panne
    return model


def predict_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    # predict_proba renvoie 2 colonnes : [proba classe 0, proba classe 1].
    # `[:, 1]` = on garde la 2e colonne = la PROBABILITÉ DE PANNE (classe 1).
    # On préfère une proba (0→1) à une décision brute 0/1 : on choisit le seuil ensuite.
    return model.predict_proba(x)[:, 1]


def save_model(model: Any, path: Path) -> None:
    # Crée le dossier de destination si besoin (`parents=True` = crée toute l'arborescence,
    # `exist_ok=True` = ne plante pas s'il existe déjà).
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)  # écrit le modèle entraîné dans le fichier (ex. rf.joblib)


def load_model(path: Path) -> Any:
    # Relit un modèle déjà entraîné depuis le disque (pour prédire sans ré-entraîner).
    return joblib.load(path)
