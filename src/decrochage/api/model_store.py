# =============================================================================
#  src/decrochage/api/model_store.py  —  STOCKAGE du modele de ML en memoire
# -----------------------------------------------------------------------------
#  Charge le modele (+ imputer + catalogue des formations + metadonnees) UNE
#  SEULE FOIS au demarrage de l'API et les garde en memoire pour que chaque
#  prediction soit rapide (pas de relecture disque a chaque requete).
#
#  NOTE : ce fichier remplace une version anterieure du starter (maintenance
#  predictive industrielle, modele `rf.joblib`) qui ne correspondait plus au
#  cas d'usage actuel du projet.
# =============================================================================
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from decrochage.models.tabular import load_model


@dataclass
class ModelBundle:
    model: Any  # le pipeline scikit-learn entraine (StandardScaler + LogisticRegression)
    regressor: Any | None  # GradientBoostingRegressor pour moyenne_finale (optionnel)
    imputer: Any  # SimpleImputer fit sur le train (cf. chapitre imputation du notebook)
    catalogue: pd.DataFrame  # table filiere -> faculte/capacite_accueil/... (jointure)
    feature_columns: list[str]  # colonnes EXACTES vues a l'entrainement, dans l'ordre
    version: str
    threshold: float
    target_col: str


_BUNDLE: ModelBundle | None = None


def load_bundle(model_dir: Path, data_dir: Path, threshold: float) -> ModelBundle:
    """Lit le modele, l'imputer, le catalogue et les metadonnees depuis le disque."""
    meta = json.loads((model_dir / "model_metadata.json").read_text())
    catalogue = pd.read_csv(data_dir / "dataset catalogue_formations_V5.csv")

    regressor_path = model_dir / "regressor.joblib"
    regressor = load_model(regressor_path) if regressor_path.exists() else None

    return ModelBundle(
        model=load_model(model_dir / "logistic.joblib"),
        regressor=regressor,
        imputer=joblib.load(model_dir / "imputer.joblib"),
        catalogue=catalogue,
        feature_columns=list(meta["features"]),
        version=str(meta.get("package_version", "0")),
        threshold=threshold,
        target_col=str(meta.get("target_col", "abandon")),
    )


def get_model_bundle() -> ModelBundle | None:
    """Injecte le bundle courant dans les routes via `Depends(get_model_bundle)`."""
    return _BUNDLE
