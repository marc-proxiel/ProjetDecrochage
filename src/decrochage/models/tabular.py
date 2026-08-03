# =============================================================================
#  src/decrochage/models/tabular.py  —  LE MODELE (donnees tabulaires)
# -----------------------------------------------------------------------------
#  Contexte (Sprint 3 CISIA) : predire le RISQUE DE DECROCHAGE (`abandon`, 0/1)
#  d'un etudiant a mi-semestre, a partir du dataset gold.
#
#  MODELE RETENU : regression logistique (StandardScaler + LogisticRegression),
#  choisie dans EtudeDecrochageMVA.ipynb (section "9.8. Choix du modele") apres
#  comparaison avec Random Forest et Gradient Boosting, validation croisee
#  5-fold et optimisation des hyperparametres (recherche sur ROC-AUC). Meilleur
#  hyperparametre trouve : C=0.01 (forte regularisation, coherente avec des
#  features correlees entre elles). Voir le notebook pour le detail complet de
#  la demarche (comparaison de modeles, SHAP, choix du seuil, suivi MLflow).
#
#  NOTE : ce fichier remplace une version anterieure du starter (maintenance
#  predictive industrielle, cible "panne") qui ne correspondait plus au cas
#  d'usage actuel du projet.
# =============================================================================
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Colonnes a exclure des features : la cible elle-meme, l'AUTRE cible
# (`moyenne_finale` : fuite de donnees si utilisee pour predire `abandon`,
# cf. chapitre 6.5 du notebook), et une colonne texte pas encore transformee
# en feature numerique.
COLONNES_NON_FEATURES = ["abandon", "moyenne_finale", "date_inscription"]

# Colonnes numeriques imputees par mediane (fit sur le train uniquement, cf.
# chapitre "Strategie d'imputation" du notebook) : mention_bac est ordinale
# (0-3) a ce stade, donc traitee comme numerique elle aussi.
COLONNES_A_IMPUTER = [
    "heures_travail_remunere_sem",
    "satisfaction",
    "sentiment_appartenance",
    "distance_domicile_km",
    "motivation",
    "mention_bac",
    "connexions_lms_30j",
]

MAPPING_MENTION = {
    "p": "Passable",
    "passable": "Passable",
    "ab": "Assez Bien",
    "assez bien": "Assez Bien",
    "b": "Bien",
    "bien": "Bien",
    "tb": "Tres Bien",
    "tres bien": "Tres Bien",
}
MAPPING_ORDINAL_MENTION = {"Passable": 0, "Assez Bien": 1, "Bien": 2, "Tres Bien": 3}


def select_features(df: pd.DataFrame, target_col: str = "abandon") -> pd.DataFrame:
    """Isole les colonnes explicatives (retire la cible et COLONNES_NON_FEATURES)."""
    a_exclure = {target_col, *COLONNES_NON_FEATURES}
    return df.drop(columns=[c for c in a_exclure if c in df.columns])


def _normaliser_texte(s: pd.Series) -> pd.Series:
    """Strip + minuscules + suppression des accents (comparaison robuste)."""
    s = s.astype(str).str.strip().str.lower()
    return s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("utf-8")


def normaliser_bac_type(s: pd.Series) -> pd.Series:
    """Reduit les variantes de bac_type (GEN/general/Generale/...) a 3 categories."""
    s = _normaliser_texte(s)
    return (
        s.mask(s.str.startswith("gen"), "general")
        .mask(s.str.startswith("techno"), "techno")
        .mask(s.str.contains("pro"), "pro")
    )


def normaliser_mention_bac(s: pd.Series) -> pd.Series:
    """Reduit les variantes de mention_bac (P/passable/AB/...) aux 4 categories propres."""
    return _normaliser_texte(s).map(MAPPING_MENTION)


def fusionner_catalogue(df: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Jointure sur `filiere` (casse/espaces normalises) pour recuperer faculte,
    ects_semestre, capacite_accueil, volume_horaire_s1, taux_reussite_historique_pct."""
    df = df.copy()
    catalogue = catalogue.copy()
    df["_filiere_norm"] = _normaliser_texte(df["filiere"])
    catalogue["_filiere_norm"] = _normaliser_texte(catalogue["filiere"])
    fusion = df.merge(catalogue.drop(columns="filiere"), on="_filiere_norm", how="left").drop(
        columns="_filiere_norm"
    )
    filiere_canonique = {f.lower(): f for f in catalogue["filiere"]}
    fusion["filiere"] = _normaliser_texte(fusion["filiere"]).map(filiere_canonique)
    return fusion


def encoder_categorielles(df: pd.DataFrame) -> pd.DataFrame:
    """Encode filiere/bac_type/faculte/etablissement_origine (one-hot) et
    mention_bac (ordinal) — memes regles que dans le notebook (chapitre gold)."""
    df = df.copy()
    df["bac_type"] = normaliser_bac_type(df["bac_type"])
    if "mention_bac" in df.columns:
        df["mention_bac"] = normaliser_mention_bac(df["mention_bac"]).map(MAPPING_ORDINAL_MENTION)

    df = pd.get_dummies(df, columns=["filiere", "bac_type", "faculte"])
    df = pd.get_dummies(df, columns=["etablissement_origine"], dummy_na=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def aligner_colonnes(df: pd.DataFrame, colonnes_attendues: list[str]) -> pd.DataFrame:
    """Reindexe vers exactement les colonnes vues a l'entrainement : essentiel pour
    le one-hot encoding, qui ne cree une colonne que pour les categories presentes
    dans les donnees encodees (une seule ligne -> une seule categorie par variable)."""
    return df.reindex(columns=colonnes_attendues, fill_value=0)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    C: float = 0.01,
    random_state: int = 42,
) -> Pipeline:
    """Entraine le modele retenu : StandardScaler + LogisticRegression.

    `C=0.01` est l'hyperparametre retenu par GridSearchCV (recherche sur
    ROC-AUC, validation croisee 5-fold) dans le notebook.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilite d'abandon (classe 1) pour chaque ligne de X."""
    return model.predict_proba(X)[:, 1]


def entrainer_et_evaluer(
    df_gold: pd.DataFrame,
    target_col: str = "abandon",
    test_size: float = 0.2,
    threshold: float = 0.5,
    random_state: int = 42,
) -> tuple[Pipeline, SimpleImputer, dict, list[str]]:
    """Split stratifie, imputation (fit sur le train uniquement), entrainement et
    evaluation sur le test — la demarche retenue dans le notebook (chapitres
    "Split train/test et imputation" puis "9.8. Choix du modele").

    Utilisee a la fois par `decrochage train` (CLI) et par le flow Prefect
    (`flows/pipeline.py`) : une seule version de la logique d'entrainement.
    """
    train_df, test_df = train_test_split(
        df_gold, test_size=test_size, random_state=random_state, stratify=df_gold[target_col]
    )

    imputer = SimpleImputer(strategy="median")
    train_df[COLONNES_A_IMPUTER] = imputer.fit_transform(train_df[COLONNES_A_IMPUTER])
    test_df[COLONNES_A_IMPUTER] = imputer.transform(test_df[COLONNES_A_IMPUTER])

    x_tr = select_features(train_df, target_col)
    y_tr = train_df[target_col]
    x_te = select_features(test_df, target_col)
    y_te = test_df[target_col]

    model = train_model(x_tr, y_tr, random_state=random_state)
    proba = predict_proba(model, x_te)
    preds = (proba >= threshold).astype(int)

    metrics = {
        "precision": round(float(precision_score(y_te, preds)), 4),
        "recall": round(float(recall_score(y_te, preds)), 4),
        "f1": round(float(f1_score(y_te, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "threshold": threshold,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "taux_abandon": round(float(df_gold[target_col].mean()), 4),
    }
    return model, imputer, metrics, list(x_tr.columns)


def save_model(model: Pipeline, path: Path) -> None:
    """Sauvegarde le pipeline entraine (joblib) ; cree le dossier si besoin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    """Recharge un pipeline sauvegarde avec `save_model`."""
    return joblib.load(path)
