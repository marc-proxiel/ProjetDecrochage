import numpy as np
import pandas as pd
import pytest

from decrochage.models.tabular import (
    COLONNES_A_IMPUTER,
    aligner_colonnes,
    encoder_categorielles,
    entrainer_classifieur,
    entrainer_regresseur,
    fusionner_catalogue,
    load_model,
    normaliser_bac_type,
    normaliser_mention_bac,
    predict_proba,
    save_model,
    select_features,
    split_et_imputer,
    train_model,
)


def test_select_features_drops_targets_and_date():
    df = pd.DataFrame(
        {
            "abandon": [0, 1],
            "moyenne_finale": [12.0, 8.0],
            "date_inscription": ["a", "b"],
            "age": [20, 21],
        }
    )
    out = select_features(df, "abandon")
    assert list(out.columns) == ["age"]


@pytest.mark.parametrize(
    "brut,attendu",
    [
        ("GEN", "general"),
        ("general", "general"),
        ("Generale", "general"),
        ("TECHNO", "techno"),
        ("Technologique", "techno"),
        ("Professionnel", "pro"),
        (" pro", "pro"),
        ("PRO", "pro"),
    ],
)
def test_normaliser_bac_type(brut, attendu):
    assert normaliser_bac_type(pd.Series([brut])).iloc[0] == attendu


@pytest.mark.parametrize(
    "brut,attendu",
    [
        ("P", "Passable"),
        ("passable", "Passable"),
        ("AB", "Assez Bien"),
        ("Assez Bien", "Assez Bien"),
        ("B", "Bien"),
        ("Bien", "Bien"),
        ("TB", "Tres Bien"),
        ("tres bien", "Tres Bien"),
    ],
)
def test_normaliser_mention_bac(brut, attendu):
    assert normaliser_mention_bac(pd.Series([brut])).iloc[0] == attendu


def test_fusionner_catalogue_recupere_les_colonnes_du_catalogue(catalogue_df):
    df = pd.DataFrame({"filiere": [" informatique ", "DROIT"]})
    fusion = fusionner_catalogue(df, catalogue_df)
    assert "faculte" in fusion.columns
    assert fusion["faculte"].notna().all()
    assert list(fusion["filiere"]) == ["Informatique", "Droit"]


def test_fusionner_catalogue_filiere_inconnue_donne_nan(catalogue_df):
    df = pd.DataFrame({"filiere": ["Philosophie"]})
    fusion = fusionner_catalogue(df, catalogue_df)
    assert fusion["faculte"].isna().all()


def test_encoder_categorielles_one_hot_et_ordinal():
    df = pd.DataFrame(
        {
            "filiere": ["Droit"],
            "bac_type": ["general"],
            "faculte": ["Droit-Science Po"],
            "etablissement_origine": [None],
            "mention_bac": ["Bien"],
        }
    )
    out = encoder_categorielles(df)
    assert out.loc[0, "filiere_Droit"] == 1
    assert out.loc[0, "bac_type_general"] == 1
    assert out.loc[0, "mention_bac"] == 2  # Bien -> 2 (ordinal)
    assert "etablissement_origine_nan" in out.columns


def test_aligner_colonnes_remplit_les_absentes_a_zero():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = aligner_colonnes(df, ["b", "c", "a"])
    assert list(out.columns) == ["b", "c", "a"]
    assert out.loc[0, "c"] == 0


def test_train_model_and_predict_proba_roundtrip():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
    y = (X["x1"] + X["x2"] > 0).astype(int)
    model = train_model(X, y, random_state=42)
    proba = predict_proba(model, X)
    assert proba.shape == (200,)
    assert (proba >= 0).all() and (proba <= 1).all()
    # Signal net -> le modele doit largement battre le hasard.
    assert ((proba >= 0.5).astype(int) == y).mean() > 0.8


def test_save_and_load_model_roundtrip(tmp_path):
    X = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4], "x2": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    model = train_model(X, y, random_state=0)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    reloaded = load_model(path)
    np.testing.assert_allclose(predict_proba(model, X), predict_proba(reloaded, X))


def test_split_et_imputer_no_nan_left(gold_df):
    train_df, test_df, _ = split_et_imputer(gold_df, "abandon", test_size=0.25, random_state=0)
    assert train_df[COLONNES_A_IMPUTER].isna().sum().sum() == 0
    assert test_df[COLONNES_A_IMPUTER].isna().sum().sum() == 0
    assert len(train_df) + len(test_df) == len(gold_df)


def test_entrainer_classifieur_returns_sane_metrics(gold_df):
    train_df, test_df, _ = split_et_imputer(gold_df, "abandon", test_size=0.25, random_state=0)
    _, metrics, feature_columns = entrainer_classifieur(train_df, test_df, "abandon", 0.5, 0)
    assert "abandon" not in feature_columns
    assert "moyenne_finale" not in feature_columns
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_entrainer_regresseur_returns_sane_metrics(gold_df):
    train_df, test_df, _ = split_et_imputer(gold_df, "abandon", test_size=0.25, random_state=0)
    _, metrics, feature_columns = entrainer_regresseur(train_df, test_df, "moyenne_finale", 0)
    assert "moyenne_finale" not in feature_columns
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]  # toujours vrai (inegalite de Cauchy-Schwarz)
