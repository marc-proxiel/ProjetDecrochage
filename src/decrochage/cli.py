# =============================================================================
#  src/decrochage/cli.py  —  INTERFACE EN LIGNE DE COMMANDE (CLI)
# -----------------------------------------------------------------------------
#  Commandes :
#     decrochage check-data   -> resume sante des donnees brutes
#     decrochage build-gold   -> fabrique et sauvegarde le dataset gold
#     decrochage train        -> entraine le modele + ecrit ses metadonnees
#     decrochage predict      -> score un CSV de nouvelles observations
#
#  Reproduit en script (donc reproductible/automatisable) la demarche menee
#  dans EtudeDecrochageMVA.ipynb : fusion bronze, nettoyage silver, encodage
#  gold, puis entrainement de la regression logistique retenue (voir
#  decrochage.models.tabular pour le detail du modele).
#
#  NOTE : ce fichier remplace une version anterieure du starter (maintenance
#  predictive industrielle) qui ne correspondait plus au cas d'usage actuel.
# =============================================================================
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import typer

from decrochage import __version__
from decrochage.config import settings
from decrochage.models.tabular import (
    COLONNES_A_IMPUTER,
    encoder_categorielles,
    entrainer_classifieur,
    entrainer_regresseur,
    fusionner_catalogue,
    predict_proba,
    save_model,
    select_features,
    split_et_imputer,
)

app = typer.Typer(help="Decrochage - CLI de detection precoce du decrochage etudiant")

# Colonnes propres au fichier brut (identifiants, variables leurres, fuite
# temporelle) : sans objet pour un appel API a la ligne, donc gardees ici
# plutot que dans decrochage.models.tabular (partage avec l'API).
COLONNES_A_SUPPRIMER = [
    "student_id",
    "id_dossier",
    "annee_universitaire",
    "groupe_td",
    "couleur_carte_etudiante",
    "jour_inscription",
    "moyenne_partiels_s1",
    "nb_ue_total",
    "nb_ue_validees_s1",
    "commentaire_tuteur",
    "sexe",
    "boursier",
    "niveau",
]


def build_gold_dataset(raw_dir: Path) -> pd.DataFrame:
    """Fusion bronze + nettoyage silver + encodage gold (voir le notebook)."""
    df_etudiants = pd.read_csv(raw_dir / "dataset decrochage_etudiants_complet_V5.csv")
    df_catalogue = pd.read_csv(raw_dir / "dataset catalogue_formations_V5.csv")

    df = fusionner_catalogue(df_etudiants, df_catalogue)
    df = df.drop(columns=[c for c in COLONNES_A_SUPPRIMER if c in df.columns])

    df["date_inscription"] = pd.to_datetime(
        df["date_inscription"], format="mixed", dayfirst=True
    ).dt.strftime("%d-%m-%Y")

    for col in ["distance_domicile_km", "taux_presence_pct"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("km", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    df = df.drop_duplicates().reset_index(drop=True)
    df = encoder_categorielles(df)

    if (df["nb_devoirs_rendus"] / df["nb_devoirs_total"]).max() < 1:
        df["taux_rendu"] = df["nb_devoirs_rendus"] / df["nb_devoirs_total"]
        df = df.drop(columns=["nb_devoirs_rendus"])
    df = df.drop(columns=["nb_devoirs_total"])

    return df


@app.command()
def check_data(data_dir: Path = settings.data_dir) -> None:
    """Charge les sources brutes et affiche un resume sante (shape, nulls)."""
    noms = ["dataset decrochage_etudiants_complet_V5.csv", "dataset catalogue_formations_V5.csv"]
    for nom in noms:
        df = pd.read_csv(data_dir / nom)
        typer.echo(f"\n{nom} : {df.shape[0]} lignes x {df.shape[1]} colonnes")
        nulls = df.isna().sum()
        nulls = nulls[nulls > 0]
        if nulls.empty:
            typer.echo("  aucune valeur manquante")
        else:
            typer.echo(f"  valeurs manquantes :\n{nulls.to_string()}")


@app.command()
def build_gold(
    data_dir: Path = settings.data_dir,
    gold_dir: Path = settings.gold_dir,
) -> None:
    """Fabrique et sauvegarde le dataset gold a partir des sources brutes."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    df_gold = build_gold_dataset(data_dir)
    out_path = gold_dir / "gold-dataset.csv"
    df_gold.to_csv(out_path, index=False, sep=";")
    typer.echo(f"gold -> {out_path} ({df_gold.shape[0]} lignes, {df_gold.shape[1]} colonnes)")


@app.command()
def train(
    gold_dir: Path = settings.gold_dir,
    model_dir: Path = settings.model_dir,
    test_size: float = 0.2,
    seed: int = settings.random_seed,
) -> None:
    """Entraine le classifieur (abandon) et le regresseur (moyenne_finale), sauvegarde les deux."""
    gold_path = gold_dir / "gold-dataset.csv"
    if not gold_path.exists():
        raise typer.BadParameter(
            f"Gold introuvable : {gold_path}. Lance `decrochage build-gold` d'abord."
        )
    df = pd.read_csv(gold_path, sep=";")

    # Meme split/imputation partage par les deux modeles (memes lignes train/test).
    train_df, test_df, imputer = split_et_imputer(df, settings.target_col, test_size, seed)

    model, metrics, feature_columns = entrainer_classifieur(
        train_df, test_df, settings.target_col, settings.decision_threshold, seed
    )
    typer.echo(f"classifieur abandon (test) : {metrics}")

    regressor, metrics_reg, _ = entrainer_regresseur(train_df, test_df, "moyenne_finale", seed)
    typer.echo(f"regresseur moyenne_finale (test) : {metrics_reg}")

    save_model(model, model_dir / "logistic.joblib")
    save_model(regressor, model_dir / "regressor.joblib")
    joblib.dump(imputer, model_dir / "imputer.joblib")

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "random_seed": seed,
        "target_col": settings.target_col,
        "features": feature_columns,
        "metrics_holdout": metrics,
        "metrics_holdout_regression": metrics_reg,
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    typer.echo(f"modeles -> {model_dir / 'logistic.joblib'}, {model_dir / 'regressor.joblib'}")


@app.command()
def predict(features_csv: Path) -> None:
    """Score chaque ligne d'un CSV (proba d'abandon + note predite), meme colonnes que le gold."""
    from decrochage.models.tabular import load_model

    model_path = settings.model_dir / "logistic.joblib"
    regressor_path = settings.model_dir / "regressor.joblib"
    imputer_path = settings.model_dir / "imputer.joblib"
    if not model_path.exists():
        raise typer.BadParameter(f"Model not found: {model_path}. Run `decrochage train` first.")
    model = load_model(model_path)
    regressor = load_model(regressor_path) if regressor_path.exists() else None

    df = pd.read_csv(features_csv, sep=";")
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
        df[COLONNES_A_IMPUTER] = imputer.transform(df[COLONNES_A_IMPUTER])
    X = select_features(df, settings.target_col)
    proba = predict_proba(model, X)
    notes = regressor.predict(X) if regressor is not None else [None] * len(proba)
    for i, (p, note) in enumerate(zip(proba, notes, strict=True)):
        decision = "a_risque" if p >= settings.decision_threshold else "ok"
        suffixe = f", moyenne_predite={note:.2f}" if note is not None else ""
        typer.echo(f"ligne {i} : proba_abandon={p:.3f} -> {decision}{suffixe}")


def main() -> None:
    app()


# Si on execute ce fichier directement (`python -m decrochage.cli` ou python cli.py),
# on lance aussi la CLI (pratique pour deboguer sans passer par l'entry point installe).
if __name__ == "__main__":
    main()
