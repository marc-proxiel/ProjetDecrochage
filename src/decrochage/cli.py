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
    entrainer_et_evaluer,
    fusionner_catalogue,
    predict_proba,
    save_model,
    select_features,
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
    """Entraine le modele retenu (regression logistique) et sauvegarde ses metadonnees."""
    gold_path = gold_dir / "gold-dataset.csv"
    if not gold_path.exists():
        raise typer.BadParameter(
            f"Gold introuvable : {gold_path}. Lance `decrochage build-gold` d'abord."
        )
    df = pd.read_csv(gold_path, sep=";")

    model, imputer, metrics, feature_columns = entrainer_et_evaluer(
        df,
        target_col=settings.target_col,
        test_size=test_size,
        threshold=settings.decision_threshold,
        random_state=seed,
    )
    typer.echo(f"metrics (test) : {metrics}")

    model_path = model_dir / "logistic.joblib"
    save_model(model, model_path)
    joblib.dump(imputer, model_dir / "imputer.joblib")

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "random_seed": seed,
        "target_col": settings.target_col,
        "features": feature_columns,
        "metrics_holdout": metrics,
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    typer.echo(f"modele -> {model_path}")


@app.command()
def predict(features_csv: Path) -> None:
    """Score chaque ligne d'un CSV de features (meme colonnes que le gold, sans la cible)."""
    from decrochage.models.tabular import load_model

    model_path = settings.model_dir / "logistic.joblib"
    imputer_path = settings.model_dir / "imputer.joblib"
    if not model_path.exists():
        raise typer.BadParameter(f"Model not found: {model_path}. Run `decrochage train` first.")
    model = load_model(model_path)

    df = pd.read_csv(features_csv, sep=";")
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
        df[COLONNES_A_IMPUTER] = imputer.transform(df[COLONNES_A_IMPUTER])
    X = select_features(df, settings.target_col)
    proba = predict_proba(model, X)
    for i, p in enumerate(proba):
        decision = "a_risque" if p >= settings.decision_threshold else "ok"
        typer.echo(f"ligne {i} : proba_abandon={p:.3f} -> {decision}")


def main() -> None:
    app()


# Si on execute ce fichier directement (`python -m decrochage.cli` ou python cli.py),
# on lance aussi la CLI (pratique pour deboguer sans passer par l'entry point installe).
if __name__ == "__main__":
    main()
