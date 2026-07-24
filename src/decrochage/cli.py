# =============================================================================
#  src/decrochage/cli.py  —  INTERFACE EN LIGNE DE COMMANDE (CLI)
# -----------------------------------------------------------------------------
#  Donne 4 commandes utilisables au terminal (grâce à la lib `typer`) :
#     decrochage check-data   → vérifie/affiche un résumé santé des données
#     decrochage build-gold   → fabrique et sauvegarde le dataset « gold »
#     decrochage train        → entraîne le modèle + écrit ses métadonnées
#     decrochage predict      → score la dernière mesure de chaque machine
#
#  Le nom `decrochage` est branché sur la fonction `main()` de ce fichier par
#  `pyproject.toml` ([project.scripts] decrochage = "decrochage.cli:main").
#  Ce fichier ne fait que de l'ORCHESTRATION : il appelle les fonctions des
#  autres modules (loaders, temporal, tabular). Aucune logique ML dupliquée ici.
# =============================================================================
from __future__ import annotations  # annotations de type modernes (Path | None)

import json  # pour écrire les métadonnées du modèle en JSON
from datetime import UTC, datetime  # horodatage en temps universel (UTC) — reproductible
from pathlib import Path  # chemins de fichiers

import pandas as pd  # tableaux
import typer  # construit la CLI (commandes, options, --help)
from loguru import logger  # logs lisibles

from decrochage import __version__  # version du package (mise dans les métadonnées)
from decrochage.config import settings  # réglages centraux (chemins, graine, cible…)

# Fonctions de chargement / nettoyage des données :
from decrochage.data.loaders import build_dataset, load_incidents, load_pressure, load_temperature
from decrochage.features.temporal import (
    add_temporal_features,  # fabrication des features lag/rolling
)
from decrochage.models.tabular import (  # utilitaires du modèle
    load_model,
    predict_proba,
    save_model,
    select_features,
    train_model,
)

# Objet « application » Typer : c'est lui qui regroupe toutes les commandes.
app = typer.Typer(help="Decrochage Sprint 3 starter CLI")


def _load_gold(data_dir: Path) -> pd.DataFrame:
    # Fonction PRIVÉE (préfixe `_`) qui rejoue tout le pipeline de données,
    # réutilisée par plusieurs commandes pour éviter de dupliquer le code :
    temp = load_temperature(data_dir / "capteurs_temperature.csv")  # 1) charge la température
    pres = load_pressure(data_dir / "capteurs_pression.tsv")  # 2) charge la pression
    inc = load_incidents(data_dir / "releves_incidents.csv")  # 3) charge les incidents
    # 4) jointure capteurs + cible `panne` (fenêtre d'incident depuis la config)
    dataset = build_dataset(temp, pres, inc, window_hours=settings.incident_window_hours)
    # 5) ajoute les features temporelles, PUIS supprime les lignes à NaN
    #    (les premières lignes de chaque machine, sans passé) et renumérote.
    return add_temporal_features(dataset).dropna().reset_index(drop=True)


@app.command()  # ce décorateur transforme la fonction en commande `decrochage check-data`
def check_data(data_dir: Path | None = None) -> None:
    """Load sample sources and print a short health summary."""
    data_dir = data_dir or settings.data_dir  # si non précisé, prend le dossier par défaut
    dataset = _load_gold(data_dir)  # rejoue le pipeline complet
    typer.echo(f"rows={len(dataset)}")  # nombre de lignes obtenues
    typer.echo(f"machines={dataset['machine'].nunique()}")  # nombre de machines distinctes
    typer.echo(
        f"panne_rate={dataset[settings.target_col].mean():.4f}"
    )  # taux de pannes (moyenne de 0/1)


@app.command()  # → commande `decrochage build-gold`
def build_gold(data_dir: Path | None = None, out: Path | None = None) -> None:
    """Build and save the gold dataset from raw Decrochage sources."""
    data_dir = data_dir or settings.data_dir  # dossier source (défaut: data/raw)
    out = out or settings.gold_dir / "gold_dataset.csv"  # fichier de sortie (défaut: data/gold/…)
    dataset = _load_gold(data_dir)  # construit le dataset « gold »
    out.parent.mkdir(parents=True, exist_ok=True)  # crée le dossier de sortie si besoin
    dataset.to_csv(out, index=False)  # écrit le CSV (sans la colonne d'index)
    logger.info("Gold dataset written: {} rows -> {}", len(dataset), out)  # trace l'opération


@app.command()  # → commande `decrochage train`
def train(data_dir: Path | None = None, out: Path | None = None) -> None:
    """Train the baseline RandomForest model on sample data."""
    data_dir = data_dir or settings.data_dir  # dossier source
    out = out or settings.model_dir / "rf.joblib"  # où sauver le modèle (défaut: artifacts/…)
    dataset = _load_gold(data_dir)  # données prêtes
    x = select_features(dataset, settings.target_col)  # X = features (sans machine/timestamp/cible)
    y = dataset[settings.target_col]  # y = la cible `panne`
    model = train_model(
        x, y, random_state=settings.random_seed
    )  # entraîne (graine = reproductible)
    save_model(model, out)  # sauvegarde le modèle sur disque

    # MÉTADONNÉES = « carte d'identité » du modèle : indispensable pour la traçabilité.
    # On note QUAND, avec QUELLE version, QUELLE graine, QUELLES features, COMBIEN de
    # lignes et QUEL taux de panne le modèle a été entraîné → reproductibilité & audit.
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),  # date/heure UTC de l'entraînement
        "package_version": __version__,  # version du package decrochage
        "random_seed": settings.random_seed,  # graine utilisée
        "target_col": settings.target_col,  # nom de la cible
        "features": list(x.columns),  # liste exacte des features (ordre compris)
        "n_train_rows": int(len(dataset)),  # nombre de lignes d'entraînement
        "panne_rate": round(float(y.mean()), 4),  # proportion de pannes dans les données
        "dataset": str(data_dir),  # d'où venaient les données
    }
    metadata_path = out.parent / "model_metadata.json"  # à côté du modèle
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )  # écrit le JSON lisible
    logger.info("Model trained on {} rows -> {}", len(dataset), out)
    logger.info("Metadata -> {}", metadata_path)


@app.command()  # → commande `decrochage predict`
def predict(data_dir: Path | None = None, model_path: Path | None = None) -> None:
    """Score the latest observation per machine."""
    data_dir = data_dir or settings.data_dir  # dossier source
    model_path = model_path or settings.model_dir / "rf.joblib"  # modèle à charger
    if not model_path.exists():  # garde-fou : modèle absent ?
        # Message d'erreur ACTIONNABLE : on dit quoi faire (lancer `train` d'abord).
        raise typer.BadParameter(f"Model not found: {model_path}. Run `decrochage train` first.")

    # On ne score que la DERNIÈRE mesure de chaque machine (`groupby(machine).tail(1)`)
    # = l'état le plus récent, ce qui intéresse la maintenance « ici et maintenant ».
    dataset = _load_gold(data_dir).groupby("machine").tail(1)
    model = load_model(model_path)  # recharge le modèle entraîné
    scores = predict_proba(model, select_features(dataset, settings.target_col))  # proba de panne
    # Affiche, machine par machine, la probabilité de panne (3 décimales).
    # `strict=False` : tolère que les deux séries n'aient pas exactement la même longueur.
    for machine, score in zip(dataset["machine"], scores, strict=False):
        typer.echo(f"{machine}: P(panne)={score:.3f}")


def main() -> None:
    # Point d'entrée appelé par la commande `decrochage` (cf. pyproject.toml).
    app()


# Si on exécute ce fichier directement (`python -m decrochage.cli` ou python cli.py),
# Python met __name__ == "__main__" → on lance la CLI. (Import simple : ne lance rien.)
if __name__ == "__main__":
    main()
