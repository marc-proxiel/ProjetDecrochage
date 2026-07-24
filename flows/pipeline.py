"""Orchestration Prefect du pipeline Decrochage (Sprint 3 — modules B9/B10 industrialisation).

POURQUOI UN ORCHESTRATEUR ?
    Jusqu'ici on lançait le pipeline à la main (`decrochage train`). En production,
    un orchestrateur apporte ce que un script + cron ne donnent pas :
      - observabilité : chaque exécution (= "flow run") est tracée dans une UI,
        avec logs, durées, graphe des étapes ;
      - résilience : retries automatiques sur les étapes fragiles (ex : I/O) ;
      - planification : exécutions programmées (toutes les heures, cron...) ;
      - historique : on peut comparer les runs entre eux (dérive, régressions).

PRINCIPE DE CE FICHIER
    Le code métier reste dans src/decrochage/ (loaders, features, modèle).
    Ici on ne fait QUE l'orchestration : chaque étape devient une `@task`,
    l'enchaînement devient un `@flow`. C'est la séparation orchestration / métier.

COMMANDES (depuis decrochage-skeleton/, après `uv sync --extra dev`)
    uv run prefect cloud login              # 1 seule fois : relier le poste au compte Cloud
    uv run python flows/pipeline.py         # exécuter le pipeline → visible dans l'UI Cloud
    uv run python flows/pipeline.py --serve # créer un déploiement planifié (voir README.md)

Pas-à-pas complet (création compte, quoi regarder dans l'UI...) : flows/README.md
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Racine du projet (decrochage-skeleton/), calculée depuis ce fichier :
# le flow marche quel que soit le dossier depuis lequel on le lance.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # filet de sécurité si `uv pip install -e .` n'a pas été fait
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402

# On réutilise le code métier existant : le flow n'invente RIEN, il orchestre.
from decrochage.config import settings  # noqa: E402
from decrochage.data.loaders import (  # noqa: E402
    build_dataset,
    load_incidents,
    load_pressure,
    load_temperature,
)
from decrochage.features.temporal import add_temporal_features  # noqa: E402
from decrochage.models.tabular import (  # noqa: E402
    load_model,
    predict_proba,
    save_model,
    select_features,
    train_model,
)

# ---------------------------------------------------------------------------
# Les TASKS : une task = une étape observable, rejouable, avec retry possible.
# Dans l'UI Prefect, chaque task apparaît comme un nœud du graphe d'exécution.
# ---------------------------------------------------------------------------


@task(retries=2, retry_delay_seconds=5)
def charger_sources(data_dir: Path) -> pd.DataFrame:
    """Charge et harmonise les 3 sources réelles (CSV ';', TSV, incidents).

    `retries=2` : si la lecture échoue (fichier verrouillé, réseau...), Prefect
    retente 2 fois à 5 s d'intervalle AVANT de mettre le run en échec.
    C'est le genre d'étape I/O qu'on protège toujours en production.
    """
    logger = get_run_logger()  # logger Prefect : les messages remontent dans l'UI Cloud
    temp = load_temperature(data_dir / "capteurs_temperature.csv")
    pres = load_pressure(data_dir / "capteurs_pression.tsv")
    inc = load_incidents(data_dir / "releves_incidents.csv")
    ds = build_dataset(temp, pres, inc, window_hours=settings.incident_window_hours)
    logger.info(f"Dataset assemblé : {len(ds)} lignes, {ds['machine'].nunique()} machines")
    return ds


@task
def construire_features(ds: pd.DataFrame) -> pd.DataFrame:
    """Ajoute lags + moyennes glissantes par machine (sans fuite temporelle)."""
    logger = get_run_logger()
    ds = add_temporal_features(ds).dropna()
    logger.info(f"Features temporelles : {ds.shape[1]} colonnes, {len(ds)} lignes exploitables")
    return ds


@task
def entrainer_modele(ds: pd.DataFrame, out: Path, data_dir: Path) -> dict:
    """Entraîne le RandomForest et persiste modèle + métadonnées (traçabilité)."""
    logger = get_run_logger()
    X, y = select_features(ds, settings.target_col), ds[settings.target_col]
    model = train_model(X, y, random_state=settings.random_seed)
    save_model(model, out)
    # Mêmes métadonnées que `decrochage train` + provenance de l'orchestration :
    # en audit, on doit pouvoir dire QUI a produit CE modèle, QUAND, avec QUELLES données.
    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "package_version": "0.1.0",
        "random_seed": settings.random_seed,
        "target_col": settings.target_col,
        "features": list(X.columns),
        "n_train_rows": int(len(ds)),
        "panne_rate": round(float(y.mean()), 4),
        "dataset": str(data_dir),
        "orchestrator": "prefect",
    }
    (out.parent / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Modèle entraîné ({len(ds)} lignes, panne={y.mean():.2%}) → {out}")
    return meta


@task
def scorer_machines(model_path: Path, ds: pd.DataFrame) -> dict[str, float]:
    """Score la dernière observation de chaque machine : P(panne) ∈ [0, 1]."""
    model = load_model(model_path)
    last = ds.groupby("machine").tail(1)  # 1 ligne par machine = son état le plus récent
    proba = predict_proba(model, select_features(last, settings.target_col))
    return {m: round(float(p), 3) for m, p in zip(last["machine"], proba, strict=False)}


@task
def publier_rapport(meta: dict, scores: dict[str, float]) -> None:
    """Publie un rapport markdown : UI Cloud → onglet Artifacts du run.

    Un "artifact" Prefect = un livrable lisible attaché au run (rapport, tableau...).
    Intérêt : le métier consulte le résultat dans l'UI sans ouvrir de terminal.
    """
    seuil = settings.decision_threshold
    lignes = "\n".join(
        f"| {machine} | {p:.3f} | {'A RISQUE' if p >= seuil else 'ok'} |"
        for machine, p in sorted(scores.items())
    )
    create_markdown_artifact(
        key="rapport-decrochage",  # clé stable : l'UI garde l'historique des versions
        description="Scoring maintenance prédictive Decrochage",
        markdown=(
            f"# Decrochage — rapport de run\n\n"
            f"- Entraînement : {meta['n_train_rows']} lignes, "
            f"taux de panne {meta['panne_rate']:.2%}\n"
            f"- Seuil de décision : {seuil}\n\n"
            f"| Machine | P(panne) | Statut |\n|---|---|---|\n{lignes}\n"
        ),
    )


# ---------------------------------------------------------------------------
# Le FLOW : le chef d'orchestre. Il enchaîne les tasks ; Prefect trace tout.
# ---------------------------------------------------------------------------


@flow(name="decrochage-pipeline", log_prints=True)  # log_prints : les print() → logs du run
def pipeline_decrochage(data_dir: str | None = None) -> dict[str, float]:
    """Pipeline complet : sources → features → entraînement → scoring → rapport.

    `data_dir` est un PARAMÈTRE de flow : dans l'UI Cloud on peut relancer le
    pipeline sur un autre jeu de données sans toucher au code (Deployments → Run).
    """
    dd = Path(data_dir) if data_dir else ROOT / "data" / "sample"
    out = ROOT / "artifacts" / "models" / "rf.joblib"

    ds = construire_features(charger_sources(dd))  # les tasks s'enchaînent comme des fonctions
    meta = entrainer_modele(ds, out, dd)
    scores = scorer_machines(out, ds)
    publier_rapport(meta, scores)

    a_risque = [m for m, p in scores.items() if p >= settings.decision_threshold]
    print(f"{len(scores)} machines scorées, {len(a_risque)} au-dessus du seuil : {a_risque}")
    return scores


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # MODE DÉPLOIEMENT : `serve()` enregistre un "deployment" planifié dans
        # Prefect Cloud et transforme CE process en mini-worker local qui exécute
        # les runs. Tant qu'il tourne (Ctrl+C pour arrêter) :
        #   - exécution automatique toutes les heures (interval=3600 s) ;
        #   - déclenchement à la demande depuis l'UI : Deployments → Run.
        # Pas d'infra à gérer : idéal pour la démo. En prod réelle : workers + work pools.
        pipeline_decrochage.serve(
            name="decrochage-horaire",
            interval=3600,
            tags=["decrochage", "sprint3"],
        )
    else:
        # MODE SIMPLE : une exécution immédiate, tracée dans le Cloud si le poste
        # est connecté (`prefect cloud login`), sinon suivie par un serveur local éphémère.
        pipeline_decrochage()
