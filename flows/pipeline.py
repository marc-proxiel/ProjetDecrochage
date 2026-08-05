"""Orchestration Prefect du pipeline Decrochage (Sprint 3 — modules B9/B10 industrialisation).

POURQUOI UN ORCHESTRATEUR ?
    Jusqu'ici on lancait le pipeline a la main (`decrochage build-gold` puis
    `decrochage train`). En production, un orchestrateur apporte ce qu'un
    script + cron ne donnent pas :
      - observabilite : chaque execution (= "flow run") est tracee dans une UI,
        avec logs, durees, graphe des etapes ;
      - resilience : retries automatiques sur les etapes fragiles (ex : I/O) ;
      - planification : executions programmees (toutes les heures, cron...) ;
      - historique : on peut comparer les runs entre eux (derive, regressions).

PRINCIPE DE CE FICHIER
    Le code metier reste dans src/decrochage/ (construction du gold, modele).
    Ici on ne fait QUE l'orchestration : chaque etape devient une `@task`,
    l'enchainement devient un `@flow`. C'est la separation orchestration / metier.

COMMANDES (depuis la racine du repo, apres `uv sync --extra dev`)
    uv run prefect cloud login              # 1 seule fois : relier le poste au compte Cloud
    uv run python flows/pipeline.py         # executer le pipeline -> visible dans l'UI Cloud
    uv run python flows/pipeline.py --serve # creer un deploiement planifie (voir README.md)

Pas-a-pas complet (creation compte, quoi regarder dans l'UI...) : flows/README.md
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Racine du projet, calculee depuis ce fichier : le flow marche quel que soit
# le dossier depuis lequel on le lance.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # filet de securite si `uv pip install -e .` n'a pas ete fait
    sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402

from decrochage import __version__  # noqa: E402

# On reutilise le code metier existant : le flow n'invente RIEN, il orchestre.
from decrochage.cli import build_gold_dataset  # noqa: E402
from decrochage.config import settings  # noqa: E402
from decrochage.models.tabular import (  # noqa: E402
    COLONNES_A_IMPUTER,
    entrainer_et_evaluer,
    load_model,
    predict_proba,
    save_model,
    select_features,
)

# ---------------------------------------------------------------------------
# Les TASKS : une task = une etape observable, rejouable, avec retry possible.
# Dans l'UI Prefect, chaque task apparait comme un noeud du graphe d'execution.
# ---------------------------------------------------------------------------


@task(retries=2, retry_delay_seconds=5)
def construire_gold(data_dir: Path, gold_dir: Path) -> Path:
    """Fusionne les sources brutes et encode le dataset gold (voir le notebook).

    `retries=2` : si la lecture des CSV echoue (fichier verrouille, disque
    reseau...), Prefect retente 2 fois a 5 s d'intervalle AVANT de mettre le
    run en echec. C'est le genre d'etape I/O qu'on protege toujours en prod.
    """
    logger = get_run_logger()  # logger Prefect : les messages remontent dans l'UI Cloud
    df_gold = build_gold_dataset(data_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    out = gold_dir / "gold-dataset.csv"
    df_gold.to_csv(out, index=False, sep=";")
    logger.info(f"Gold construit : {len(df_gold)} lignes, {df_gold.shape[1]} colonnes -> {out}")
    return out


@task
def entrainer_modele(gold_path: Path, model_dir: Path, seed: int) -> dict:
    """Split/imputation/entrainement/evaluation, puis persiste modele + metadonnees."""
    logger = get_run_logger()
    df_gold = pd.read_csv(gold_path, sep=";")

    model, imputer, metrics, feature_columns = entrainer_et_evaluer(
        df_gold,
        target_col=settings.target_col,
        threshold=settings.decision_threshold,
        random_state=seed,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, model_dir / "logistic.joblib")
    joblib.dump(imputer, model_dir / "imputer.joblib")

    # Memes metadonnees que `decrochage train` + provenance de l'orchestration :
    # en audit, on doit pouvoir dire QUI a produit CE modele, QUAND, avec QUELLES donnees.
    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "random_seed": seed,
        "target_col": settings.target_col,
        "features": feature_columns,
        "metrics_holdout": metrics,
        "dataset": str(gold_path),
        "orchestrator": "prefect",
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        f"Modele entraine (rappel={metrics['recall']}, AUC={metrics['roc_auc']}) -> {model_dir}"
    )
    return meta


@task
def scorer_echantillon(
    gold_path: Path, model_dir: Path, n: int = 20, seed: int = 42
) -> dict[str, float]:
    """Score un echantillon d'etudiants du gold : P(abandon) in [0, 1] chacun."""
    df_gold = pd.read_csv(gold_path, sep=";")
    model = load_model(model_dir / "logistic.joblib")
    imputer = joblib.load(model_dir / "imputer.joblib")

    echantillon = df_gold.sample(n=min(n, len(df_gold)), random_state=seed).reset_index(drop=True)
    echantillon[COLONNES_A_IMPUTER] = imputer.transform(echantillon[COLONNES_A_IMPUTER])
    proba = predict_proba(model, select_features(echantillon, settings.target_col))
    return {f"etudiant_{i}": round(float(p), 3) for i, p in enumerate(proba)}


@task
def publier_rapport(meta: dict, scores: dict[str, float]) -> None:
    """Publie un rapport markdown : UI Cloud -> onglet Artifacts du run.

    Un "artifact" Prefect = un livrable lisible attache au run (rapport, tableau...).
    Interet : le metier consulte le resultat dans l'UI sans ouvrir de terminal.
    """
    seuil = settings.decision_threshold
    m = meta["metrics_holdout"]
    lignes = "\n".join(
        f"| {nom} | {p:.3f} | {'A RISQUE' if p >= seuil else 'ok'} |"
        for nom, p in sorted(scores.items())
    )
    create_markdown_artifact(
        key="rapport-decrochage",  # cle stable : l'UI garde l'historique des versions
        description="Detection precoce du decrochage etudiant",
        markdown=(
            f"# Decrochage — rapport de run\n\n"
            f"- Entrainement : {m['n_train']} lignes train / {m['n_test']} lignes test, "
            f"taux d'abandon {m['taux_abandon']:.2%}\n"
            f"- Performance (test) : rappel {m['recall']}, precision {m['precision']}, "
            f"F1 {m['f1']}, AUC {m['roc_auc']}\n"
            f"- Seuil de decision : {seuil}\n\n"
            f"| Etudiant (echantillon) | P(abandon) | Statut |\n|---|---|---|\n{lignes}\n"
        ),
    )


# ---------------------------------------------------------------------------
# Le FLOW : le chef d'orchestre. Il enchaine les tasks ; Prefect trace tout.
# ---------------------------------------------------------------------------


@flow(name="decrochage-pipeline", log_prints=True)  # log_prints : les print() -> logs du run
def pipeline_decrochage(data_dir: str | None = None) -> dict[str, float]:
    """Pipeline complet : sources brutes -> gold -> entrainement -> scoring -> rapport.

    `data_dir` est un PARAMETRE de flow : dans l'UI Cloud on peut relancer le
    pipeline sur un autre jeu de donnees sans toucher au code (Deployments -> Run).
    """
    dd = Path(data_dir) if data_dir else settings.data_dir

    gold_path = construire_gold(dd, settings.gold_dir)
    meta = entrainer_modele(gold_path, settings.model_dir, settings.random_seed)
    scores = scorer_echantillon(gold_path, settings.model_dir)
    publier_rapport(meta, scores)

    a_risque = [nom for nom, p in scores.items() if p >= settings.decision_threshold]
    print(
        f"{len(scores)} etudiants (echantillon) scores, "
        f"{len(a_risque)} au-dessus du seuil : {a_risque}"
    )
    return scores


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # MODE DEPLOIEMENT : `serve()` enregistre un "deployment" planifie dans
        # Prefect Cloud et transforme CE process en mini-worker local qui execute
        # les runs. Tant qu'il tourne (Ctrl+C pour arreter) :
        #   - execution automatique toutes les heures (interval=3600 s) ;
        #   - declenchement a la demande depuis l'UI : Deployments -> Run.
        # Pas d'infra a gerer : ideal pour la demo. En prod reelle : workers + work pools.
        pipeline_decrochage.serve(
            name="decrochage-horaire",
            interval=3600,
            tags=["decrochage", "sprint3"],
        )
    else:
        # MODE SIMPLE : une execution immediate, tracee dans le Cloud si le poste
        # est connecte (`prefect cloud login`), sinon suivie par un serveur local ephemere.
        pipeline_decrochage()
