#!/usr/bin/env python3
# =============================================================================
#  scripts/demo_versioning.py  —  DÉMO « versionner DONNÉES + MODÈLE »
# -----------------------------------------------------------------------------
#  Script « tout-en-un » qui déroule une démo complète de MLOps :
#    1. ENTRAÎNE un modèle sur le gold (split temporel honnête, sans fuite) ;
#    2. JOURNALISE params + métriques + modèle dans MLflow (+ Model Registry) ;
#    3. SAUVE l'artefact rf.joblib + ses métadonnées ;
#    4. ÉCRIT metrics.json / params.yaml (lus par `dvc metrics show`) ;
#    5. VERSIONNE les fichiers lourds avec DVC (`dvc add` data + modèle) ;
#    6. AFFICHE les commandes git/dvc à copier-coller pour figer la version.
#
#  Idempotent (on peut le relancer) et pilotable : --split, --no-mlflow,
#  --no-dvc, --remote… Chaque brique externe (MLflow, DVC) est protégée par
#  try/except : si l'outil manque, on saute l'étape proprement au lieu de planter.
# =============================================================================
r"""Demo : versionner DONNEES + MODELE avec DVC et MLflow (Decrochage Sprint 3).

Ce script "tout-en-un" :
  1. entraine le modele sur le gold dataset (split TEMPOREL, sans fuite) ;
  2. journalise params + metriques + modele dans MLflow (+ enregistrement au
     Model Registry) ;
  3. sauvegarde l'artefact `artifacts/models/rf.joblib` + ses metadonnees ;
  4. ecrit `metrics.json` / `params.yaml` (lisibles par `dvc metrics show`) ;
  5. versionne les fichiers lourds avec DVC (`dvc add` data + modele) ;
  6. affiche les commandes git/dvc a copier-coller pour figer la version.

Tout est idempotent et pilotable :  --no-mlflow / --no-dvc / --remote ...

    uv run python scripts/demo_versioning.py
    uv run python scripts/demo_versioning.py --remote /tmp/dvc-store    # Mac/Linux
    uv run python scripts/demo_versioning.py --remote C:\dvc-store      # Windows
"""

from __future__ import annotations  # annotations de type modernes

import argparse  # options de ligne de commande
import hashlib  # calcul d'empreinte md5 (identifie une version de données)
import json  # écriture des métadonnées / métriques en JSON
import shutil  # shutil.which() : vérifier qu'un outil (dvc) est installé
import subprocess  # lancer des commandes externes (dvc, git)
import sys  # bootstrap du sys.path + code de sortie
from datetime import (  # horodatage UTC (timezone.utc = portable Python 3.10+)
    UTC,
    datetime,
)
from pathlib import Path  # chemins de fichiers

import numpy as np  # calcul (argmax, vecteurs de proba)
import pandas as pd  # DataFrame (lecture du gold, splits)
from sklearn.metrics import (  # métriques d'évaluation du classifieur
    average_precision_score,  # PR-AUC : qualité sur la classe rare (pannes)
    f1_score,  # compromis précision/rappel à un seuil donné
    precision_recall_curve,  # courbe précision/rappel → pour choisir le seuil
    precision_score,  # part de vraies pannes parmi les alertes
    recall_score,  # part de pannes détectées
    roc_auc_score,  # ROC-AUC : capacité de classement global
)
from sklearn.model_selection import (
    train_test_split,  # split aléatoire stratifié (mode démo « fuite »)
)

# --- Bootstrap : rendre le package `decrochage` importable --------------------
ROOT = Path(__file__).resolve().parents[1]  # racine du repo (scripts/ → ..)
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# On réutilise les VRAIES fonctions du package (pas de logique ML dupliquée).
from decrochage.models.tabular import (  # noqa: E402
    predict_proba,
    save_model,
    select_features,
    train_model,
)

try:
    # On récupère la version du package pour la tracer dans les métadonnées.
    from decrochage import __version__ as PKG_VERSION  # noqa: E402
except Exception:  # pragma: no cover  (filet de sécurité si l'import échoue)
    PKG_VERSION = "0.1.0"

TARGET = "panne"  # nom de la colonne cible
# Dépendances FIGÉES pour l'environnement du modèle MLflow. On les précise pour
# éviter l'inférence automatique d'environnement (fragile selon les machines).
PIP_REQS = ["scikit-learn", "pandas", "numpy", "joblib"]


def sh(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    """Lance une commande, renvoie (code, sortie) et l'affiche."""
    print(f"   $ {' '.join(cmd)}")  # affiche la commande (pédagogie/transparence)
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True
    )  # exécute, capture la sortie
    out = (proc.stdout + proc.stderr).strip()  # concatène sortie standard + erreurs
    if out:
        print("   " + out.replace("\n", "\n   "))  # ré-indente joliment la sortie
    return proc.returncode, out  # code 0 = succès ; texte = sortie


def md5(path: Path) -> str:
    # Empreinte md5 du fichier (12 1ers caractères) = « signature » d'une version
    # de données. Si le CSV change d'un octet, l'empreinte change → on sait qu'on
    # a une nouvelle version (on l'utilise pour nommer le run MLflow et le tag git).
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def best_f1_threshold(y: pd.Series, proba: np.ndarray) -> float:
    """Seuil qui maximise le F1 (calibre sur le TRAIN), pour un point de
    fonctionnement utile au lieu d'un 0.5 arbitraire."""
    # precision_recall_curve renvoie precision, recall et les seuils testés.
    prec, rec, thr = precision_recall_curve(y, proba)
    if len(thr) == 0:  # cas dégénéré (une seule classe) → seuil neutre
        return 0.5
    # F1 = moyenne harmonique précision/rappel ; +1e-12 évite la division par zéro.
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1))])  # seuil qui donne le meilleur F1


def make_split(df: pd.DataFrame, test_size: float, kind: str):
    """kind='temporal' : split temporel PAR MACHINE (honnete, sans fuite).
    kind='stratified' : split iid stratifie (PRATIQUE A FUITE - pour la demo
    du contraste : features glissantes + meme machine a cheval train/test)."""
    # --- Mode DÉMO « fuite » : split aléatoire stratifié ---------------------
    # Train et test piochés au hasard → des lignes voisines dans le temps se
    # retrouvent des deux côtés. Le score paraît excellent… mais c'est une ILLUSION.
    if kind == "stratified":
        tr, te = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=42)
        return tr.reset_index(drop=True), te.reset_index(drop=True), "stratifie (fuite)"
    # --- Mode HONNÊTE : split temporel par machine ---------------------------
    df = df.sort_values(["machine", "timestamp"]).reset_index(
        drop=True
    )  # ordre chronologique par machine
    test_idx: list[int] = []
    for _, grp in df.groupby("machine"):  # pour chaque machine…
        k = max(1, int(round(len(grp) * test_size)))  # taille du test (au moins 1 ligne)
        test_idx.extend(
            grp.index[-k:].tolist()
        )  # …on prend ses DERNIÈRES lignes (le futur) en test
    mask = df.index.isin(test_idx)  # masque booléen : True = ligne de test
    train, test = df[~mask], df[mask]  # ~mask = le passé (train), mask = le futur (test)
    # Garde-fou : si le test n'a aucune panne, les métriques (PR-AUC, ROC-AUC)
    # n'ont pas de sens → on bascule sur un split stratifié de secours.
    if test[TARGET].nunique() < 2 or test[TARGET].sum() < 1:
        print("   [!] split temporel sans panne en test -> repli sur stratifie.")
        tr, te = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=42)
        return tr.reset_index(drop=True), te.reset_index(drop=True), "stratifie (repli)"
    return train.reset_index(drop=True), test.reset_index(drop=True), "temporel/machine"


def main() -> int:
    # --- Options de la ligne de commande -------------------------------------
    ap = argparse.ArgumentParser(description="Demo versioning DVC + MLflow")
    ap.add_argument(
        "--gold", type=Path, default=ROOT / "data/gold/gold_dataset.csv"
    )  # données d'entrée
    ap.add_argument(
        "--model-out", type=Path, default=ROOT / "artifacts/models/rf.joblib"
    )  # sortie modèle
    ap.add_argument("--experiment", default="decrochage-maintenance")  # nom de l'expérience MLflow
    ap.add_argument("--registered-name", default="decrochage-rf")  # nom au Model Registry
    ap.add_argument(
        "--tracking-uri", default=None, help="URI MLflow (defaut: file:./mlruns dans le repo)"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="seuil de decision (defaut: auto, max-F1 sur le train)",
    )
    ap.add_argument(
        "--split",
        choices=["temporal", "stratified"],
        default="temporal",
        help="temporal=honnete (defaut) | stratified=demo de la fuite",
    )
    ap.add_argument("--test-size", type=float, default=0.2)  # part du jeu de test (20 %)
    ap.add_argument("--n-estimators", type=int, default=200)  # nombre d'arbres du RandomForest
    ap.add_argument("--seed", type=int, default=42)  # graine (reproductibilité)
    ap.add_argument(
        "--remote",
        type=str,
        default=None,
        help=(
            r"chemin d'un remote DVC local "
            r"(ex: /tmp/dvc-store sous Mac/Linux, C:\dvc-store sous Windows)"
        ),
    )
    ap.add_argument("--no-mlflow", action="store_true")  # sauter l'étape MLflow
    ap.add_argument("--no-dvc", action="store_true")  # sauter l'étape DVC
    args = ap.parse_args()

    if not args.gold.exists():  # garde-fou : le gold doit exister
        ap.error(f"Gold introuvable : {args.gold} (lance `decrochage build-gold`).")

    # ---------------------------------------------------------------- 1. TRAIN
    print("== 1. Entrainement ==")
    df = pd.read_csv(args.gold, parse_dates=["timestamp"])  # lit le gold (dates parsées)
    train, test, mode = make_split(
        df, args.test_size, args.split
    )  # sépare train/test selon le mode
    x_tr, y_tr = select_features(train, TARGET), train[TARGET]  # X/y d'entraînement
    x_te, y_te = select_features(test, TARGET), test[TARGET]  # X/y de test
    model = train_model(
        x_tr, y_tr, n_estimators=args.n_estimators, random_state=args.seed
    )  # entraîne
    # Seuil de décision : par défaut calibré (max-F1 sur le train), sinon imposé en option.
    proba_tr = predict_proba(model, x_tr)  # probas sur le train (pour le seuil)
    threshold = args.threshold if args.threshold is not None else best_f1_threshold(y_tr, proba_tr)
    proba = predict_proba(model, x_te)  # probas de panne sur le test
    preds = (proba >= threshold).astype(int)  # décisions 0/1 au seuil choisi

    # Dictionnaire des métriques (arrondies) : ce qu'on va journaliser dans MLflow.
    metrics = {
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),  # qualité sur classe rare
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),  # classement global
        "precision": round(float(precision_score(y_te, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "panne_rate_test": round(float(y_te.mean()), 4),  # taux de pannes dans le test
        "threshold": round(float(threshold), 4),  # seuil retenu
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "split": mode,
    }
    print(
        f"   split={mode} | train={len(train)} test={len(test)} | "
        f"PR-AUC={metrics['pr_auc']} ROC-AUC={metrics['roc_auc']} "
        f"seuil={threshold:.3f} recall={metrics['recall']}"
    )

    # MODÈLE DE PRODUCTION : on ré-entraîne sur TOUT le gold (train+test) pour
    # ne pas « gâcher » de données une fois l'évaluation faite. Les métriques
    # ci-dessus restent celles du test (honnêtes), mais l'artefact livré voit tout.
    x_all, y_all = select_features(df, TARGET), df[TARGET]
    model_full = train_model(x_all, y_all, n_estimators=args.n_estimators, random_state=args.seed)
    save_model(model_full, args.model_out)  # sauvegarde rf.joblib

    # Métadonnées du modèle = sa carte d'identité (traçabilité / reproductibilité).
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),  # date UTC
        "package_version": PKG_VERSION,
        "random_seed": args.seed,
        "target_col": TARGET,
        "features": list(x_all.columns),  # features exactes utilisées
        "n_train_rows": int(len(df)),
        "panne_rate": round(float(y_all.mean()), 4),
        "dataset": str(args.gold.relative_to(ROOT)),  # quelle source de données
        "threshold": round(float(threshold), 4),
        "gold_md5": md5(args.gold),  # signature de la VERSION de données
        "metrics_holdout": metrics,  # métriques d'évaluation
    }
    meta_path = args.model_out.parent / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")  # écrit le JSON
    print(f"   modele -> {args.model_out.relative_to(ROOT)} | metadata -> {meta_path.name}")

    # Fichiers DVC metrics/params : versionnés EN CLAIR dans git (petits, lisibles).
    # `dvc metrics show` et `dvc params diff` sauront les lire pour comparer des versions.
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "params.yaml").write_text(
        "train:\n"
        f"  n_estimators: {args.n_estimators}\n"
        f"  random_state: {args.seed}\n"
        f"  threshold: {round(float(threshold), 4)}\n"
        f"  test_size: {args.test_size}\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------- 2. MLflow
    run_id = None
    if not args.no_mlflow:  # sauf si --no-mlflow
        print("== 2. MLflow (params + metriques + modele + registre) ==")
        try:
            import mlflow  # import LOCAL : si mlflow manque, on tombe dans l'except plus bas
            import mlflow.sklearn  # sous-module pour journaliser un modèle scikit-learn
            from mlflow.models import (
                infer_signature,  # déduit la « signature » (types entrée/sortie)
            )

            # Backend SQLite : un fichier mlflow.db. Nécessaire pour le Model Registry
            # (un simple dossier ./mlruns ne supporterait pas l'enregistrement).
            uri = args.tracking_uri or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
            mlflow.set_tracking_uri(uri)  # où MLflow enregistre tout
            mlflow.set_experiment(args.experiment)  # regroupe les runs sous une expérience nommée
            with mlflow.start_run(
                run_name=f"rf-{metadata['gold_md5']}"
            ) as run:  # un « run » = un essai
                run_id = run.info.run_id
                mlflow.log_params(
                    {  # PARAMÈTRES (les réglages de l'entraînement)
                        "n_estimators": args.n_estimators,
                        "random_state": args.seed,
                        "threshold": round(float(threshold), 4),
                        "test_size": args.test_size,
                        "n_features": x_all.shape[1],
                        "split": mode,
                    }
                )
                # MÉTRIQUES (résultats chiffrés) : on ne garde que les valeurs numériques.
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
                # TAGS (étiquettes libres) : on trace la source et la signature des données.
                mlflow.set_tags({"dataset": metadata["dataset"], "gold_md5": metadata["gold_md5"]})
                mlflow.log_artifact(str(meta_path))  # joint le fichier de métadonnées au run
                sig = infer_signature(x_te, proba)  # signature = schéma entrée→sortie du modèle

                def _log(**extra):
                    # Compatibilité : MLflow 3.x attend `name=`, MLflow 2.x `artifact_path=`.
                    # On tente d'abord la syntaxe récente, et on retombe sur l'ancienne si erreur.
                    try:
                        return mlflow.sklearn.log_model(
                            model_full,
                            name="model",
                            signature=sig,
                            pip_requirements=PIP_REQS,
                            **extra,
                        )
                    except TypeError:
                        return mlflow.sklearn.log_model(
                            model_full,
                            artifact_path="model",
                            signature=sig,
                            pip_requirements=PIP_REQS,
                            **extra,
                        )

                try:
                    # Enregistre le modèle au REGISTRE (versions nommées : v1, v2, …).
                    _log(input_example=x_te.head(2), registered_model_name=args.registered_name)
                    print(f"   modele enregistre au registre : '{args.registered_name}'")
                except Exception as exc:  # registre indispo (vieux backend) -> log simple
                    _log()  # au pire on journalise le modèle sans l'enregistrer
                    print(f"   [i] modele journalise (registre indispo : {exc})")
            print(f"   run_id={run_id} | tracking={uri}")
            print(f"   -> visualiser :  uv run mlflow ui --backend-store-uri {uri}")
        except ImportError:  # mlflow n'est pas installé
            print("   [!] mlflow non installe -> `uv add mlflow`. Etape sautee.")
        except Exception as exc:  # tout autre souci MLflow : on n'interrompt pas la démo
            print(f"   [!] MLflow a echoue : {exc}")

    # ---------------------------------------------------------------- 3. DVC
    if not args.no_dvc:  # sauf si --no-dvc
        print("== 3. DVC (versionner data + modele) ==")
        if shutil.which("dvc") is None:  # dvc est-il installé / dans le PATH ?
            print("   [!] dvc non installe -> `uv add dvc`. Etape sautee.")
        else:
            if not (ROOT / ".dvc").exists():  # 1re fois : initialise DVC dans le repo
                sh(["dvc", "init", "-q"])
            if args.remote:  # configure un « remote » (stockage des gros fichiers)
                Path(args.remote).mkdir(parents=True, exist_ok=True)
                sh(["dvc", "remote", "add", "-d", "-f", "localremote", args.remote])
            # Les chemins relatifs à versionner : le gold (données) et le modèle.
            rels = [str(args.gold.relative_to(ROOT)), str(args.model_out.relative_to(ROOT))]
            for rel in rels:
                # MIGRATION git → DVC : DVC refuse un fichier déjà suivi par git.
                # On teste s'il est suivi (git ls-files), et si oui on le sort de l'index git.
                tracked = (
                    subprocess.run(
                        ["git", "ls-files", "--error-unmatch", rel],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                    ).returncode
                    == 0
                )
                if tracked:
                    print(f"   [migration git->dvc] {rel} etait suivi par git")
                    sh(
                        ["git", "rm", "-r", "--cached", "-q", rel]
                    )  # --cached = retire de git, garde le fichier
            sh(["dvc", "add", *rels])  # crée les pointeurs .dvc (le vrai fichier part au remote)
            print(
                "   -> 2 pointeurs crees : data/gold/gold_dataset.csv.dvc + "
                "artifacts/models/rf.joblib.dvc"
            )

    # ---------------------------------------------------------------- 4. SUITE
    # On n'exécute PAS git commit à la place de l'apprenant : on lui DONNE les
    # commandes à copier-coller pour figer la version (commit + tag + push).
    print("\n== 4. Figer la version (a copier-coller) ==")
    print("   git add data/gold/gold_dataset.csv.dvc artifacts/models/rf.joblib.dvc \\")
    print("           .gitignore metrics.json params.yaml")
    print(f'   git commit -m "data+model {metadata["gold_md5"]} | PR-AUC {metrics["pr_auc"]}"')
    print(f'   git tag data-{metadata["gold_md5"]}')  # tag = étiquette lisible de cette version
    if args.remote:
        print("   dvc push            # envoie data+modele vers le remote")
    print("   # revenir a une version :  git checkout <tag> && dvc checkout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # propage le code de retour au shell
