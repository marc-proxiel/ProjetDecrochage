#!/usr/bin/env python
# =============================================================================
#  scripts/demo_prefect_idempotence.py
#  Démo autonome du MODULE 30 : un flow Prefect ingest -> predict -> store,
#  qui prouve l'IDEMPOTENCE (rejouer ne crée AUCUN doublon) et permet
#  d'AJOUTER de nouvelles entrées à volonté.
# -----------------------------------------------------------------------------
#  Ce que ça montre :
#    - Prefect en action : 3 tâches (@task) enchaînées dans un @flow, avec logs
#      et RETRIES (option --flaky pour voir un réessai en direct).
#    - L'idempotence : la table `predictions` a une clé (machine, prediction_ts)
#      et on écrit en UPSERT (INSERT ... ON CONFLICT ... DO UPDATE). Donc rejouer
#      le flow ne duplique rien : `count(*)` reste stable.
#    - L'ajout de nouvelles données : --new N ajoute N moments de prédiction
#      supplémentaires par machine (nouvelles clés -> nouvelles lignes).
#
#  Base de données : SQLite local (`predictions.db`) => AUCUN Docker requis.
#  (En prod / stack compose, ce serait PostgreSQL, avec le même principe d'upsert.)
#
#  USAGE (depuis le dossier du dépôt) :
#     uv run python scripts/demo_prefect_idempotence.py
#     uv run python scripts/demo_prefect_idempotence.py --new 2
#     uv run python scripts/demo_prefect_idempotence.py --flaky
#     uv run python scripts/demo_prefect_idempotence.py --reset
#     uv run python scripts/demo_prefect_idempotence.py --show
# =============================================================================
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task

# On réutilise le code TESTÉ du projet (aucune logique métier réécrite ici) :
from decrochage.config import settings
from decrochage.features.temporal import add_temporal_features
from decrochage.models.tabular import load_model, predict_proba, select_features

DB = Path("predictions.db")
MACHINES = ["MACH-01", "MACH-02", "MACH-03", "MACH-04"]  # nos 4 machines de démo
T0 = dt.datetime(2025, 2, 1, 0, 0, 0)


def _readings(machine: str, end_ts: dt.datetime, n: int = 10, seed: int = 0) -> pd.DataFrame:
    """Fabrique n relevés horaires (temp, pression) finissant à `end_ts`, de façon
    DÉTERMINISTE (même graine => mêmes valeurs) : ainsi rejouer donne les mêmes
    prédictions, ce qui rend l'idempotence parfaitement visible."""
    rng = np.random.default_rng(seed)
    base_t = 50 + MACHINES.index(machine) * 2
    rows = []
    for i in range(n):
        ts = end_ts - dt.timedelta(hours=n - 1 - i)
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "temperature": base_t + i * 0.5 + rng.normal(0, 0.3),
                "pressure_bar": 195 + i * 0.4 + rng.normal(0, 0.2),
            }
        )
    return pd.DataFrame(rows)


@task
def ingest(moments_per_machine: int) -> list[tuple[str, dt.datetime]]:
    """ÉTAPE 1 — décide QUOI prédire : la liste des (machine, horodatage)."""
    log = get_run_logger()
    moments = [
        (m, T0 + dt.timedelta(hours=h)) for m in MACHINES for h in range(moments_per_machine)
    ]
    log.info(
        "ingest : %d moments à prédire (%d machines x %d)",
        len(moments),
        len(MACHINES),
        moments_per_machine,
    )
    return moments


@task
def predict_batch(model, moments: list[tuple[str, dt.datetime]]) -> list[tuple]:
    """ÉTAPE 2 — prédit pour chaque moment (même pipeline que l'API : features
    sans fuite -> dernière ligne -> proba)."""
    log = get_run_logger()
    out = []
    for machine, ts in moments:
        df = _readings(machine, ts, seed=MACHINES.index(machine) * 1000 + ts.hour)
        df["machine"] = machine
        feats = add_temporal_features(df).dropna()
        X = select_features(feats, settings.target_col).iloc[[-1]]
        proba = float(predict_proba(model, X)[0])
        decision = "alerte" if proba >= settings.decision_threshold else "ok"
        out.append((machine, ts.isoformat(), round(proba, 4), decision))
    log.info("predict : %d prédictions calculées", len(out))
    return out


# retries=2 : si la tâche échoue (erreur TRANSITOIRE), Prefect la relance 2 fois.
@task(retries=2, retry_delay_seconds=1)
def store(rows: list[tuple], flaky: bool = False) -> tuple[int, int]:
    """ÉTAPE 3 — persiste en base par UPSERT (clé = machine + prediction_ts).
    C'est ICI que se joue l'idempotence : une clé déjà vue est MISE À JOUR,
    pas dupliquée."""
    log = get_run_logger()

    # --flaky : on fait échouer la PREMIÈRE tentative pour VOIR Prefect réessayer.
    if flaky and not getattr(store, "_already_failed", False):
        store._already_failed = True
        raise RuntimeError("panne transitoire simulée de la base - Prefect va réessayer")

    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
               machine       TEXT NOT NULL,
               prediction_ts TEXT NOT NULL,
               proba_panne   REAL NOT NULL,
               decision      TEXT NOT NULL,
               written_at    TEXT NOT NULL,
               PRIMARY KEY (machine, prediction_ts)
           )""")
    before = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    now = dt.datetime.now().isoformat(timespec="seconds")
    con.executemany(
        """INSERT INTO predictions (machine, prediction_ts, proba_panne, decision, written_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(machine, prediction_ts) DO UPDATE SET
               proba_panne = excluded.proba_panne,
               decision    = excluded.decision,
               written_at  = excluded.written_at""",
        [(m, ts, p, d, now) for (m, ts, p, d) in rows],
    )
    con.commit()
    after = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    con.close()
    log.info(
        "store : %d prédictions traitées | lignes en base %d -> %d (nouvelles : %d)",
        len(rows),
        before,
        after,
        after - before,
    )
    return before, after


@flow(name="decrochage-predict-demo")
def predict_flow(moments_per_machine: int = 3, flaky: bool = False) -> int:
    """Le FLOW : ingest -> predict -> store. Rejoue-le : le total ne bouge pas."""
    log = get_run_logger()
    model = load_model(settings.model_dir / "rf.joblib")
    moments = ingest(moments_per_machine)
    rows = predict_batch(model, moments)
    before, after = store(rows, flaky)
    if after == before:
        verdict = "IDEMPOTENT (0 nouvelle ligne)"
    else:
        verdict = f"{after - before} nouvelle(s) ligne(s)"
    log.info("FLOW terminé : %d lignes en base -> %s", after, verdict)
    return after


def _write_html(path: str = "predictions.html") -> str:
    """Écrit une vue HTML lisible de la table (retour visuel dans le navigateur)."""
    p = Path(path)
    if not DB.exists():
        p.write_text("<p>(base vide : lance le flow une fois)</p>", encoding="utf-8")
        return str(p.resolve())
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT machine, prediction_ts, proba_panne, decision, written_at "
        "FROM predictions ORDER BY machine, prediction_ts"
    ).fetchall()
    con.close()
    total = len(rows)
    n_alerte = sum(1 for r in rows if r[3] == "alerte")
    last = max((r[4] for r in rows), default="-")
    trs = ""
    for m, ts, proba, dec, wat in rows:
        color = "#ED1548" if dec == "alerte" else "#0B9E6E"
        trs += (
            f"<tr><td>{m}</td><td>{ts}</td>"
            f"<td style='text-align:right'>{proba:.4f}</td>"
            f"<td><span style='background:{color};color:#fff;padding:2px 9px;"
            f"border-radius:10px;font-size:12px'>{dec}</span></td>"
            f"<td style='color:#5A6B73'>{wat}</td></tr>"
        )
    css = (
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#093544}"
        "h1{margin:0 0 4px}.sub{color:#5A6B73;margin:0 0 16px}"
        ".kpis{display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap}"
        ".kpi{background:#F4F7F8;border-left:4px solid #0B4F6C;"
        "padding:8px 16px;border-radius:6px}"
        ".kpi b{font-size:22px}table{border-collapse:collapse;width:100%}"
        "th,td{border-bottom:1px solid #E3EDF0;padding:8px 10px;"
        "text-align:left;font-size:14px}"
        "th{background:#093544;color:#fff}tr:nth-child(even){background:#FAFCFD}"
    )
    html = (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
        "<meta http-equiv='refresh' content='3'>"
        "<title>Decrochage - table predictions</title><style>" + css + "</style></head><body>"
        "<h1>Table <code>predictions</code> - Decrochage 4.0</h1>"
        "<p class='sub'>Cette page se rafraichit toute seule (3 s). "
        "Relance le flow et regarde le tableau changer.</p>"
        "<div class='kpis'>"
        "<div class='kpi'><b>" + str(total) + "</b><br>lignes</div>"
        "<div class='kpi'><b>" + str(n_alerte) + "</b><br>alerte</div>"
        "<div class='kpi'><b>" + str(total - n_alerte) + "</b><br>ok</div>"
        "<div class='kpi'><b>" + str(last) + "</b><br>dernier ecrit</div></div>"
        "<table><thead><tr><th>machine</th><th>prediction_ts</th><th>proba_panne</th>"
        "<th>decision</th><th>written_at</th></tr></thead><tbody>"
        + trs
        + "</tbody></table></body></html>"
    )
    p.write_text(html, encoding="utf-8")
    return str(p.resolve())


def _show() -> None:
    if not DB.exists():
        print("(base vide : lance le script une première fois)")
        return
    con = sqlite3.connect(DB)
    total = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    print(f"\n=== predictions.db : {total} lignes ===")
    for machine, ts, proba, decision in con.execute(
        "SELECT machine, prediction_ts, proba_panne, decision FROM predictions "
        "ORDER BY machine, prediction_ts"
    ):
        print(f"  {machine:<8} {ts}  proba={proba:.4f}  {decision}")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Démo Prefect + idempotence (module 30).")
    ap.add_argument(
        "--new",
        type=int,
        default=0,
        help="ajoute N moments/machine en plus des 3 de base",
    )
    ap.add_argument(
        "--flaky",
        action="store_true",
        help="simule une panne transitoire (voir Prefect réessayer)",
    )
    ap.add_argument("--reset", action="store_true", help="vide la base avant de commencer")
    ap.add_argument("--show", action="store_true", help="affiche le contenu de la base puis quitte")
    ap.add_argument(
        "--html",
        action="store_true",
        help="ouvre la table dans le navigateur (predictions.html)",
    )
    args = ap.parse_args()

    if args.reset and DB.exists():
        DB.unlink()
        print("[base remise à zéro]")
    if args.show:
        _show()
    else:
        predict_flow(moments_per_machine=3 + args.new, flaky=args.flaky)
        _show()
    view = _write_html()
    print(
        f"\nVue visuelle -> {view}\n"
        "(ouvre-la dans un navigateur ; elle se rafraîchit toute seule)"
    )
    if args.html:
        webbrowser.open(Path(view).resolve().as_uri())
