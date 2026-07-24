#!/usr/bin/env python3
# =============================================================================
#  scripts/generate_synthetic_gold.py  —  GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES
# -----------------------------------------------------------------------------
#  À quoi ça sert ? Simuler un FLUX de nouvelles mesures capteurs qui s'ajoutent
#  au gold dataset, pour faire vivre la démo de versioning (DVC + MLflow) :
#  la donnée « grossit » → on peut re-versionner, ré-entraîner, comparer.
#
#  Garanties importantes :
#    • on respecte EXACTEMENT le schéma du gold (les 15 colonnes, même ordre) ;
#    • on recalcule les features temporelles avec la VRAIE fonction du package
#      (add_temporal_features) → pas de NaN, pas de fuite, statistiquement cohérent ;
#    • le label `panne` est SYNTHÉTIQUE (loi logistique liée à temp/pression),
#      calibré autour du taux réel (~10,5 %) — ce n'est PAS une vraie jointure d'incidents.
#
#  Les valeurs suivent un modèle AR(1) (retour à la moyenne + bruit) borné aux
#  plages physiques, par machine, en faisant avancer l'horloge de chaque machine.
# =============================================================================
"""Generateur de donnees synthetiques pour Decrochage (demo flux temps reel).

Ajoute periodiquement des lignes synthetiques au gold dataset, en respectant
EXACTEMENT son schema (15 colonnes) et en recalculant les features temporelles
(lags / rolling) via la vraie fonction du package -> aucune fuite, aucun NaN.

Usage typique
-------------
    # Un seul lot de 200 lignes (pour un test / la demo DVC) :
    uv run python scripts/generate_synthetic_gold.py --once

    # Flux continu : 200 lignes / minute pendant 5 minutes :
    uv run python scripts/generate_synthetic_gold.py --minutes 5

    # Simuler une derive capteur (+0.4 degC de moyenne par minute) :
    uv run python scripts/generate_synthetic_gold.py --minutes 10 --drift 0.4

Le label `panne` est SYNTHETIQUE (loi logistique correlee a la temperature et a
la pression), calibre autour du taux reel du gold (~10,5 %). Ce n'est donc pas
une jointure d'incidents : c'est un flux de demonstration pour le versioning.
"""

from __future__ import annotations  # annotations de type modernes

import argparse  # analyse les arguments de la ligne de commande (--once, --minutes…)
import signal  # pour intercepter Ctrl-C proprement (arrêt du flux continu)
import sys  # accès à sys.path (rendre le package importable) et au code de sortie
import time  # time.sleep() = attendre entre deux « ticks » (chaque minute)
from datetime import datetime  # horodatage des logs (heure courante)
from pathlib import Path  # chemins de fichiers portables

import numpy as np  # tirages aléatoires + calcul vectoriel (sigmoïde, bruit)
import pandas as pd  # DataFrame (lecture/écriture du gold, features)

# --- Bootstrap : rendre le package `decrochage` importable sans installation ----
# __file__ = ce script ; parents[1] = remonte de scripts/ vers la racine du repo.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"  # dossier des sources du package
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # on ajoute src/ au chemin d'import Python

# Import APRÈS le bootstrap. Le commentaire `noqa: E402` dit à ruff que c'est volontaire.
from decrochage.features.temporal import add_temporal_features  # noqa: E402

# Ordre EXACT des 15 colonnes du gold : on écrira toujours dans cet ordre.
GOLD_COLUMNS = [
    "machine",
    "timestamp",
    "temperature",
    "pressure_bar",
    "panne",
    "temperature_lag1",
    "temperature_lag3",
    "temperature_lag6",
    "temperature_roll3_mean",
    "temperature_roll6_mean",
    "pressure_bar_lag1",
    "pressure_bar_lag3",
    "pressure_bar_lag6",
    "pressure_bar_roll3_mean",
    "pressure_bar_roll6_mean",
]
HIST = 6  # lignes d'historique necessaires pour lag6 / roll6 (la plus longue mémoire)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Sigmoïde : transforme n'importe quel nombre réel en une probabilité [0, 1].
    return 1.0 / (1.0 + np.exp(-x))


def _panne_proba(temp: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Loi logistique : chaud + sous/sur-pression => proba de panne plus haute."""
    # « logit » = score linéaire ; les coefficients sont choisis pour que la
    # probabilité moyenne tourne autour de ~10 % (comme le vrai gold).
    # -2.75 = biais de base ; +0.18 par °C au-dessus de 52 ; +0.10 par bar au-dessus de 200.
    logit = -2.75 + 0.18 * (temp - 52.0) + 0.10 * (pressure - 200.0)
    return _sigmoid(logit)  # score → probabilité


class MachineState:
    """Etat par machine : derniere date + buffer d'historique + lignes de base."""

    def __init__(self, machine: str, tail: pd.DataFrame, rng: np.random.Generator):
        self.machine = machine  # identifiant de la machine (ex. MACH-01)
        self.rng = rng  # générateur aléatoire partagé (reproductible si --seed)
        self.last_ts: pd.Timestamp = tail["timestamp"].iloc[
            -1
        ]  # dernière date connue de cette machine
        # buffer minimal (machine, timestamp, temperature, pressure_bar) = les HIST
        # dernières lignes réelles, pour recalculer correctement lag/rolling.
        self.buffer = tail[["machine", "timestamp", "temperature", "pressure_bar"]].copy()
        self.temp_base = float(tail["temperature"].tail(HIST).mean())  # niveau moyen de température
        self.pres_base = float(tail["pressure_bar"].tail(HIST).mean())  # niveau moyen de pression
        self.temp_prev = float(tail["temperature"].iloc[-1])  # dernière température (départ AR(1))
        self.pres_prev = float(tail["pressure_bar"].iloc[-1])  # dernière pression (départ AR(1))

    def _next_value(self, prev: float, base: float, sigma: float, lo: float, hi: float) -> float:
        # AR(1) : la valeur suivante = base + 0.7×(écart précédent) + bruit gaussien.
        # → ça « colle » à la moyenne (réalisme) tout en variant un peu (bruit sigma).
        nxt = base + 0.7 * (prev - base) + self.rng.normal(0.0, sigma)
        return float(np.clip(nxt, lo, hi))  # on borne aux plages physiques plausibles (lo..hi)

    def generate(self, n: int, freq: pd.Timedelta, drift: float) -> pd.DataFrame:
        self.temp_base += drift  # derive eventuelle de la moyenne (simulate un capteur qui dérive)
        rows = []  # on accumule les nouvelles lignes BRUTES (avant features)
        for _ in range(n):  # produire n nouvelles mesures pour cette machine
            self.last_ts = self.last_ts + freq  # avance l'horloge (ex. +1 h)
            t = self._next_value(
                self.temp_prev, self.temp_base, 0.8, 40.0, 70.0
            )  # nouvelle température
            p = self._next_value(
                self.pres_prev, self.pres_base, 0.9, 190.0, 210.0
            )  # nouvelle pression
            self.temp_prev, self.pres_prev = t, p  # mémorise pour le prochain pas (AR(1))
            rows.append(
                (self.machine, self.last_ts, round(t, 2), round(p, 3))
            )  # arrondis « capteur »
        new = pd.DataFrame(rows, columns=["machine", "timestamp", "temperature", "pressure_bar"])
        # On colle l'historique (HIST lignes) DEVANT les nouvelles lignes : ainsi
        # add_temporal_features dispose du passé pour calculer lag/rolling sans NaN.
        ctx = pd.concat([self.buffer, new], ignore_index=True)
        feat = add_temporal_features(ctx)  # recalcule les 12 features temporelles
        out = feat.tail(n).reset_index(drop=True)  # on ne garde que les n NOUVELLES lignes
        # Label panne synthétique : tirage Bernoulli selon la proba logistique.
        proba = _panne_proba(out["temperature"].to_numpy(), out["pressure_bar"].to_numpy())
        out["panne"] = (self.rng.random(len(out)) < proba).astype(
            int
        )  # 1 si tirage < proba, sinon 0
        # Met à jour le buffer = les HIST dernières lignes (pour le prochain appel).
        self.buffer = ctx.tail(HIST).reset_index(drop=True)
        return out[GOLD_COLUMNS]  # renvoie dans l'ordre EXACT du gold


def _split_counts(total: int, n_machines: int) -> list[int]:
    # Répartit `total` lignes sur `n_machines` (ex. 200 sur 4 → [50, 50, 50, 50]).
    base, rem = divmod(total, n_machines)  # quotient + reste
    return [
        base + (1 if i < rem else 0) for i in range(n_machines)
    ]  # distribue le reste sur les premières


def main() -> int:
    # --- 1) Déclaration des options de la ligne de commande -------------------
    ap = argparse.ArgumentParser(description="Flux de donnees synthetiques -> gold dataset")
    ap.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data/gold/gold_dataset.csv",
        help="CSV gold a alimenter (defaut: data/gold/gold_dataset.csv)",
    )
    ap.add_argument(
        "--rows-per-minute",
        type=int,
        default=200,
        help="lignes ajoutees a chaque tick (defaut: 200)",
    )
    ap.add_argument(
        "--interval", type=float, default=60.0, help="secondes entre deux ticks (defaut: 60)"
    )
    ap.add_argument(
        "--minutes", type=float, default=None, help="duree totale en minutes (defaut: infini)"
    )
    ap.add_argument(
        "--ticks", type=int, default=None, help="nombre de ticks (prioritaire sur --minutes)"
    )
    ap.add_argument("--once", action="store_true", help="un seul tick puis sortie")
    ap.add_argument(
        "--freq",
        type=str,
        default="1h",
        help="pas de temps entre 2 mesures d'une machine (defaut: 1h)",
    )
    ap.add_argument(
        "--drift",
        type=float,
        default=0.0,
        help="derive temperature (+degC) ajoutee par tick (defaut: 0)",
    )
    ap.add_argument("--seed", type=int, default=None, help="graine aleatoire")
    ap.add_argument("--quiet", action="store_true", help="moins de logs")
    args = ap.parse_args()  # lit réellement les arguments fournis par l'utilisateur

    # --- 2) Garde-fou : le gold doit exister ---------------------------------
    if not args.gold.exists():
        ap.error(f"Gold introuvable : {args.gold} (lance d'abord `decrochage build-gold`)")

    # --- 3) Préparation : aléatoire, pas de temps, état par machine ----------
    rng = np.random.default_rng(args.seed)  # générateur (reproductible si --seed donné)
    freq = pd.Timedelta(args.freq)  # "1h" → objet durée pandas
    gold = pd.read_csv(args.gold, parse_dates=["timestamp"])  # lit le gold existant
    machines = sorted(gold["machine"].unique())  # liste triée des machines présentes
    # Un MachineState par machine, initialisé avec ses HIST dernières lignes réelles.
    states = {
        m: MachineState(m, gold[gold["machine"] == m].tail(HIST).reset_index(drop=True), rng)
        for m in machines
    }
    # Combien de lignes par machine à chaque tick (répartition du --rows-per-minute).
    counts = dict(zip(machines, _split_counts(args.rows_per_minute, len(machines)), strict=False))

    # --- 4) Combien de ticks faut-il faire ? ---------------------------------
    if args.once:  # --once = un seul lot
        total_ticks = 1
    elif args.ticks is not None:  # --ticks N = N lots
        total_ticks = args.ticks
    elif args.minutes is not None:  # --minutes M = M*60/interval lots
        total_ticks = max(1, int(round(args.minutes * 60.0 / args.interval)))
    else:
        total_ticks = None  # infini (jusqu'à Ctrl-C)

    # Gestion propre de Ctrl-C : on lève un drapeau au lieu de planter brutalement.
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    # Bandeau d'info au démarrage.
    print(
        f"[start] gold={args.gold.name} | machines={len(machines)} | "
        f"{args.rows_per_minute} lignes/tick | freq={args.freq} | "
        f"ticks={'inf' if total_ticks is None else total_ticks}"
    )

    # --- 5) Boucle principale : un « tick » = un lot ajouté ------------------
    grand_total = 0  # total de lignes ajoutées durant cette session
    tick = 0  # compteur de ticks
    while not stop["flag"]:
        tick += 1
        # Chaque machine produit son quota de nouvelles lignes (avec features + panne).
        batch = [states[m].generate(counts[m], freq, args.drift) for m in machines]
        new_rows = pd.concat(batch, ignore_index=True)  # rassemble toutes les machines
        # Écriture en mode APPEND (ajout), même ordre de colonnes, SANS réécrire l'en-tête.
        new_rows.to_csv(args.gold, mode="a", header=False, index=False)
        grand_total += len(new_rows)
        if not args.quiet:  # logs (sauf si --quiet)
            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{now}] tick {tick} | +{len(new_rows)} lignes "
                f"(total session {grand_total}) | panne_lot={new_rows['panne'].mean():.3f} "
                f"| derniere_date={new_rows['timestamp'].max()}"
            )
        if total_ticks is not None and tick >= total_ticks:  # nombre de ticks atteint ?
            break
        if stop["flag"]:  # Ctrl-C reçu ?
            break
        time.sleep(args.interval)  # attend avant le tick suivant (débit réel)

    # --- 6) Bilan final ------------------------------------------------------
    # Recompte les lignes du fichier (− 1 pour l'en-tête) pour afficher le total réel.
    final = sum(1 for _ in open(args.gold, encoding="utf-8")) - 1
    print(f"[done] +{grand_total} lignes ajoutees | gold = {final} lignes au total")
    return 0  # code de sortie 0 = succès


if __name__ == "__main__":
    # raise SystemExit(...) propage le code de retour de main() au shell (0 = OK).
    raise SystemExit(main())
