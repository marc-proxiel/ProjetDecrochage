# =============================================================================
#  src/decrochage/features/temporal.py  —  FEATURES TEMPORELLES (sans fuite)
# -----------------------------------------------------------------------------
#  On fabrique deux familles de variables à partir des séries de capteurs :
#    • lag (retard)      : la valeur d'il y a N pas de temps   → ex. température_lag3
#    • rolling (glissant): la moyenne des N derniers pas        → ex. température_roll6_mean
#
#  RÈGLE D'OR ANTI-FUITE (« data leakage ») : pour prédire l'instant t, on n'a le
#  droit d'utiliser QUE des informations connues AVANT t. Si une feature incluait
#  la valeur de l'instant t lui-même, le modèle « tricherait » : excellent en test,
#  catastrophique en production. Tout ce fichier est construit pour éviter ça.
# =============================================================================
from __future__ import annotations  # autorise les annotations de type modernes (PEP 563)

import pandas as pd  # pandas = manipulation de tableaux de données (DataFrame)


def add_temporal_features(
    df: pd.DataFrame,  # tableau d'entrée (1 ligne = 1 mesure)
    group_col: str = "machine",  # on calcule SÉPARÉMENT par machine
    timestamp_col: str = "timestamp",  # colonne de date/heure (pour ordonner)
    value_cols: tuple[str, ...] = ("temperature", "pressure_bar"),  # colonnes sources à dériver
    lags: tuple[int, ...] = (1, 3, 6),  # retards voulus : t-1, t-3, t-6
    windows: tuple[int, ...] = (3, 6),  # tailles des fenêtres glissantes : 3 et 6 pas
) -> pd.DataFrame:  # renvoie le tableau enrichi de colonnes
    """Add lag and rolling features per machine without temporal leakage."""
    # --- 1) Garde-fou : vérifier que les colonnes nécessaires existent -------
    required = {group_col, timestamp_col, *value_cols}  # ensemble des colonnes indispensables
    missing = required - set(df.columns)  # celles qui manquent (différence d'ensembles)
    if missing:  # s'il en manque au moins une…
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing)}"
        )  # …on échoue tôt, avec un message clair

    # --- 2) Trier par machine puis par temps (ordre chronologique) -----------
    # Indispensable : un « lag » n'a de sens que si les lignes sont dans l'ordre du temps,
    # et séparées par machine. `.reset_index(drop=True)` renumérote proprement ; `.copy()`
    # évite de modifier le tableau d'origine de l'appelant (effet de bord).
    df = df.sort_values([group_col, timestamp_col]).reset_index(drop=True).copy()

    # --- 3) Pour chaque colonne source, créer ses features -------------------
    for col in value_cols:
        # `groupby(machine)[col]` = on raisonne machine par machine : le passé d'une
        # machine ne « déborde » jamais sur une autre (pas de mélange entre machines).
        grouped = df.groupby(group_col)[col]

        # 3a) FEATURES DE RETARD (lag) : la valeur d'il y a `lag` pas de temps.
        for lag in lags:
            # shift(lag) décale vers le bas → la 1re ligne de chaque machine vaut NaN
            # (pas de passé disponible). C'est NORMAL et voulu (on ne devine pas le passé).
            df[f"{col}_lag{lag}"] = grouped.shift(lag)

        # 3b) FEATURES DE MOYENNE GLISSANTE (rolling) : moyenne des `window` derniers pas.
        for window in windows:
            df[f"{col}_roll{window}_mean"] = grouped.transform(
                # POINT CLÉ ANTI-FUITE : `shift(1)` AVANT `rolling(window)` →
                # la moyenne porte sur les valeurs STRICTEMENT antérieures à t
                # (elle n'inclut JAMAIS la valeur courante).
                # `window=window` en argument par défaut = capture la bonne valeur de
                # `window` à chaque tour de boucle (évite le piège de « late binding »
                # des lambdas Python, qui sinon prendraient toutes la dernière valeur).
                lambda series, window=window: series.shift(1)
                .rolling(window)
                .mean()
            )
    # --- 4) Renvoyer le tableau enrichi (colonnes _lagN et _rollN_mean ajoutées)
    return df
