# =============================================================================
#  src/decrochage/data/loaders.py  —  CHARGEMENT & NETTOYAGE des données brutes
# -----------------------------------------------------------------------------
#  Les données arrivent « sales » et hétérogènes, comme dans la vraie vie :
#    • température : CSV séparé par des  ;   (point-virgule)
#    • pression    : TSV séparé par des  \t  (tabulation), dates avec fuseau horaire
#    • incidents   : CSV classique  ,  avec une colonne date + une colonne time
#    • machines    : un script SQL (INSERT ...) qu'on lit à la regex
#  …et les identifiants machine sont écrits de 10 façons (MACH-01, M_07, M-2…).
#
#  Rôle du fichier : tout uniformiser, puis FABRIQUER le dataset d'entraînement
#  (`build_dataset`) avec la cible `panne`. La fonction clé est `build_dataset`
#  et son `merge_asof(by="machine")` (jointure temporelle SANS mélanger les machines).
# =============================================================================
from __future__ import annotations  # annotations de type modernes

import re  # expressions régulières (regex) : reconnaître des motifs dans du texte
from pathlib import Path  # chemins de fichiers portables

import pandas as pd  # manipulation de tableaux
from loguru import logger  # journalisation (logs lisibles) — mieux que des print()

# Regex qui capture le PREMIER groupe de chiffres d'une chaîne. `\d+` = un ou
# plusieurs chiffres ; les parenthèses « capturent » ce nombre pour le réutiliser.
_DIGITS = re.compile(r"(\d+)")

# Regex qui reconnaît une ligne d'INSERT SQL de machine :
#   ('MACH-01', '2024-01-15', 1200, 'Modele', 'LigneA', 'Atelier1', 'HIGH')
# Chaque ('...') ou nombre est capturé dans un groupe (group(1) … group(7)).
_MACHINE_ROW = re.compile(
    r"\('([^']+)',\s*'([^']+)',\s*(\d+),\s*'([^']+)',\s*'([^']+)',\s*'([^']+)',\s*'([^']+)'\)"
)

# Échelle ORDINALE de criticité : on transforme un texte en nombre ordonné
# (LOW < MEDIUM < HIGH) pour pouvoir le comparer/trier numériquement.
CRITICALITY_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def normalize_machine_id(raw: str) -> str:
    """Normalize raw machine identifiers to MACH-0N."""
    # On cherche le nombre présent dans l'identifiant, quel que soit son format.
    match = _DIGITS.search(str(raw))  # str(raw) : on tolère aussi un nombre déjà numérique
    if not match:  # aucun chiffre trouvé (ex. "NOPE") → identifiant invalide
        raise ValueError(f"machine_id sans numero : {raw!r}")  # on échoue clairement
    # On reformate TOUJOURS pareil : "MACH-" + numéro sur 2 chiffres (02d = padding à 2 zéros).
    # Ainsi "M-2", "MACH_02", "M_2" deviennent tous "MACH-02" → plus de doublons cachés.
    return f"MACH-{int(match.group(1)):02d}"


def load_temperature(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")  # fichier température : séparateur point-virgule
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # texte → vrai objet date/heure (comparable)
    df["machine"] = df["machine_id"].map(normalize_machine_id)  # uniformise l'identifiant machine
    return df[["machine", "timestamp", "temperature"]]  # ne garde que les 3 colonnes utiles


def load_pressure(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")  # fichier pression : séparateur TABULATION (TSV)
    # Les dates de pression portent un fuseau horaire (« tz-aware »). On les lit
    # (format="mixed" = formats variés tolérés, utc=True), PUIS on enlève le fuseau
    # (tz_localize(None)) pour qu'elles soient comparables avec les autres capteurs.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
    df["machine"] = df["machine_id"].map(normalize_machine_id)
    return df[["machine", "timestamp", "pressure_bar"]]


def load_incidents(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)  # incidents : CSV classique (séparateur virgule)
    df["machine"] = df["machine_id"].map(normalize_machine_id)
    # La date et l'heure de l'incident sont dans DEUX colonnes séparées : on les
    # concatène ("2025-09-01" + " " + "14:30") puis on convertit en date/heure.
    df["incident_ts"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    # On ne garde QUE machine + horodatage de l'incident. Les autres colonnes
    # (sévérité, opérateur, commentaire…) sont volontairement écartées : ce sont
    # des infos du futur/contextuelles → jamais des features (sinon = fuite).
    return df[["machine", "incident_ts"]]


def load_machines(path: Path) -> pd.DataFrame:
    text = Path(path).read_text(encoding="utf-8")  # lit tout le script SQL comme du texte
    rows = [  # liste en compréhension : 1 dict par machine trouvée
        {
            "machine": normalize_machine_id(match.group(1)),  # 1er groupe capturé = identifiant
            "commissioning_date": pd.to_datetime(match.group(2)),  # date de mise en service
            "max_daily_capacity": int(match.group(3)),  # capacité (nombre → int)
            "model": match.group(4),  # modèle de machine
            "production_line": match.group(5),  # ligne de production
            "location": match.group(6),  # emplacement
            "criticality": match.group(7),  # criticité (LOW/MEDIUM/HIGH)
        }
        for match in _MACHINE_ROW.finditer(text)  # parcourt toutes les lignes INSERT du SQL
    ]
    if not rows:  # aucune machine reconnue → fichier vide/mal formé
        raise ValueError(f"Aucune machine trouvee dans {path}")
    return pd.DataFrame(rows)  # transforme la liste de dicts en tableau


def build_dataset(
    temp: pd.DataFrame,  # mesures de température (machine, timestamp, temperature)
    pres: pd.DataFrame,  # mesures de pression   (machine, timestamp, pressure_bar)
    inc: pd.DataFrame,  # incidents             (machine, incident_ts)
    window_hours: int = 24,  # on étiquette « panne » les N heures AVANT chaque incident
    tolerance_minutes: int = 90,  # écart max toléré pour apparier température et pression
) -> pd.DataFrame:
    """Join sensors and derive binary target `panne`.

    The `by="machine"` parameter is critical: without it, one machine can inherit
    the pressure value of another machine.
    """
    # merge_asof exige que les deux tables soient triées par la clé temporelle.
    temp = temp.sort_values("timestamp")
    pres = pres.sort_values("timestamp")
    # JOINTURE TEMPORELLE « approximative » : pour chaque mesure de température,
    # on attrape la mesure de pression la PLUS PROCHE dans le temps (direction="nearest"),
    # mais uniquement de LA MÊME machine (by="machine" → garde-fou anti-mélange),
    # et seulement si elle tombe à ±tolerance_minutes (sinon : pas d'appariement).
    sensors = pd.merge_asof(
        temp,
        pres,
        on="timestamp",  # clé d'alignement = le temps
        by="machine",  # CRUCIAL : on n'apparie jamais 2 machines différentes
        direction="nearest",  # pression la plus proche (avant OU après)
        tolerance=pd.Timedelta(minutes=tolerance_minutes),  # au-delà de ±90 min → laisse vide (NaN)
    )
    before = len(sensors)  # nombre de lignes avant nettoyage
    sensors = sensors.dropna(
        subset=["pressure_bar"]
    )  # retire les températures sans pression appariée
    dropped = before - len(sensors)  # combien de lignes supprimées ?
    if dropped:  # s'il y en a, on le TRACE (transparence/qualité données)
        logger.info(
            "merge_asof: {} rows without pressure under +/-{} min dropped ({:.2%})",
            dropped,
            tolerance_minutes,
            dropped / before,  # proportion supprimée (formatée en %)
        )

    # On retrie proprement (machine puis temps) et on renumérote les lignes.
    sensors = sensors.sort_values(["machine", "timestamp"]).reset_index(drop=True)

    # --- Construction de la CIBLE `panne` ------------------------------------
    sensors["panne"] = 0  # par défaut : tout est « sain » (0)
    window = pd.Timedelta(hours=window_hours)  # durée de la fenêtre d'alerte (ex. 24 h)
    for row in inc.itertuples():  # pour chaque incident connu…
        # …on marque « panne = 1 » toutes les mesures de CETTE machine situées
        # dans la fenêtre [incident - 24 h ; incident]. Idée métier : « les heures
        # qui précèdent une panne portent déjà des signes annonciateurs ».
        mask = (
            (sensors["machine"] == row.machine)  # même machine
            & (sensors["timestamp"] >= row.incident_ts - window)  # après (incident − 24 h)
            & (sensors["timestamp"] <= row.incident_ts)  # et jusqu'à l'incident
        )
        sensors.loc[mask, "panne"] = 1  # applique l'étiquette 1 aux lignes ciblées
    return sensors  # tableau capteurs + colonne cible `panne`


def add_machine_criticality(df: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """Add static criticality for monitoring or segmentation, not baseline training."""
    # Pour chaque ligne, on va chercher la criticité de sa machine (table `machines`),
    # puis on la convertit en nombre via CRITICALITY_ORDER (LOW=0, MEDIUM=1, HIGH=2).
    levels = df["machine"].map(machines.set_index("machine")["criticality"].map(CRITICALITY_ORDER))
    out = df.copy()  # copie pour ne pas modifier l'entrée
    # Si une machine est inconnue (NaN), on retombe sur MEDIUM par défaut, puis on
    # force le type entier. NB : cette feature sert au MONITORING/segmentation,
    # pas au modèle de base (commentaire de la docstring).
    out["criticality_level"] = levels.fillna(CRITICALITY_ORDER["MEDIUM"]).astype("int64")
    return out
