# =============================================================================
#  src/decrochage/storage.py  —  STOCKAGE BATCH des scores (PostgreSQL)
# -----------------------------------------------------------------------------
#  Couche "orchestration applicative" de l'architecture cible : au lieu de
#  recalculer une prediction a chaque consultation, un job periodique
#  (`flows/pipeline.py`) score les etudiants en batch et ECRIT le resultat ici.
#  Avantages : historisation (suivre un etudiant dans le temps), et l'API/UI
#  metier n'a plus qu'a LIRE cette table, pas a rappeler les modeles.
#
#  Utilise SQLAlchemy Core (pas l'ORM) : suffisant pour une seule table, et ça
#  reste portable entre PostgreSQL (production, cf. docker-compose.yml) et
#  SQLite (tests, sans dependance a un serveur PostgreSQL demarre).
# =============================================================================
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine

METADATA = MetaData()

PREDICTIONS = Table(
    "predictions",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("scored_at", DateTime(timezone=True), nullable=False),
    Column("student_ref", String, nullable=False),
    Column("proba_abandon", Float, nullable=False),
    Column("decision", String, nullable=False),
    Column("moyenne_predite", Float, nullable=True),
    Column("model_version", String, nullable=False),
    Column("threshold", Float, nullable=False),
)


def get_engine(db_url: str) -> Engine:
    return create_engine(db_url)


def ensure_schema(engine: Engine) -> None:
    """Cree la table `predictions` si elle n'existe pas encore (idempotent)."""
    METADATA.create_all(engine, tables=[PREDICTIONS])


def enregistrer_scores(engine: Engine, df_scores: pd.DataFrame) -> int:
    """Ajoute les scores en base (toujours en INSERT, jamais d'ecrasement : on
    garde l'historique complet pour pouvoir suivre un etudiant dans le temps).

    `df_scores` doit contenir : student_ref, proba_abandon, decision,
    moyenne_predite, model_version, threshold. `scored_at` est ajoute ici.
    """
    df = df_scores.copy()
    df["scored_at"] = datetime.now(UTC)
    df.to_sql("predictions", engine, if_exists="append", index=False)
    return len(df)


def lire_derniers_scores(engine: Engine, n: int = 50) -> pd.DataFrame:
    """Relit les n derniers scores (verification / debug / usage par une future UI)."""
    query = text("SELECT * FROM predictions ORDER BY scored_at DESC LIMIT :n")
    return pd.read_sql(query, engine, params={"n": n})
