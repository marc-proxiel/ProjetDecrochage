import pandas as pd
import pytest

from decrochage.storage import enregistrer_scores, ensure_schema, get_engine, lire_derniers_scores


@pytest.fixture
def engine():
    """SQLite en memoire : meme schema/API que PostgreSQL, sans serveur externe."""
    eng = get_engine("sqlite://")
    ensure_schema(eng)
    return eng


def _scores(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_ref": [f"etudiant_{i}" for i in range(n)],
            "proba_abandon": [0.1 * i for i in range(n)],
            "decision": ["a_risque" if i % 2 else "ok" for i in range(n)],
            "moyenne_predite": [15.0 - i for i in range(n)],
            "model_version": ["0.1.0"] * n,
            "threshold": [0.5] * n,
        }
    )


def test_enregistrer_scores_insere_toutes_les_lignes(engine):
    n = enregistrer_scores(engine, _scores(3))
    assert n == 3
    df = lire_derniers_scores(engine, n=10)
    assert len(df) == 3
    assert "scored_at" in df.columns


def test_enregistrer_scores_est_un_historique_pas_un_ecrasement(engine):
    enregistrer_scores(engine, _scores(3))
    enregistrer_scores(engine, _scores(3))
    df = lire_derniers_scores(engine, n=100)
    assert len(df) == 6  # les deux lots s'additionnent, rien n'est ecrase


def test_ensure_schema_est_idempotent(engine):
    ensure_schema(engine)  # ne doit pas lever meme si la table existe deja
    enregistrer_scores(engine, _scores(1))
    assert len(lire_derniers_scores(engine, n=10)) == 1


def test_lire_derniers_scores_respecte_la_limite(engine):
    enregistrer_scores(engine, _scores(5))
    df = lire_derniers_scores(engine, n=2)
    assert len(df) == 2
