from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def catalogue_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "raw" / "dataset catalogue_formations_V5.csv")


@pytest.fixture
def gold_df() -> pd.DataFrame:
    """Echantillon du vrai dataset gold (rapide a charger, structure realiste)."""
    df = pd.read_csv(ROOT / "data" / "gold" / "gold-dataset.csv", sep=";")
    return df.sample(n=500, random_state=0).reset_index(drop=True)
