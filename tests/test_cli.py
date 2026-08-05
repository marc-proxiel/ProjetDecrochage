import json
from pathlib import Path

from typer.testing import CliRunner

from decrochage.cli import app, build_gold_dataset

runner = CliRunner()

RAW_DIR = Path("data/raw")


def test_build_gold_dataset_matches_expected_shape():
    df = build_gold_dataset(RAW_DIR)
    assert df.shape[0] > 5000
    assert "abandon" in df.columns
    assert "moyenne_finale" in df.columns
    assert "student_id" not in df.columns  # identifiant, retire
    assert "nb_devoirs_total" not in df.columns  # retire (cf. logique taux_rendu)


def test_check_data_command_runs():
    result = runner.invoke(app, ["check-data"])
    assert result.exit_code == 0
    assert "lignes" in result.output


def test_build_gold_command_writes_file(tmp_path):
    gold_dir = tmp_path / "gold"
    result = runner.invoke(app, ["build-gold", "--gold-dir", str(gold_dir)])
    assert result.exit_code == 0
    assert (gold_dir / "gold-dataset.csv").exists()


def test_train_command_creates_both_models(tmp_path):
    gold_dir = tmp_path / "gold"
    model_dir = tmp_path / "models"
    runner.invoke(app, ["build-gold", "--gold-dir", str(gold_dir)], catch_exceptions=False)

    result = runner.invoke(
        app,
        ["train", "--gold-dir", str(gold_dir), "--model-dir", str(model_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert (model_dir / "logistic.joblib").exists()
    assert (model_dir / "regressor.joblib").exists()
    assert (model_dir / "imputer.joblib").exists()

    meta = json.loads((model_dir / "model_metadata.json").read_text())
    assert "metrics_holdout" in meta
    assert "metrics_holdout_regression" in meta


def test_predict_command_against_shipped_model():
    """Utilise le modele et le gold deja versionnes dans le depot (pas de tmp_path)."""
    result = runner.invoke(app, ["predict", "data/gold/gold-dataset-test.csv"])
    assert result.exit_code == 0
    assert "proba_abandon=" in result.output
    assert "moyenne_predite=" in result.output
