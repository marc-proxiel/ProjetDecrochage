import pytest
from fastapi.testclient import TestClient

from decrochage.api.main import app
from decrochage.api.security import MAX_BODY_BYTES

PAYLOAD = {
    "filiere": "Informatique",
    "age": 19,
    "bac_type": "general",
    "mention_bac": "Bien",
    "etablissement_origine": "lycee_public",
    "distance_domicile_km": 12.5,
    "heures_travail_remunere_sem": 5.0,
    "taux_presence_pct": 82.0,
    "connexions_lms_30j": 30.0,
    "heures_lms_total": 42.0,
    "ressources_consultees": 60,
    "retards_rendus": 1,
    "nb_devoirs_rendus": 9,
    "messages_forum": 3,
    "motivation": 3.0,
    "satisfaction": 4.0,
    "sentiment_appartenance": 3.0,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_predict_tabular_requires_api_key(client):
    r = client.post("/predict-tabular", json=PAYLOAD)
    assert r.status_code == 401


def test_predict_tabular_success(client):
    r = client.post("/predict-tabular", json=PAYLOAD, headers={"X-API-Key": "dev-key"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["proba_abandon"] <= 1.0
    assert body["decision"] in {"ok", "a_risque"}
    assert body["moyenne_predite"] is not None


def test_predict_tabular_unknown_filiere_is_422(client):
    payload = {**PAYLOAD, "filiere": "Philosophie"}
    r = client.post("/predict-tabular", json=payload, headers={"X-API-Key": "dev-key"})
    assert r.status_code == 422


def test_predict_tabular_missing_required_field_is_422(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "taux_presence_pct"}
    r = client.post("/predict-tabular", json=payload, headers={"X-API-Key": "dev-key"})
    assert r.status_code == 422


def test_predict_tabular_body_too_large_is_413(client):
    payload = {**PAYLOAD, "filler": "x" * (MAX_BODY_BYTES + 1000)}
    r = client.post("/predict-tabular", json=payload, headers={"X-API-Key": "dev-key"})
    assert r.status_code == 413


def test_predict_image_rejects_empty_file(client):
    r = client.post(
        "/predict-image",
        headers={"X-API-Key": "dev-key"},
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 422


def test_predict_image_accepts_valid_image(client):
    r = client.post(
        "/predict-image",
        headers={"X-API-Key": "dev-key"},
        files={"file": ("photo.png", b"contenu-binaire", "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["decision"] == "ok"
