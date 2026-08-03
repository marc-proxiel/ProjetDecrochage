# =============================================================================
#  src/decrochage/api/main.py  —  L'APPLICATION FastAPI (le coeur de l'API)
# -----------------------------------------------------------------------------
#  Routes :
#    - GET  /health          : « le serveur est-il VIVANT ? » (liveness)
#    - GET  /ready           : « le serveur est-il PRET a servir ? » (readiness)
#    - POST /predict-tabular : probabilite de decrochage a partir du releve
#                              mi-semestre d'un etudiant (voir schemas.py)
#    - GET  /metrics         : metriques Prometheus (auto)
#
#  NOTE : ce fichier remplace une version anterieure du starter (maintenance
#  predictive industrielle, relevés capteurs par machine) qui ne correspondait
#  plus au cas d'usage actuel du projet (decrochage etudiant).
# =============================================================================
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, status
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

import decrochage.api.model_store as store
from decrochage.api.model_store import ModelBundle, get_model_bundle
from decrochage.api.schemas import PredictionResponse, StudentFeatures
from decrochage.api.security import limit_body_size, rate_limit
from decrochage.config import settings
from decrochage.models.tabular import (
    COLONNES_A_IMPUTER,
    aligner_colonnes,
    encoder_categorielles,
    fusionner_catalogue,
    predict_proba,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store._BUNDLE = store.load_bundle(
            settings.model_dir, settings.data_dir, settings.decision_threshold
        )
        logger.info("Modele charge")
    except FileNotFoundError:
        store._BUNDLE = None
        logger.warning("Aucun modele — /ready renverra 503 (lancer `decrochage train` d'abord)")
    yield


app = FastAPI(title="Decrochage API", version="0.1.0", lifespan=lifespan)

app.middleware("http")(limit_body_size)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    with logger.contextualize(request_id=rid):
        response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    if x_api_key is None or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Cle API absente ou invalide"
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready(bundle: ModelBundle | None = Depends(get_model_bundle)) -> dict:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    return {"status": "ready", "model_version": bundle.version}


@app.post(
    "/predict-tabular",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def predict_tabular(
    payload: StudentFeatures,
    bundle: ModelBundle | None = Depends(get_model_bundle),
) -> PredictionResponse:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    # Une seule ligne : meme pipeline de preparation que build-gold, mais sans
    # les etapes propres au CSV brut (dedoublonnage, parsing de dates...).
    df = pd.DataFrame([payload.model_dump()])
    df = fusionner_catalogue(df, bundle.catalogue)
    if df["faculte"].isna().any():
        raise HTTPException(status_code=422, detail=f"Filiere inconnue : {payload.filiere!r}")

    df = encoder_categorielles(df)
    df = aligner_colonnes(df, bundle.feature_columns)

    colonnes_a_imputer = [c for c in COLONNES_A_IMPUTER if c in df.columns]
    df[colonnes_a_imputer] = bundle.imputer.transform(df[colonnes_a_imputer])

    # `df` a exactement les colonnes vues a l'entrainement (aligner_colonnes) :
    # pas besoin de select_features ici.
    proba = float(predict_proba(bundle.model, df)[0])

    return PredictionResponse(
        proba_abandon=proba,
        decision="a_risque" if proba >= bundle.threshold else "ok",
        model_version=bundle.version,
        threshold=bundle.threshold,
    )


@app.post("/predict-image", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def predict_image(
    file: UploadFile,
    bundle: ModelBundle | None = Depends(get_model_bundle),
) -> dict:
    """Vision (intermediaire) : validation reelle du fichier ; a brancher plus tard."""
    content = await file.read()

    if not content:
        raise HTTPException(status_code=422, detail="Fichier image vide")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Le fichier n'est pas une image")

    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "anomaly_score": 0.0,
        "decision": "ok",
    }