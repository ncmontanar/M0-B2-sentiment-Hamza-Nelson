"""API d'analyse du sentiment des avis clients — Escale Atlantique (mission FastIA)."""

import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from loguru import logger

from app.inference import load_pipeline, predict_sentiment
from app.schemas import ReviewIn, SentimentOut

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = os.getenv("MODEL_NAME", "cmarkea/distilcamembert-base-sentiment")


# 4.1 Chemin vers le fichier de log
LOG_PATH = ROOT_DIR / "logs" / "api.log"
LOG_TEXT_LIMIT = 80

# Objets partagés par toutes les requêtes (ici : le pipeline de classification)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 4.2 Configuration du logger avec Loguru
    sink_id = logger.add(
        LOG_PATH,
        level="INFO",
        rotation="5 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        diagnose=False,
    )
    try:
        # Au 1er lancement, le modèle est téléchargé (≈ 270 Mo) : ça peut prendre quelques minutes
        logger.info("Chargement du modèle {}…", MODEL_NAME)
        state["classifier"] = load_pipeline(MODEL_NAME)
        logger.info("Modèle chargé")
        yield
    finally:
        # Exécuté à l'arrêt du serveur
        state.clear()
        # 4.3 Journalisation de l'arrêt de l'application
        logger.info("Arrêt de l'application")
        logger.remove(sink_id)


app = FastAPI(
    title="Escale Atlantique — Sentiment des avis clients",
    description="Classe un avis client en négatif, neutre ou positif.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_prediction_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.method != "POST" or request.url.path != "/predict":
        return await call_next(request)

    start = perf_counter()
    request.state.log_text = None
    request.state.log_sentiment = "indisponible"
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (perf_counter() - start) * 1000
        if status_code >= 500:
            level = "ERROR"
        elif status_code >= 400:
            level = "WARNING"
        else:
            level = "INFO"
        logger.log(
            level,
            "POST /predict | texte={!r} | sentiment={} | statut={} | duree_ms={:.2f}",
            request.state.log_text,
            request.state.log_sentiment,
            status_code,
            duration_ms,
        )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> Response:
    # Une réponse 422 est produite avant l'exécution de predict().
    if request.method == "POST" and request.url.path == "/predict":
        if isinstance(exc.body, dict):
            text = exc.body.get("text")
            if isinstance(text, str):
                request.state.log_text = text[:LOG_TEXT_LIMIT]
    return await request_validation_exception_handler(request, exc)


@app.get("/health")
def health() -> dict:
    """Le service est-il en vie, et le modèle chargé ?"""
    return {"status": "ok", "model_loaded": "classifier" in state}


@app.get("/info")
def info() -> dict:
    """Quel modèle tourne, et que renvoie l'API ?"""
    return {
        "model_name": MODEL_NAME,
        "sortie_modele": "1 à 5 étoiles",
        "sortie_api": ["négatif", "neutre", "positif"],
    }


@app.post("/predict", response_model=SentimentOut)
def predict(review: ReviewIn, request: Request) -> SentimentOut:
    """Analyse le sentiment d'un avis : classe métier, scores des 5 étoiles et latence."""
    request.state.log_text = review.text[:LOG_TEXT_LIMIT]
    model = state.get("classifier")
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé")

    result = predict_sentiment(model, review.text)
    response = SentimentOut(**result)
    request.state.log_sentiment = response.sentiment
    return response