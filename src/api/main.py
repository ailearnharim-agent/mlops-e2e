"""FastAPI serving application for the Heart Disease risk model.

Endpoints:
    GET  /health   - liveness/readiness probe, reports whether the model loaded
    POST /predict  - JSON patient features -> prediction + confidence
    GET  /metrics  - Prometheus exposition format
"""

import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.logging_config import configure_logging, log_with_fields
from src.api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.config import MODEL_PATH
from src.models.predict import load_model, predict_one

APP_VERSION = "1.0.0"

logger = configure_logging()

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts presence of heart disease from patient clinical features.",
    version=APP_VERSION,
)

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made, by predicted class", ["predicted_class"]
)

_model_state = {"loaded": False, "version": "unknown"}


@app.on_event("startup")
def _load_model_on_startup() -> None:
    try:
        model = load_model(MODEL_PATH)
        clf_name = model.named_steps["clf"].__class__.__name__
        _model_state["loaded"] = True
        _model_state["version"] = f"{clf_name}-{APP_VERSION}"
        log_with_fields(
            logger, logging.INFO, "model_loaded", model_class=clf_name, model_path=str(MODEL_PATH)
        )
    except Exception as exc:  # noqa: BLE001 - want the API to still start and report unhealthy
        log_with_fields(logger, logging.ERROR, "model_load_failed", error=str(exc))


@app.middleware("http")
async def request_logging_and_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    log_with_fields(
        logger,
        logging.INFO,
        "request_handled",
        method=request.method,
        path=endpoint,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
        client=request.client.host if request.client else None,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model_state["loaded"] else "degraded", model_loaded=_model_state["loaded"]
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    prediction, confidence = predict_one(features.model_dump(), model_path=MODEL_PATH)
    label = "Disease present" if prediction == 1 else "No disease"
    PREDICTION_COUNT.labels(predicted_class=str(prediction)).inc()
    log_with_fields(
        logger,
        logging.INFO,
        "prediction_made",
        prediction=prediction,
        confidence=round(confidence, 4),
    )
    return PredictionResponse(
        prediction=prediction, label=label, confidence=confidence, model_version=_model_state["version"]
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log_with_fields(logger, logging.ERROR, "unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
