"""FastAPI app: prediction API, web frontend and Prometheus metrics."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict
from starlette.routing import Match

from app.model import FishPredictor, get_model_path
from app.preprocessing import SPECIES, PreprocessingError, preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marine-fish-classifier-api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 10 * 1024  # 10 KB is plenty for five numbers

# ---------------------------------------------------------------------------
# Prometheus metrics (scraped from GET /metrics)
# ---------------------------------------------------------------------------
REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests, by method, endpoint and status code",
    ["method", "endpoint", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter(
    "predictions_total", "Successful predictions, by predicted species", ["species"]
)
CONFIDENCE = Histogram(
    "prediction_confidence",
    "Model confidence for the predicted species",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Rejected prediction requests, by reason", ["reason"]
)
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded, otherwise 0")

ERROR_REASONS = (
    "invalid_input",
    "validation_error",
    "unsupported_type",
    "too_large",
    "model_unavailable",
)

# Create every label up front so dashboards show 0 instead of "no data".
for _species in SPECIES:
    PREDICTIONS.labels(species=_species)
for _reason in ERROR_REASONS:
    PREDICTION_ERRORS.labels(reason=_reason)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = FishPredictor.from_path(get_model_path())
        MODEL_LOADED.set(1)
        logger.info("Model loaded.")
    except Exception:
        logger.exception("Model failed to load; /predict will return 503.")
        app.state.predictor = None
        MODEL_LOADED.set(0)
    yield


app = FastAPI(
    title="Marine Fish Classifier",
    description="Predicts a fish's species from its body measurements.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _endpoint_label(request: Request) -> str:
    """Route template (e.g. "/predict"), so random URLs don't create new series."""
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", "other")
    return "unmatched"


@app.middleware("http")
async def track_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    endpoint = _endpoint_label(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(
            method=request.method, endpoint=endpoint, status=str(status)
        ).inc()


@app.exception_handler(RequestValidationError)
async def count_validation_errors(request: Request, exc: RequestValidationError):
    """Missing fields / wrong types -> 422, and count it for the dashboard."""
    if request.url.path == "/predict":
        PREDICTION_ERRORS.labels(reason="validation_error").inc()
    return await request_validation_exception_handler(request, exc)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class FishMeasurements(BaseModel):
    length_cm: float
    weight_kg: float
    height_cm: float
    fin_count: int
    water_temp_c: float

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "length_cm": 75, "weight_kg": 4.5, "height_cm": 15,
            "fin_count": 8, "water_temp_c": 10,
        }]
    })


class PredictionResponse(BaseModel):
    species: str
    confidence: float
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Dependencies (run before the body is validated; tests can override them)
# ---------------------------------------------------------------------------
def require_json(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        PREDICTION_ERRORS.labels(reason="unsupported_type").inc()
        raise HTTPException(
            status_code=415, detail="Send JSON with Content-Type: application/json."
        )
    try:
        size = int(request.headers.get("content-length", "0"))
    except ValueError:
        size = 0
    if size > MAX_BODY_BYTES:
        PREDICTION_ERRORS.labels(reason="too_large").inc()
        raise HTTPException(status_code=413, detail="Request body is larger than 10 KB.")


def get_predictor(request: Request) -> FishPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        PREDICTION_ERRORS.labels(reason="model_unavailable").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return predictor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    loaded = getattr(request.app.state, "predictor", None) is not None
    return {"status": "ok", "model_loaded": loaded}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    fish: FishMeasurements,
    _json: None = Depends(require_json),
    predictor: FishPredictor = Depends(get_predictor),
):
    try:
        batch = preprocess(fish.model_dump())
    except PreprocessingError as exc:
        PREDICTION_ERRORS.labels(reason="invalid_input").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probs = predictor.predict(batch)
    best = int(probs.argmax())
    species = predictor.classes[best]
    confidence = float(probs[best])

    PREDICTIONS.labels(species=species).inc()
    CONFIDENCE.observe(confidence)

    return PredictionResponse(
        species=species,
        confidence=round(confidence, 4),
        probabilities={c: round(float(p), 4) for c, p in zip(predictor.classes, probs)},
    )
