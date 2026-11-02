"""API tests.

Unit tests swap in a fake model so they run fast and don't depend on the
trained model. The two tests at the bottom use the real model and are
skipped automatically if it (or the processed data) is missing.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_predictor
from app.model import get_model_path
from app.preprocessing import MODEL_FEATURES, RAW_FEATURES, SPECIES

SALMON = {"length_cm": 75, "weight_kg": 4.5, "height_cm": 15, "fin_count": 8, "water_temp_c": 10}


class FakePredictor:
    classes = SPECIES

    def predict(self, batch):
        assert batch.shape == (1, len(MODEL_FEATURES))
        probs = np.full(len(SPECIES), 0.02)
        probs[SPECIES.index("Clownfish")] = 0.90
        return probs


@pytest.fixture
def client():
    # No "with" block: the startup hook (which loads the real model) is skipped.
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_index_serves_form(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "<form" in res.text


def test_predict_returns_species(client):
    res = client.post("/predict", json=SALMON)
    assert res.status_code == 200
    body = res.json()
    assert body["species"] == "Clownfish"
    assert body["confidence"] == pytest.approx(0.90, abs=1e-3)
    assert set(body["probabilities"]) == set(SPECIES)


def test_predict_rejects_non_json(client):
    res = client.post("/predict", content=b"hello", headers={"Content-Type": "text/plain"})
    assert res.status_code == 415


def test_predict_rejects_out_of_range(client):
    res = client.post("/predict", json={**SALMON, "weight_kg": -3})
    assert res.status_code == 400
    assert "weight_kg" in res.json()["detail"]


def test_predict_rejects_missing_field(client):
    data = dict(SALMON)
    del data["length_cm"]
    assert client.post("/predict", json=data).status_code == 422


def test_predict_rejects_wrong_type(client):
    assert client.post("/predict", json={**SALMON, "fin_count": "many"}).status_code == 422


def test_predict_rejects_large_body(client):
    big = json.dumps({**SALMON, "notes": "x" * 20_000})
    res = client.post("/predict", content=big, headers={"Content-Type": "application/json"})
    assert res.status_code == 413


def test_predict_503_when_model_missing():
    app.dependency_overrides.clear()
    app.state.predictor = None
    res = TestClient(app).post("/predict", json=SALMON)
    assert res.status_code == 503


def _metric_value(text: str, line_start: str) -> float:
    for line in text.splitlines():
        if line.startswith(line_start):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_exposed_and_counting(client):
    before = client.get("/metrics").text
    client.post("/predict", json=SALMON)
    client.post("/predict", content=b"hello", headers={"Content-Type": "text/plain"})
    after = client.get("/metrics")

    assert after.status_code == 200
    assert "text/plain" in after.headers["content-type"]
    text = after.text

    key = 'predictions_total{species="Clownfish"}'
    assert _metric_value(text, key) == _metric_value(before, key) + 1

    key = 'prediction_errors_total{reason="unsupported_type"}'
    assert _metric_value(text, key) == _metric_value(before, key) + 1

    # Label order in the output isn't guaranteed, so check each label separately.
    request_lines = [
        line for line in text.splitlines()
        if line.startswith("http_requests_total{")
        and 'endpoint="/predict"' in line
        and 'method="POST"' in line
        and 'status="200"' in line
    ]
    assert request_lines, "no http_requests_total line for POST /predict 200"
    assert "http_request_duration_seconds_bucket" in text


# ---------------------------------------------------------------------------
# Real-model tests. The model file is committed to git, so the first one also
# runs in GitHub Actions and acts as a model-quality gate.
# ---------------------------------------------------------------------------
TYPICAL_FISH = {
    "Atlantic Herring": (30, 0.25, 6, 7, 10),
    "Atlantic Mackerel": (35, 0.50, 7, 9, 13),
    "Atlantic Salmon": (75, 4.50, 15, 8, 10),
    "Bluefin Tuna": (200, 200.0, 50, 9, 18),
    "Clownfish": (9, 0.02, 4, 7, 26),
    "European Sardine": (20, 0.08, 4, 7, 17),
}


@pytest.mark.skipif(not get_model_path().exists(), reason="trained model not found")
def test_real_model_on_typical_fish():
    app.dependency_overrides.clear()
    with TestClient(app) as real_client:  # runs startup, loads the model
        assert real_client.get("/health").json()["model_loaded"] is True
        correct = 0
        for species, values in TYPICAL_FISH.items():
            res = real_client.post("/predict", json=dict(zip(RAW_FEATURES, values)))
            assert res.status_code == 200
            correct += int(res.json()["species"] == species)
    assert correct >= 5


@pytest.mark.skipif(not get_model_path().exists(), reason="trained model not found")
def test_real_model_on_test_set():
    """Real held-out rows through the full API path (checks training-serving skew)."""
    data_file = get_model_path().parents[1] / "data" / "processed" / "fish.npz"
    if not data_file.exists():
        pytest.skip("processed data not found (run python -m src.data_preprocessing)")

    data = np.load(data_file)
    raw = data["X_test"][:, : len(RAW_FEATURES)]
    labels = data["y_test"]

    app.dependency_overrides.clear()
    with TestClient(app) as real_client:
        correct = 0
        for row, label in zip(raw, labels):
            payload = dict(zip(RAW_FEATURES, row.tolist()))
            payload["fin_count"] = int(payload["fin_count"])
            res = real_client.post("/predict", json=payload)
            assert res.status_code == 200
            correct += int(res.json()["species"] == SPECIES[int(label)])

    assert correct / len(labels) >= 0.85
