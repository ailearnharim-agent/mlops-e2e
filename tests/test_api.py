import pytest
from fastapi.testclient import TestClient

from src.api.main import app

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("No disease", "Disease present")
    assert 0.0 <= body["confidence"] <= 1.0
    assert "model_version" in body


def test_predict_valid_payload_without_optional_fields(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k not in ("slope", "ca", "thal")}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)


def test_predict_missing_required_field_returns_422(client):
    payload = dict(VALID_PAYLOAD)
    del payload["age"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_out_of_range_field_returns_422(client):
    payload = dict(VALID_PAYLOAD)
    payload["sex"] = 5  # only 0/1 allowed
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422(client):
    payload = dict(VALID_PAYLOAD)
    payload["age"] = "sixty-three"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json=VALID_PAYLOAD)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "predictions_total" in response.text
