import pytest
from tests.conftest import SAMPLE_TX, MOCK_PREDICTION, MOCK_EXPLANATION


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_fraud_fields(client):
    r = client.post("/api/v1/predict", json=SAMPLE_TX)
    assert r.status_code == 200
    data = r.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_tier" in data
    assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_predict_high_amount_flagged(client, mock_predictor):
    mock_predictor.predict.return_value = {**MOCK_PREDICTION, "fraud_probability": 0.91, "risk_tier": "CRITICAL"}
    r = client.post("/api/v1/predict", json={**SAMPLE_TX, "TransactionAmt": 9999.0})
    assert r.status_code == 200
    assert r.json()["risk_tier"] == "CRITICAL"


def test_predict_invalid_amount(client):
    r = client.post("/api/v1/predict", json={**SAMPLE_TX, "TransactionAmt": -50.0})
    assert r.status_code == 422


def test_predict_missing_required_field(client):
    payload = {k: v for k, v in SAMPLE_TX.items() if k != "TransactionAmt"}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_explain_includes_shap(client):
    r = client.post("/api/v1/predict/explain", json=SAMPLE_TX)
    assert r.status_code == 200
    data = r.json()
    assert "top_features" in data
    assert len(data["top_features"]) > 0
    assert "shap_value" in data["top_features"][0]


def test_batch_predict(client, mock_predictor):
    mock_predictor.predict.return_value = MOCK_PREDICTION
    payload = [SAMPLE_TX] * 3
    r = client.post("/api/v1/predict/batch", json=payload)
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_batch_size_limit(client):
    payload = [SAMPLE_TX] * 501
    r = client.post("/api/v1/predict/batch", json=payload)
    assert r.status_code == 400
    assert "500" in r.json()["detail"]


def test_predict_inference_error_returns_500(client, mock_predictor):
    mock_predictor.predict.side_effect = RuntimeError("model exploded")
    r = client.post("/api/v1/predict", json=SAMPLE_TX)
    assert r.status_code == 500
