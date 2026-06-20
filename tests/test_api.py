from tests.conftest import MOCK_PREDICTION, SAMPLE_TX


def test_health_ok(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_fraud_fields(client):
    response = client.post("/api/v1/predict", json=SAMPLE_TX)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx-001"
    assert data["tenant_id"] == "tenant-demo"
    assert data["request_id"] == "req-123"
    assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_predict_high_amount_flagged(client, mock_predictor):
    mock_predictor.predict.return_value = {**MOCK_PREDICTION, "fraud_probability": 0.91, "risk_tier": "CRITICAL"}
    response = client.post("/api/v1/predict", json={**SAMPLE_TX, "TransactionAmt": 9999.0})
    assert response.status_code == 200
    assert response.json()["risk_tier"] == "CRITICAL"


def test_predict_invalid_amount(client):
    response = client.post("/api/v1/predict", json={**SAMPLE_TX, "TransactionAmt": -50.0})
    assert response.status_code == 422


def test_predict_missing_required_field(client):
    payload = {key: value for key, value in SAMPLE_TX.items() if key != "TransactionAmt"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_explain_includes_shap(client):
    response = client.post("/api/v1/predict/explain", json=SAMPLE_TX)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx-001"
    assert "top_features" in data
    assert len(data["top_features"]) > 0
    assert "shap_value" in data["top_features"][0]


def test_batch_predict(client, mock_predictor):
    mock_predictor.predict.return_value = MOCK_PREDICTION
    payload = [SAMPLE_TX] * 3
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_batch_size_limit(client):
    payload = [SAMPLE_TX] * 501
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 400
    assert "500" in response.json()["detail"]


def test_predict_inference_error_returns_500(client, mock_predictor):
    mock_predictor.predict.side_effect = RuntimeError("model exploded")
    response = client.post("/api/v1/predict", json=SAMPLE_TX)
    assert response.status_code == 500


def test_update_outcome(client):
    response = client.post(
        "/api/v1/transactions/tx-001/outcome",
        json={"status": "approved", "notes": "Reviewed by ops"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approved"


def test_get_transaction_record(client):
    response = client.get("/api/v1/transactions/tx-001")
    assert response.status_code == 200
    assert response.json()["transaction_id"] == "tx-001"
