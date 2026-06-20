from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import RequestContext, get_request_context, get_runtime_store

MOCK_PREDICTION = {
    "fraud_probability": 0.823,
    "is_fraud": True,
    "risk_tier": "CRITICAL",
    "threshold_used": 0.55,
    "model_version": "xgb-v1.1",
}

MOCK_EXPLANATION = {
    **MOCK_PREDICTION,
    "top_features": [
        {"feature": "TransactionAmt", "shap_value": 0.42, "feature_value": 1200.0},
        {"feature": "card4", "shap_value": -0.18, "feature_value": 1},
    ],
}

SAMPLE_TX = {
    "TransactionAmt": 1200.0,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "gmail.com",
    "DeviceType": "desktop",
}


@dataclass(frozen=True)
class FakeTenant:
    tenant_id: str = "tenant-demo"
    display_name: str = "Tenant Demo"
    api_key: str = "test-key"
    decision_threshold: float = 0.55
    batch_limit: int = 500


@pytest.fixture
def mock_predictor():
    with patch("api.routes.predictor") as mocked:
        mocked.model_is_loaded.return_value = True
        mocked.MODEL_VERSION = "xgb-v1.1"
        mocked.predict.return_value = MOCK_PREDICTION
        mocked.predict_with_explanation.return_value = MOCK_EXPLANATION
        yield mocked


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.record_prediction.side_effect = ["tx-001", "tx-002", "tx-003", "tx-004", "tx-005", "tx-006"]
    store.update_outcome.return_value = True
    store.get_transaction.return_value = {
        "transaction_id": "tx-001",
        "tenant_id": "tenant-demo",
        "request_id": "req-123",
        "client_transaction_id": "ord_10294",
        "request_payload": SAMPLE_TX,
        "prediction_payload": MOCK_PREDICTION,
        "explanation_payload": MOCK_EXPLANATION,
        "created_at": "2026-06-20T00:00:00+00:00",
        "outcome_status": "approved",
        "outcome_notes": "Reviewed by ops",
        "outcome_updated_at": "2026-06-20T01:00:00+00:00",
    }
    return store


@pytest.fixture
def client(mock_predictor, mock_store):
    from app import app

    def _request_context_override():
        return RequestContext(tenant=FakeTenant(), request_id="req-123")

    app.dependency_overrides[get_request_context] = _request_context_override
    app.dependency_overrides[get_runtime_store] = lambda: mock_store
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
