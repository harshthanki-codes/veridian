import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


MOCK_PREDICTION = {
    "fraud_probability": 0.823,
    "is_fraud": True,
    "risk_tier": "CRITICAL",
    "threshold_used": 0.5,
    "model_version": "xgb-v1.0",
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


@pytest.fixture
def mock_predictor():
    with patch("api.routes.predictor") as m:
        m.model_is_loaded.return_value = True
        m.MODEL_VERSION = "xgb-v1.0"
        m.predict.return_value = MOCK_PREDICTION
        m.predict_with_explanation.return_value = MOCK_EXPLANATION
        yield m


@pytest.fixture
def client(mock_predictor):
    from app import app
    return TestClient(app)
