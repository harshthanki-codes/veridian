import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from api.predictor import _risk_tier, _build_feature_vector, DECISION_THRESHOLD


def test_risk_tier_critical():
    assert _risk_tier(0.85) == "CRITICAL"


def test_risk_tier_high():
    assert _risk_tier(0.65) == "HIGH"


def test_risk_tier_medium():
    assert _risk_tier(0.35) == "MEDIUM"


def test_risk_tier_low():
    assert _risk_tier(0.10) == "LOW"


def test_risk_tier_boundary_high():
    assert _risk_tier(0.80) == "CRITICAL"
    assert _risk_tier(0.799) == "HIGH"


def test_build_feature_vector_aligns_columns():
    data = {"TransactionAmt": 100.0, "ProductCD": "W", "card4": "visa"}
    feature_columns = ["TransactionAmt", "ProductCD", "card4", "C1", "V1"]
    df = _build_feature_vector(data, feature_columns)
    assert list(df.columns) == feature_columns
    # Missing columns must be NaN
    assert np.isnan(df["C1"].iloc[0])
    assert np.isnan(df["V1"].iloc[0])


def test_build_feature_vector_encodes_categoricals():
    data = {"ProductCD": "W", "TransactionAmt": 50.0}
    feature_columns = ["TransactionAmt", "ProductCD"]
    df = _build_feature_vector(data, feature_columns)
    # After Categorical encoding, ProductCD must be numeric
    assert df["ProductCD"].dtype in [np.int8, np.int16, np.int32, np.int64]


def test_decision_threshold_is_half():
    # If you change this, fraud_probability contract breaks for consumers
    assert DECISION_THRESHOLD == 0.50
