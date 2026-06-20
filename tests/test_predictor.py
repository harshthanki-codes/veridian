import numpy as np
import pandas as pd

from api.predictor import DEFAULT_DECISION_THRESHOLD, _build_feature_vector, _risk_tier
from src.preprocessing import PreprocessingArtifacts


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


def test_build_feature_vector_aligns_columns(monkeypatch):
    artifacts = PreprocessingArtifacts(
        feature_columns=["TransactionAmt", "ProductCD", "card4", "C1", "V1"],
        numeric_fill_values={"TransactionAmt": 0.0, "C1": 0.0, "V1": 0.0},
        categorical_mappings={
            "ProductCD": {"W": 1, "__missing__": 0},
            "card4": {"visa": 1, "__missing__": 0},
        },
        dropped_columns=[],
    )
    monkeypatch.setattr(
        "api.predictor._load_artifacts",
        lambda: (None, artifacts, artifacts.feature_columns, None),
    )

    data = {"TransactionAmt": 100.0, "ProductCD": "W", "card4": "visa"}
    frame = _build_feature_vector(data)
    assert list(frame.columns) == artifacts.feature_columns
    assert frame["C1"].iloc[0] == 0.0
    assert frame["V1"].iloc[0] == 0.0


def test_build_feature_vector_uses_saved_category_mapping(monkeypatch):
    artifacts = PreprocessingArtifacts(
        feature_columns=["TransactionAmt", "ProductCD"],
        numeric_fill_values={"TransactionAmt": 0.0},
        categorical_mappings={"ProductCD": {"W": 7, "__missing__": 0}},
        dropped_columns=[],
    )
    monkeypatch.setattr(
        "api.predictor._load_artifacts",
        lambda: (None, artifacts, artifacts.feature_columns, None),
    )

    frame = _build_feature_vector({"ProductCD": "W", "TransactionAmt": 50.0})
    assert frame["ProductCD"].iloc[0] == 7
    assert pd.api.types.is_integer_dtype(frame["ProductCD"])


def test_build_feature_vector_unknown_categories_fall_back_to_negative_one(monkeypatch):
    artifacts = PreprocessingArtifacts(
        feature_columns=["TransactionAmt", "ProductCD"],
        numeric_fill_values={"TransactionAmt": 0.0},
        categorical_mappings={"ProductCD": {"W": 1, "__missing__": 0}},
        dropped_columns=[],
    )
    monkeypatch.setattr(
        "api.predictor._load_artifacts",
        lambda: (None, artifacts, artifacts.feature_columns, None),
    )

    frame = _build_feature_vector({"ProductCD": "Z", "TransactionAmt": 50.0})
    assert frame["ProductCD"].iloc[0] == -1


def test_decision_threshold_is_half():
    assert DEFAULT_DECISION_THRESHOLD == 0.50
