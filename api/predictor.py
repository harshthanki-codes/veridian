from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import PreprocessingArtifacts, load_preprocessing_artifacts
from src.tenant_config import TenantConfig

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_VERSION = "xgb-v1.1"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.pkl"

RISK_TIERS = [
    (0.80, "CRITICAL"),
    (0.50, "HIGH"),
    (0.20, "MEDIUM"),
    (0.00, "LOW"),
]

DEFAULT_DECISION_THRESHOLD = 0.50


class LegacyPreprocessor:
    def __init__(self, feature_columns: list[str]):
        self._artifacts = PreprocessingArtifacts(
            feature_columns=feature_columns,
            numeric_fill_values={},
            categorical_mappings={},
            dropped_columns=[],
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for column in work.select_dtypes(include="object").columns:
            work[column] = pd.Categorical(work[column]).codes
        return work.reindex(columns=self._artifacts.feature_columns, fill_value=np.nan)

    @property
    def feature_columns(self) -> list[str]:
        return self._artifacts.feature_columns


@lru_cache(maxsize=1)
def _load_artifacts():
    with open(MODEL_DIR / "xgb_model.pkl", "rb") as handle:
        model = pickle.load(handle)

    if PREPROCESSING_PATH.exists():
        preprocessor: PreprocessingArtifacts | LegacyPreprocessor = load_preprocessing_artifacts(
            PREPROCESSING_PATH
        )
        feature_columns = preprocessor.feature_columns
    else:
        with open(MODEL_DIR / "feature_columns.pkl", "rb") as handle:
            feature_columns = pickle.load(handle)
        preprocessor = LegacyPreprocessor(feature_columns)

    return model, preprocessor, feature_columns


@lru_cache(maxsize=1)
def _load_explainer():
    model, _, _ = _load_artifacts()
    try:
        import shap
    except Exception as exc:
        raise RuntimeError(
            "SHAP is not available in the current environment. "
            "Install a SHAP/NumPy-compatible stack to use /predict/explain."
        ) from exc

    try:
        return shap.TreeExplainer(model)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize SHAP explainer. "
            "This usually means the installed SHAP build is incompatible with the current NumPy version."
        ) from exc


def _risk_tier(probability: float) -> str:
    for threshold, tier in RISK_TIERS:
        if probability >= threshold:
            return tier
    return "LOW"


def _decision_threshold(tenant: TenantConfig | None) -> float:
    if tenant is None:
        return DEFAULT_DECISION_THRESHOLD
    return tenant.decision_threshold


def _build_feature_vector(data: dict) -> pd.DataFrame:
    _, preprocessor, _ = _load_artifacts()
    frame = pd.DataFrame([data])
    return preprocessor.transform(frame)


def predict(data: dict, tenant: TenantConfig | None = None) -> dict:
    model, _, _ = _load_artifacts()
    X = _build_feature_vector(data)
    threshold = _decision_threshold(tenant)

    probability = float(model.predict_proba(X)[0][1])
    return {
        "fraud_probability": round(probability, 6),
        "is_fraud": probability >= threshold,
        "risk_tier": _risk_tier(probability),
        "threshold_used": threshold,
        "model_version": MODEL_VERSION,
    }


def predict_with_explanation(data: dict, tenant: TenantConfig | None = None) -> dict:
    model, _, feature_columns = _load_artifacts()
    explainer = _load_explainer()
    X = _build_feature_vector(data)
    threshold = _decision_threshold(tenant)

    probability = float(model.predict_proba(X)[0][1])
    shap_values = explainer.shap_values(X)
    contributions = shap_values[0] if getattr(shap_values, "ndim", 1) == 2 else shap_values[1][0]

    top_features = sorted(
        [
            {
                "feature": column,
                "shap_value": round(float(value), 6),
                "feature_value": X[column].iloc[0],
            }
            for column, value in zip(feature_columns, contributions)
            if not np.isnan(value)
        ],
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )[:10]

    return {
        "fraud_probability": round(probability, 6),
        "is_fraud": probability >= threshold,
        "risk_tier": _risk_tier(probability),
        "threshold_used": threshold,
        "model_version": MODEL_VERSION,
        "top_features": top_features,
    }


def model_is_loaded() -> bool:
    try:
        _load_artifacts()
        return True
    except Exception:
        return False
