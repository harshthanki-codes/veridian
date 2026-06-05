import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from functools import lru_cache

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_VERSION = "xgb-v1.0"

# Risk tiers based on fraud probability
RISK_TIERS = [
    (0.80, "CRITICAL"),
    (0.50, "HIGH"),
    (0.20, "MEDIUM"),
    (0.00, "LOW"),
]

DECISION_THRESHOLD = 0.50


@lru_cache(maxsize=1)
def _load_artifacts():
    with open(MODEL_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, feature_columns, explainer


def _risk_tier(prob: float) -> str:
    for threshold, tier in RISK_TIERS:
        if prob >= threshold:
            return tier
    return "LOW"


def _build_feature_vector(data: dict, feature_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data])
    # Encode object columns to category codes, matching training preprocessing
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.Categorical(df[col]).codes

    # Align to training feature set — missing cols become NaN (XGBoost handles this)
    df = df.reindex(columns=feature_columns, fill_value=np.nan)
    return df


def predict(data: dict) -> dict:
    model, feature_columns, _ = _load_artifacts()
    X = _build_feature_vector(data, feature_columns)

    prob = float(model.predict_proba(X)[0][1])
    return {
        "fraud_probability": round(prob, 6),
        "is_fraud": prob >= DECISION_THRESHOLD,
        "risk_tier": _risk_tier(prob),
        "threshold_used": DECISION_THRESHOLD,
        "model_version": MODEL_VERSION,
    }


def predict_with_explanation(data: dict) -> dict:
    model, feature_columns, explainer = _load_artifacts()
    X = _build_feature_vector(data, feature_columns)

    prob = float(model.predict_proba(X)[0][1])
    shap_values = explainer.shap_values(X)

    # shap_values shape: (1, n_features) for binary XGBoost
    contributions = shap_values[0] if shap_values.ndim == 2 else shap_values[1][0]

    top_features = sorted(
        [
            {"feature": col, "shap_value": round(float(val), 6), "feature_value": X[col].iloc[0]}
            for col, val in zip(feature_columns, contributions)
            if not np.isnan(val)
        ],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )[:10]

    return {
        "fraud_probability": round(prob, 6),
        "is_fraud": prob >= DECISION_THRESHOLD,
        "risk_tier": _risk_tier(prob),
        "threshold_used": DECISION_THRESHOLD,
        "model_version": MODEL_VERSION,
        "top_features": top_features,
    }


def model_is_loaded() -> bool:
    try:
        _load_artifacts()
        return True
    except Exception:
        return False
