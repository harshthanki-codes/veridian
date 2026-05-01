import joblib
import pandas as pd

MODEL_PATH = "models/xgb_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


def encode_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def align_features(df):
    df = df.copy()

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]


def predict_transaction(df):
    df = encode_categoricals(df)
    df = align_features(df)

    prob = model.predict_proba(df)[0][1]
    return float(prob)