import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Drop columns with >90% missing — they add noise, not signal
    threshold = 0.9 * len(df)
    df = df.dropna(thresh=len(df) - threshold, axis=1)

    target = df.pop("isFraud")

    # Median fill for numeric, -1 for categorical (XGBoost handles this gracefully)
    for col in df.select_dtypes(include=["float32", "float64", "int32", "int64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col]).codes

    # Drop ID columns — no predictive value
    drop_cols = [c for c in df.columns if "TransactionID" in c or "id_" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df, target


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
