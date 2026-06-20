from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MISSING_RATIO_THRESHOLD = 0.90
DEFAULT_TIME_COLUMN = "TransactionDT"
TARGET_COLUMN = "isFraud"


@dataclass
class PreprocessingArtifacts:
    feature_columns: list[str]
    numeric_fill_values: dict[str, float]
    categorical_mappings: dict[str, dict[str, int]]
    dropped_columns: list[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work = work.drop(columns=self.dropped_columns, errors="ignore")

        for column, fill_value in self.numeric_fill_values.items():
            if column not in work.columns:
                work[column] = np.nan
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(fill_value)

        for column, mapping in self.categorical_mappings.items():
            if column not in work.columns:
                work[column] = -1
                continue
            normalized = work[column].fillna("__missing__").astype(str)
            work[column] = normalized.map(mapping).fillna(-1).astype("int32")

        transformed = work.reindex(columns=self.feature_columns, fill_value=np.nan)
        return transformed


def prepare_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in training data")

    features = df.drop(columns=[TARGET_COLUMN]).copy()
    target = df[TARGET_COLUMN].astype(int).copy()
    return features, target


def fit_preprocessing_artifacts(
    df: pd.DataFrame,
    *,
    missing_ratio_threshold: float = MISSING_RATIO_THRESHOLD,
) -> PreprocessingArtifacts:
    work = df.copy()
    dropped_columns: list[str] = []

    missing_cutoff = int(len(work) * missing_ratio_threshold)
    high_missing_columns = [
        column for column in work.columns if work[column].isna().sum() > missing_cutoff
    ]
    dropped_columns.extend(high_missing_columns)

    identifier_columns = [
        column
        for column in work.columns
        if column == "TransactionID" or column.lower().startswith("id_")
    ]
    dropped_columns.extend(identifier_columns)

    if dropped_columns:
        work = work.drop(columns=sorted(set(dropped_columns)), errors="ignore")

    numeric_columns = work.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in work.columns if column not in numeric_columns]

    numeric_fill_values: dict[str, float] = {}
    for column in numeric_columns:
        values = pd.to_numeric(work[column], errors="coerce")
        median = values.median()
        numeric_fill_values[column] = float(median) if pd.notna(median) else 0.0

    categorical_mappings: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        normalized = work[column].fillna("__missing__").astype(str)
        categories = sorted(normalized.unique().tolist())
        categorical_mappings[column] = {value: index for index, value in enumerate(categories)}

    feature_columns = work.columns.tolist()
    transformed = PreprocessingArtifacts(
        feature_columns=feature_columns,
        numeric_fill_values=numeric_fill_values,
        categorical_mappings=categorical_mappings,
        dropped_columns=sorted(set(dropped_columns)),
    ).transform(df)

    return PreprocessingArtifacts(
        feature_columns=transformed.columns.tolist(),
        numeric_fill_values=numeric_fill_values,
        categorical_mappings=categorical_mappings,
        dropped_columns=sorted(set(dropped_columns)),
    )


def fit_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingArtifacts]:
    artifacts = fit_preprocessing_artifacts(df)
    return artifacts.transform(df), artifacts


def transform(df: pd.DataFrame, artifacts: PreprocessingArtifacts) -> pd.DataFrame:
    return artifacts.transform(df)


def split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    time_column: str = DEFAULT_TIME_COLUMN,
):
    if time_column in X.columns:
        if len(X) < 2:
            return X, X.iloc[0:0], y, y.iloc[0:0]

        ordered = X.sort_values(time_column).index.to_list()
        split_at = max(1, int(len(ordered) * (1 - test_size)))
        split_at = min(split_at, len(ordered) - 1)
        train_idx = ordered[:split_at]
        test_idx = ordered[split_at:]
        return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def save_preprocessing_artifacts(artifacts: PreprocessingArtifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(artifacts, handle)


def load_preprocessing_artifacts(path: Path) -> PreprocessingArtifacts:
    with open(path, "rb") as handle:
        return pickle.load(handle)
