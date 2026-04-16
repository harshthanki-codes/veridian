import os
import sqlite3
import pandas as pd
import joblib

from src.preprocessing import preprocess
from src.model_trainer import run_training

DATA_DIR = "data"
TRANSACTION_FILE = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_FILE = os.path.join(DATA_DIR, "train_identity.csv")
DB_PATH = os.path.join(DATA_DIR, "veridian.db")


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if not isinstance(df[col], pd.Series):
            continue

        col_type = df[col].dtype

        if col_type != "object":
            if str(col_type).startswith("int"):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            else:
                df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def build_database():
    conn = sqlite3.connect(DB_PATH)

    print("Loading transactions...")
    for i, chunk in enumerate(pd.read_csv(TRANSACTION_FILE, chunksize=50000)):
        print(f"Transaction chunk {i+1}")
        chunk = reduce_memory(chunk)
        chunk.to_sql(
            "transactions",
            conn,
            if_exists="replace" if i == 0 else "append",
            index=False,
        )

    print("Loading identity...")
    for i, chunk in enumerate(pd.read_csv(IDENTITY_FILE, chunksize=50000)):
        print(f"Identity chunk {i+1}")
        chunk = reduce_memory(chunk)
        chunk.to_sql(
            "identity",
            conn,
            if_exists="replace" if i == 0 else "append",
            index=False,
        )

    conn.close()


def load_sample_data(limit: int = 100000) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    print(f"Loading {limit} rows from DB...")

    query = f"""
        SELECT t.*, i.DeviceType, i.DeviceInfo
        FROM transactions t
        LEFT JOIN identity i
        ON t.TransactionID = i.TransactionID
        LIMIT {limit}
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df = reduce_memory(df)
    return df


if __name__ == "__main__":

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(DB_PATH):
        build_database()

    df = load_sample_data()

    print(f"Loaded data shape: {df.shape}")

    # 🔥 FIX: SAVE FEATURE COLUMNS BEFORE PREPROCESS
    feature_columns = df.drop(columns=["isFraud"]).columns.tolist()
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    # PREPROCESS
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # TRAIN MODEL
    model = run_training(X_train, X_val, y_train, y_val)

    # SAVE MODEL
    joblib.dump(model, "models/xgb_model.pkl")

    print("✅ Model and feature columns saved successfully!")