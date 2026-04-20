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


# =====================================================
# MEMORY REDUCTION
# =====================================================
def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "float64":
            df[col] = df[col].astype("float32")

        elif col_type == "int64":
            df[col] = df[col].astype("int32")

    return df


# =====================================================
# BUILD SQLITE DATABASE (CHUNK BASED)
# =====================================================
def build_database():
    conn = sqlite3.connect(DB_PATH)

    print("Building database...")

    # -------- Transactions --------
    for i, chunk in enumerate(pd.read_csv(TRANSACTION_FILE, chunksize=50000)):
        print(f"Transaction chunk {i+1}")

        chunk = reduce_memory(chunk)

        chunk.to_sql(
            "transactions",
            conn,
            if_exists="replace" if i == 0 else "append",
            index=False
        )

    # -------- Identity --------
    for i, chunk in enumerate(pd.read_csv(IDENTITY_FILE, chunksize=50000)):
        print(f"Identity chunk {i+1}")

        chunk = reduce_memory(chunk)

        chunk.to_sql(
            "identity",
            conn,
            if_exists="replace" if i == 0 else "append",
            index=False
        )

    conn.close()
    print("✅ Database built successfully!")


# =====================================================
# LOAD SAMPLE DATA (SAFE)
# =====================================================
def load_sample_data(limit: int = 30000) -> pd.DataFrame:
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

    # -------- Reduce memory --------
    df = reduce_memory(df)

    # -------- DROP HEAVY COLUMNS (VERY IMPORTANT) --------
    # V columns are huge and mostly unnecessary
    v_cols = [col for col in df.columns if col.startswith("V")]
    df = df.drop(columns=v_cols[:200], errors="ignore")

    return df


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    os.makedirs("models", exist_ok=True)

    # Build DB if not exists
    if not os.path.exists(DB_PATH):
        build_database()

    # Load data safely
    df = load_sample_data(limit=30000)

    print(f"Loaded data shape: {df.shape}")

    # -------- PREPROCESS --------
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # -------- SAVE FEATURE COLUMNS (FIXED) --------
    try:
        feature_columns = list(X_train.columns)
    except:
        feature_columns = [f"f{i}" for i in range(X_train.shape[1])]

    joblib.dump(feature_columns, "models/feature_columns.pkl")

    # -------- TRAIN MODEL --------
    model = run_training(X_train, X_val, y_train, y_val)

    print("✅ Everything completed successfully!")