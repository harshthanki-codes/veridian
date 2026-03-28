import os
import sqlite3
import pandas as pd
from src.preprocessing import preprocess

DATA_DIR = "data"
TRANSACTION_FILE = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_FILE = os.path.join(DATA_DIR, "train_identity.csv")
DB_PATH = os.path.join(DATA_DIR, "veridian.db")


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != "object":
            if str(col_type).startswith("int"):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            else:
                df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    # transactions → DB
    for i, chunk in enumerate(pd.read_csv(TRANSACTION_FILE, chunksize=50000)):
        chunk = reduce_memory(chunk)
        chunk.to_sql(
            "transactions",
            conn,
            if_exists="replace" if i == 0 else "append",
            index=False,
        )

    # identity → DB
    for i, chunk in enumerate(pd.read_csv(IDENTITY_FILE, chunksize=50000)):
        chunk = reduce_memory(chunk)
        chunk.to_sql(
            "identity",
            conn,
            if_exists="replace" if i == 0 else "append",
            index=False,
        )

    # SQL join (safe, no RAM spike)
    query = """
        SELECT t.*, i.*
        FROM transactions t
        LEFT JOIN identity i
        ON t.TransactionID = i.TransactionID
    """

    df = pd.read_sql(query, conn)

    conn.close()

    df = reduce_memory(df)

    return df


def query_data(limit: int = 5) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    query = f"SELECT * FROM transactions LIMIT {limit}"
    df = pd.read_sql(query, conn)

    conn.close()
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded data shape: {df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)