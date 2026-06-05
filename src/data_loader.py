import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "veridian.db"
CHUNK_SIZE = 50_000


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_transactions_to_db(path: Path = DATA_DIR / "train_transaction.csv") -> None:
    conn = sqlite3.connect(DB_PATH)
    first = True
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = _downcast(chunk)
        chunk.to_sql("transactions", conn, if_exists="replace" if first else "append", index=False)
        first = False
        print(f"Loaded {CHUNK_SIZE} rows...")
    conn.close()
    print(f"Transactions written to {DB_PATH}")


def load_identity_to_db(path: Path = DATA_DIR / "train_identity.csv") -> None:
    conn = sqlite3.connect(DB_PATH)
    first = True
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = _downcast(chunk)
        chunk.to_sql("identity", conn, if_exists="replace" if first else "append", index=False)
        first = False
    conn.close()
    print(f"Identity written to {DB_PATH}")


def load_merged() -> pd.DataFrame:
    """SQL join avoids loading both full DataFrames into memory simultaneously."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT t.*, i.*
        FROM transactions t
        LEFT JOIN identity i ON t.TransactionID = i.TransactionID
    """
    df = pd.read_sql(query, conn)
    conn.close()
    # Drop duplicate TransactionID from identity side
    df = df.loc[:, ~df.columns.duplicated()]
    return df


if __name__ == "__main__":
    load_transactions_to_db()
    load_identity_to_db()
    print("Ingestion complete.")
