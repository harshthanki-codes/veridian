from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from src.settings import get_settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuntimeStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scored_transactions (
                    transaction_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    client_transaction_id TEXT,
                    request_payload TEXT NOT NULL,
                    prediction_payload TEXT NOT NULL,
                    explanation_payload TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS transaction_outcomes (
                    transaction_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(transaction_id) REFERENCES scored_transactions(transaction_id)
                )
                """
            )
            connection.commit()

    def record_prediction(
        self,
        *,
        tenant_id: str,
        request_id: str,
        request_payload: dict,
        prediction_payload: dict,
        explanation_payload: dict | None = None,
    ) -> str:
        transaction_id = str(uuid4())
        payload_json = json.dumps(request_payload, default=str)
        prediction_json = json.dumps(prediction_payload, default=str)
        explanation_json = json.dumps(explanation_payload, default=str) if explanation_payload else None
        client_transaction_id = request_payload.get("client_transaction_id")

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO scored_transactions (
                    transaction_id,
                    tenant_id,
                    request_id,
                    client_transaction_id,
                    request_payload,
                    prediction_payload,
                    explanation_payload,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    tenant_id,
                    request_id,
                    client_transaction_id,
                    payload_json,
                    prediction_json,
                    explanation_json,
                    _utc_now(),
                ),
            )
            connection.commit()

        return transaction_id

    def update_outcome(self, *, transaction_id: str, tenant_id: str, status: str, notes: str | None) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            exists = connection.execute(
                """
                SELECT 1
                FROM scored_transactions
                WHERE transaction_id = ? AND tenant_id = ?
                """,
                (transaction_id, tenant_id),
            ).fetchone()
            if not exists:
                return False

            connection.execute(
                """
                INSERT INTO transaction_outcomes (transaction_id, tenant_id, status, notes, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO UPDATE SET
                    status = excluded.status,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (transaction_id, tenant_id, status, notes, _utc_now()),
            )
            connection.commit()
            return True

    def get_transaction(self, *, transaction_id: str, tenant_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT
                    t.transaction_id,
                    t.tenant_id,
                    t.request_id,
                    t.client_transaction_id,
                    t.request_payload,
                    t.prediction_payload,
                    t.explanation_payload,
                    t.created_at,
                    o.status AS outcome_status,
                    o.notes AS outcome_notes,
                    o.updated_at AS outcome_updated_at
                FROM scored_transactions t
                LEFT JOIN transaction_outcomes o
                    ON o.transaction_id = t.transaction_id
                WHERE t.transaction_id = ? AND t.tenant_id = ?
                """,
                (transaction_id, tenant_id),
            ).fetchone()

        if row is None:
            return None

        return {
            "transaction_id": row["transaction_id"],
            "tenant_id": row["tenant_id"],
            "request_id": row["request_id"],
            "client_transaction_id": row["client_transaction_id"],
            "request_payload": json.loads(row["request_payload"]),
            "prediction_payload": json.loads(row["prediction_payload"]),
            "explanation_payload": json.loads(row["explanation_payload"]) if row["explanation_payload"] else None,
            "created_at": row["created_at"],
            "outcome_status": row["outcome_status"],
            "outcome_notes": row["outcome_notes"],
            "outcome_updated_at": row["outcome_updated_at"],
        }


@lru_cache(maxsize=1)
def get_store() -> RuntimeStore:
    settings = get_settings()
    store = RuntimeStore(settings.storage_path)
    store.initialize()
    return store
