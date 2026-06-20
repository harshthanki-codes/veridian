from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: str = Field(..., description="Product code: W, H, C, S, R")
    card4: Optional[str] = Field(None, description="Card network: visa, mastercard, etc.")
    card6: Optional[str] = Field(None, description="Card type: debit, credit")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    DeviceType: Optional[str] = Field(None, description="desktop or mobile")
    DeviceInfo: Optional[str] = Field(None)
    addr1: Optional[float] = Field(None)
    addr2: Optional[float] = Field(None)
    dist1: Optional[float] = Field(None)
    dist2: Optional[float] = Field(None)
    C1: Optional[float] = Field(None)
    C2: Optional[float] = Field(None)
    C3: Optional[float] = Field(None)
    C4: Optional[float] = Field(None)
    C5: Optional[float] = Field(None)
    C6: Optional[float] = Field(None)
    C7: Optional[float] = Field(None)
    C8: Optional[float] = Field(None)
    C9: Optional[float] = Field(None)
    C10: Optional[float] = Field(None)
    C11: Optional[float] = Field(None)
    C12: Optional[float] = Field(None)
    C13: Optional[float] = Field(None)
    C14: Optional[float] = Field(None)
    D1: Optional[float] = Field(None)
    D2: Optional[float] = Field(None)
    D3: Optional[float] = Field(None)
    D4: Optional[float] = Field(None)
    D5: Optional[float] = Field(None)
    D10: Optional[float] = Field(None)
    D15: Optional[float] = Field(None)
    M1: Optional[str] = Field(None)
    M2: Optional[str] = Field(None)
    M3: Optional[str] = Field(None)
    M4: Optional[str] = Field(None)
    M5: Optional[str] = Field(None)
    M6: Optional[str] = Field(None)
    M7: Optional[str] = Field(None)
    M8: Optional[str] = Field(None)
    M9: Optional[str] = Field(None)
    V1: Optional[float] = Field(None)
    V2: Optional[float] = Field(None)
    V3: Optional[float] = Field(None)
    V4: Optional[float] = Field(None)
    V5: Optional[float] = Field(None)
    V6: Optional[float] = Field(None)
    client_transaction_id: Optional[str] = Field(
        None,
        description="Caller-provided identifier used to correlate external payment events.",
    )
    event_timestamp: Optional[datetime] = Field(
        None,
        description="When the merchant observed the transaction event.",
    )
    metadata: Optional[dict[str, Any]] = Field(
        None,
        description="Opaque merchant metadata stored with the scored transaction.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionAmt": 117.0,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "DeviceType": "desktop",
                "C1": 1.0,
                "C2": 1.0,
                "client_transaction_id": "ord_10294",
                "metadata": {"checkout_source": "web"},
            }
        }
    }


class PredictionResponse(BaseModel):
    transaction_id: str
    tenant_id: str
    request_id: str
    fraud_probability: float = Field(..., description="Model confidence score [0, 1]")
    is_fraud: bool = Field(..., description="True if probability >= threshold")
    risk_tier: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    threshold_used: float
    model_version: str


class ExplanationFeature(BaseModel):
    feature: str
    shap_value: float
    feature_value: float | int | str | None


class ExplanationResponse(PredictionResponse):
    top_features: list[ExplanationFeature] = Field(
        ...,
        description="Top feature contributions sorted by absolute impact.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    tenant_configured: bool


OutcomeStatus = Literal["approved", "declined", "reviewed", "refunded", "disputed", "chargebacked"]


class OutcomeUpdateRequest(BaseModel):
    status: OutcomeStatus
    notes: Optional[str] = Field(None, max_length=2000)


class OutcomeResponse(BaseModel):
    transaction_id: str
    tenant_id: str
    status: OutcomeStatus
    notes: Optional[str] = None


class TransactionRecordResponse(BaseModel):
    transaction_id: str
    tenant_id: str
    request_id: str
    client_transaction_id: Optional[str]
    request_payload: dict[str, Any]
    prediction_payload: dict[str, Any]
    explanation_payload: Optional[dict[str, Any]]
    created_at: str
    outcome_status: Optional[OutcomeStatus]
    outcome_notes: Optional[str]
    outcome_updated_at: Optional[str]
