from pydantic import BaseModel, Field
from typing import Optional


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

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 117.0,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "DeviceType": "desktop",
                "C1": 1.0,
                "C2": 1.0,
            }
        }


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model confidence score [0, 1]")
    is_fraud: bool = Field(..., description="True if probability >= threshold")
    risk_tier: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    threshold_used: float
    model_version: str


class ExplanationResponse(PredictionResponse):
    top_features: list[dict] = Field(
        ...,
        description="Top 10 SHAP feature contributions, sorted by |impact|",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
