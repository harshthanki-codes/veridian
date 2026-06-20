from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api import predictor
from api.dependencies import RequestContext, get_request_context, get_runtime_store
from api.schemas import (
    ExplanationResponse,
    HealthResponse,
    OutcomeResponse,
    OutcomeUpdateRequest,
    PredictionResponse,
    TransactionRecordResponse,
    TransactionRequest,
)
from src.storage import RuntimeStore
from src.tenant_config import get_tenant_catalog

router = APIRouter()


def _enrich_result(result: dict, *, transaction_id: str, tenant_id: str, request_id: str) -> dict:
    return {
        "transaction_id": transaction_id,
        "tenant_id": tenant_id,
        "request_id": request_id,
        **result,
    }


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    catalog = get_tenant_catalog()
    loaded = predictor.model_is_loaded()
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_version": predictor.MODEL_VERSION,
        "tenant_configured": catalog.is_configured,
    }


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_transaction(
    tx: TransactionRequest,
    context: RequestContext = Depends(get_request_context),
    store: RuntimeStore = Depends(get_runtime_store),
):
    payload = tx.model_dump(exclude_none=False)
    try:
        result = predictor.predict(payload, tenant=context.tenant)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    transaction_id = store.record_prediction(
        tenant_id=context.tenant.tenant_id,
        request_id=context.request_id,
        request_payload=payload,
        prediction_payload=result,
    )
    return _enrich_result(
        result,
        transaction_id=transaction_id,
        tenant_id=context.tenant.tenant_id,
        request_id=context.request_id,
    )


@router.post("/predict/explain", response_model=ExplanationResponse, tags=["inference"])
def predict_with_explanation(
    tx: TransactionRequest,
    context: RequestContext = Depends(get_request_context),
    store: RuntimeStore = Depends(get_runtime_store),
):
    payload = tx.model_dump(exclude_none=False)
    try:
        result = predictor.predict_with_explanation(payload, tenant=context.tenant)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

    transaction_id = store.record_prediction(
        tenant_id=context.tenant.tenant_id,
        request_id=context.request_id,
        request_payload=payload,
        prediction_payload={key: value for key, value in result.items() if key != "top_features"},
        explanation_payload=result,
    )
    return _enrich_result(
        result,
        transaction_id=transaction_id,
        tenant_id=context.tenant.tenant_id,
        request_id=context.request_id,
    )


@router.post("/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
def predict_batch(
    transactions: list[TransactionRequest],
    context: RequestContext = Depends(get_request_context),
    store: RuntimeStore = Depends(get_runtime_store),
):
    if len(transactions) > context.tenant.batch_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size capped at {context.tenant.batch_limit}",
        )

    results: list[dict] = []
    for tx in transactions:
        payload = tx.model_dump(exclude_none=False)
        try:
            result = predictor.predict(payload, tenant=context.tenant)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Batch inference failed at index {len(results)}: {exc}",
            ) from exc

        transaction_id = store.record_prediction(
            tenant_id=context.tenant.tenant_id,
            request_id=context.request_id,
            request_payload=payload,
            prediction_payload=result,
        )
        results.append(
            _enrich_result(
                result,
                transaction_id=transaction_id,
                tenant_id=context.tenant.tenant_id,
                request_id=context.request_id,
            )
        )

    return results


@router.post(
    "/transactions/{transaction_id}/outcome",
    response_model=OutcomeResponse,
    tags=["operations"],
)
def update_outcome(
    transaction_id: str,
    outcome: OutcomeUpdateRequest,
    context: RequestContext = Depends(get_request_context),
    store: RuntimeStore = Depends(get_runtime_store),
):
    updated = store.update_outcome(
        transaction_id=transaction_id,
        tenant_id=context.tenant.tenant_id,
        status=outcome.status,
        notes=outcome.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return {
        "transaction_id": transaction_id,
        "tenant_id": context.tenant.tenant_id,
        "status": outcome.status,
        "notes": outcome.notes,
    }


@router.get(
    "/transactions/{transaction_id}",
    response_model=TransactionRecordResponse,
    tags=["operations"],
)
def get_transaction(
    transaction_id: str,
    context: RequestContext = Depends(get_request_context),
    store: RuntimeStore = Depends(get_runtime_store),
):
    record = store.get_transaction(
        transaction_id=transaction_id,
        tenant_id=context.tenant.tenant_id,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return record
