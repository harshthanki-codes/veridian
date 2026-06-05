from fastapi import APIRouter, HTTPException
from api.schemas import TransactionRequest, PredictionResponse, ExplanationResponse, HealthResponse
from api import predictor

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    loaded = predictor.model_is_loaded()
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_version": predictor.MODEL_VERSION,
    }


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(tx: TransactionRequest):
    try:
        result = predictor.predict(tx.model_dump(exclude_none=False))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    return result


@router.post("/predict/explain", response_model=ExplanationResponse, tags=["inference"])
def predict_with_explanation(tx: TransactionRequest):
    """
    Same as /predict but includes SHAP feature attributions.
    ~3–5x slower due to TreeExplainer; use sparingly on hot paths.
    """
    try:
        result = predictor.predict_with_explanation(tx.model_dump(exclude_none=False))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
    return result


@router.post("/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
def predict_batch(transactions: list[TransactionRequest]):
    if len(transactions) > 500:
        raise HTTPException(status_code=400, detail="Batch size capped at 500")
    results = []
    for tx in transactions:
        try:
            results.append(predictor.predict(tx.model_dump(exclude_none=False)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch inference failed at index {len(results)}: {e}")
    return results
