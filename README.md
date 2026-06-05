# Veridian — Financial Risk Intelligence API

Real-time fraud detection for transaction streams. Processes 143K+ historical transactions, serves sub-100ms predictions with SHAP-based per-transaction explainability.

---

## What's built

| Component | Status | Description |
|-----------|--------|-------------|
| Data ingestion | ✅ | Chunk-based CSV loading → SQLite, memory-optimized |
| Preprocessing | ✅ | Median imputation, categorical encoding, stratified split |
| Model training | ✅ | XGBoost + Optuna HPO + MLflow experiment tracking |
| Inference API | ✅ | FastAPI: `/predict`, `/predict/explain`, `/predict/batch` |
| SHAP explainability | ✅ | Per-transaction feature attributions via TreeExplainer |
| Tests | ✅ | API route tests + predictor unit tests |
| Docker | ✅ | Single-command deployment |
| CI | ✅ | GitHub Actions on push |

---

## Architecture

```
IEEE-CIS CSVs
     │
     ▼
data_loader.py ──► SQLite (veridian.db)
     │
     ▼
preprocessing.py ──► X_train, X_test, y_train, y_test
     │
     ▼
model_trainer.py ──► Optuna HPO ──► XGBClassifier ──► models/
     │                                                     │
     └──── MLflow tracking (mlruns/)                       │
                                                           ▼
                                               api/predictor.py
                                                     │
                              ┌──────────────────────┼──────────────────┐
                              ▼                      ▼                  ▼
                        /predict           /predict/explain    /predict/batch
```

---

## Project structure

```
veridian/
├── api/
│   ├── __init__.py
│   ├── predictor.py      # model loading, inference, SHAP
│   ├── routes.py         # FastAPI route handlers
│   └── schemas.py        # Pydantic request/response models
├── src/
│   ├── data_loader.py    # chunked ingestion → SQLite
│   ├── preprocessing.py  # cleaning, encoding, split
│   ├── model_trainer.py  # XGBoost + Optuna + MLflow
│   └── predictor.py      # CLI wrapper
├── models/
│   ├── xgb_model.pkl          # trained model
│   ├── feature_columns.pkl    # column order from training
│   └── xgb_booster.json       # booster (portable format)
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_predictor.py
├── data/                 # gitignored — add CSVs here
├── mlruns/               # MLflow artifacts
├── app.py                # FastAPI entrypoint
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Option A — Docker (recommended)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/veridian.git
cd veridian

# 2. Make sure models/ has xgb_model.pkl and feature_columns.pkl
#    (already present if you trained, or copy from your existing models/ folder)

# 3. Start API
docker compose up --build api
```

API live at **http://localhost:8000**  
Swagger docs at **http://localhost:8000/docs**

---

### Option B — Local (no Docker)

```bash
# Create and activate venv
python -m venv venv13
venv13\Scripts\activate          # Windows
# source venv13/bin/activate     # Mac/Linux

# Install only what the API needs (not the full requirements.txt)
pip install fastapi uvicorn pydantic xgboost scikit-learn shap numpy pandas python-dotenv

# Run
uvicorn app:app --reload --port 8000
```

---

## API Reference

### `POST /api/v1/predict`

Minimum payload:
```json
{
  "TransactionAmt": 117.0,
  "ProductCD": "W"
}
```

Response:
```json
{
  "fraud_probability": 0.823,
  "is_fraud": true,
  "risk_tier": "CRITICAL",
  "threshold_used": 0.5,
  "model_version": "xgb-v1.0"
}
```

Risk tiers: `LOW` (< 0.20) → `MEDIUM` (0.20–0.50) → `HIGH` (0.50–0.80) → `CRITICAL` (≥ 0.80)

---

### `POST /api/v1/predict/explain`

Same as `/predict` but includes SHAP attributions. ~3–5x slower — use for audit trails, not hot paths.

```json
{
  "fraud_probability": 0.823,
  "is_fraud": true,
  "risk_tier": "CRITICAL",
  "top_features": [
    {"feature": "TransactionAmt", "shap_value": 0.42, "feature_value": 1200.0},
    {"feature": "card4", "shap_value": -0.18, "feature_value": 1}
  ]
}
```

---

### `POST /api/v1/predict/batch`

Array of transactions, max 500 per request.

---

### `GET /api/v1/health`

```json
{"status": "ok", "model_loaded": true, "model_version": "xgb-v1.0"}
```

---

## Training the model

### 1. Get the dataset

Download **IEEE-CIS Fraud Detection** from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place:

```
data/train_transaction.csv
data/train_identity.csv
```

### 2. Run ingestion

```bash
python -m src.data_loader
```

### 3. Train

```bash
python -m src.model_trainer
```

This runs 50 Optuna trials (~30–60 min depending on hardware), logs to MLflow, and saves artifacts to `models/`.

### 4. View MLflow dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## Running tests

```bash
pip install pytest httpx
pytest
```

---

## Model performance (target)

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|--------------------|--------------:|--------:|
| AUC-ROC | ~0.85 | ~0.91 | **0.94+** |
| AUC-PR | ~0.61 | ~0.74 | **0.83+** |
| F1-Score | ~0.72 | ~0.81 | **0.87+** |

---

## Deploying to production

### AWS ECS (Fargate)

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker build -t veridian .
docker tag veridian:latest $ECR_URI/veridian:latest
docker push $ECR_URI/veridian:latest

# Then create ECS task definition pointing to the image
# Mount models/ from S3 or bake into image
```

### Any VPS

```bash
git clone https://github.com/YOUR_USERNAME/veridian.git && cd veridian
# Copy your models/ folder here
docker compose up -d api
```

---

## Key engineering decisions

**Chunk-based ingestion** — 590K rows × 400+ columns doesn't fit safely in memory. 50K-row chunks + SQLite join sidesteps OOM on a 16GB machine.

**JSONB-free schema** — SQLite for ingestion pipeline, no ORM needed. The API doesn't touch the training DB at all.

**`lru_cache(maxsize=1)` on model load** — model artifacts are loaded once per process, shared across all requests. No per-request disk I/O.

**SHAP on separate endpoint** — TreeExplainer adds ~200–400ms. Keeping it on `/explain` means `/predict` stays under 100ms p99.

**`scale_pos_weight`** — Derived from actual class ratio at training time (~25x imbalance). Not hardcoded.

---

Built by [Harsh Thanki](https://linkedin.com/in/harsh-thanki-60ba41317)
