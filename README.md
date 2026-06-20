# Veridian

> A fraud risk and chargeback intelligence platform for digital commerce.
>
> Veridian helps merchants detect risky transactions in real time, explain decisions clearly, and evolve from reactive fraud handling to measurable risk operations.

---

## Why Veridian Exists

Online businesses lose money in two directions at the same time:

1. Fraudulent transactions slip through and become chargebacks, refunds, losses, and operational fire drills.
2. Legitimate customers get blocked by overly strict rules, reducing revenue and damaging trust.

Most smaller merchants and growing digital businesses do not have:

- A dedicated fraud team
- A mature risk engine
- Data scientists tuning models weekly
- Internal tooling for explainability, auditability, and feedback loops

They still face the same pressure as larger companies:

- Keep approval rates high
- Keep fraud and chargebacks low
- Move fast without taking on hidden risk

Veridian exists to close that gap.

This project is building toward a world-class, managed fraud decisioning system that gives businesses better visibility, better control, and better outcomes without requiring them to build a full risk organization from scratch.

---

## The Problem We Are Solving

At a business level, Veridian is not just solving "fraud detection."

It is solving a broader operating problem:

**How do digital merchants make fast, trustworthy decisions on risky payments without sacrificing revenue, customer experience, or team bandwidth?**

That breaks into four concrete problems:

### 1. Risk Visibility

Many teams cannot easily answer:

- Which transactions are actually risky?
- Why is the system flagging them?
- Which risk patterns are increasing over time?
- Where are we losing money: fraud, false declines, or manual review overhead?

### 2. Decision Quality

Simple rule-based systems are brittle. Generic fraud tools are often too expensive, too opaque, or too broad for smaller teams.

Merchants need decisions that are:

- Fast
- Defensible
- Tunable
- Specific to their business

### 3. Operational Burden

Fraud is not only a model problem. It is an operations problem.

Teams need workflows for:

- Reviewing high-risk events
- Investigating explanations
- Tracking outcomes
- Learning from chargebacks and disputes

### 4. Scalable Trust

As payment volume grows, manual review does not scale. Teams need systems that:

- Score transactions consistently
- Explain decisions clearly
- Improve with feedback
- Fit inside real production environments

---

## What Veridian Is

Today, Veridian is an ML-powered inference API for fraud risk scoring built on the IEEE-CIS Fraud Detection dataset.

It already includes:

- Data ingestion into SQLite for memory-conscious processing
- Feature preprocessing for a high-dimensional fraud dataset
- XGBoost model training with Optuna-based tuning
- MLflow experiment tracking
- FastAPI inference endpoints
- SHAP-based per-transaction explainability
- Dockerized local deployment
- CI and test coverage for key API behavior

In other words:

- As an engineer, you can see a complete ML-to-API pipeline.
- As a system designer, you can see the major platform boundaries.
- As a founder, you can see the seed of a real product.
- As a CEO, you can evaluate whether the problem is important enough to build into a business.

---

## What Veridian Is Becoming

The long-term product is not just a scoring endpoint.

The long-term product is a **managed fraud intelligence and chargeback reduction platform** for digital merchants.

That means evolving from:

- Model API

into:

- Decision engine
- Risk operations platform
- Merchant-specific policy layer
- Outcome-driven fraud partner

The future version of Veridian should help customers:

- Score transactions in real time
- Tune approval and decline thresholds by use case
- Investigate why a transaction was flagged
- Track fraud outcomes and drift over time
- Build feedback loops from disputes, refunds, and chargebacks
- Operate a fraud program without hiring a large internal team

---

## Product Vision

### The Core Promise

**Approve more good customers. Catch more bad actors. Explain every decision. Improve over time.**

### The Ideal Customer

Veridian is best positioned for:

- Digital-first merchants
- Subscription businesses
- SaaS and AI products
- Creator and gaming platforms
- Small and mid-sized e-commerce teams
- Businesses with chargeback pain but no dedicated fraud engineering team

### The Wrong First Customer

Veridian is not yet meant to directly compete for:

- Large banks
- Card networks
- Enterprise payment processors
- Global retailers with mature internal risk teams

The strongest early strategy is to win a clear niche, prove measurable ROI, and expand from there.

---

## Current System at a Glance

### High-Level Flow

```text
Historical transaction data
        |
        v
Chunked ingestion -> SQLite
        |
        v
Preprocessing and feature preparation
        |
        v
Model training -> Optuna tuning -> MLflow tracking
        |
        v
Saved model artifacts
        |
        v
FastAPI inference layer
        |
        +--> /predict
        +--> /predict/explain
        +--> /predict/batch
        +--> /health
```

### What This Flow Means

- The data pipeline keeps training practical on normal hardware.
- The training stack is structured enough to support experimentation.
- The API layer turns offline model work into an actual product surface.
- The explainability layer makes the system more useful for operators, auditors, and customer-facing trust conversations.

---

## Architecture Overview

### Training Pipeline

1. Raw transaction and identity CSVs are loaded in chunks.
2. Data is stored in SQLite to reduce memory pressure during ingestion.
3. The merged dataset is preprocessed into model-ready features.
4. XGBoost is optimized with Optuna.
5. Results are logged in MLflow.
6. Artifacts are saved for production inference.

### Inference Pipeline

1. The API receives a transaction payload.
2. The predictor loads cached model artifacts.
3. Input data is aligned to the training feature schema.
4. The model returns a fraud probability.
5. The system maps probability to a risk tier.
6. The explanation route additionally computes top SHAP contributions.

### Major Components

| Component | Purpose |
|----------|---------|
| `src/data_loader.py` | Chunked ingestion of historical data into SQLite |
| `src/preprocessing.py` | Cleaning, feature prep, and train/test split |
| `src/model_trainer.py` | Hyperparameter tuning, model training, metrics, artifact persistence |
| `api/predictor.py` | Runtime inference, model loading, SHAP explainability |
| `api/routes.py` | FastAPI route layer |
| `api/schemas.py` | Request and response contracts |
| `app.py` | Application entrypoint |
| `tests/` | API and predictor test coverage |

---

## Why This Project Matters

Many AI/ML repos stop at one of these stages:

- Notebook-only experimentation
- Training scripts with no serving layer
- API demos with no model development discipline
- Product claims with no real implementation depth

Veridian is valuable because it already connects:

- Data pipeline
- Training pipeline
- Experiment tracking
- Model serving
- Explainability
- Deployment surface

That makes it more than a demo. It is an early product foundation.

---

## Honest Status: Where We Are Right Now

Veridian is currently best described as:

**A credible fraud-risk MVP with strong technical foundations and clear product potential.**

What is already strong:

- Clear end-to-end architecture
- Good project organization
- Real training and inference separation
- Explainability built into the product surface
- Docker support
- CI workflow
- Basic tests around API contracts and utility behavior

What is not world-class yet:

- Production-grade authentication and authorization
- Tenant isolation and customer configuration
- Observability and incident-ready operations
- Real feedback loops from live merchant outcomes
- Rule engine and policy controls
- Analyst dashboard and case management
- Robust preprocessing parity between training and inference
- Time-aware validation for realistic fraud evaluation
- Mature deployment, secrets, and compliance story

This honesty is a strength, not a weakness. Great products are trusted when they are ambitious and precise.

---

## What World-Class Looks Like

A world-class Veridian should be able to do the following:

### For Customers

- Score live transactions in milliseconds
- Show why a transaction is risky
- Reduce fraud losses and chargebacks
- Reduce false declines
- Give non-ML teams confidence in decisions
- Fit into existing payment and commerce workflows

### For Operators

- Support tenant-specific thresholds and rules
- Track fraud, disputes, approvals, and drift
- Replay and audit key decisions
- Review flagged cases efficiently
- Feed outcomes back into retraining and policy tuning

### For the Business

- Deliver measurable ROI quickly
- Support managed deployment for clients
- Keep infrastructure and operating burden low for the Veridian team
- Create a strong base for recurring revenue

---

## Commercial Direction

### Best Initial Positioning

Do not lead with:

- "general AI fraud API"
- "better than every fraud platform"
- "enterprise-grade for all industries"

Lead with:

**Fraud risk and chargeback intelligence for digital merchants that need better decisions without building an internal fraud team.**

### Best Delivery Model

The best operating model is likely:

- Client-owned infrastructure
- Veridian-managed deployment and maintenance
- Integration and risk-ops services billed by Veridian

That keeps infrastructure costs off your balance sheet while preserving revenue opportunities through:

- Setup fees
- Monthly platform fees
- Managed support retainers
- Outcome-based pricing in selected accounts

### How to Sell It

The strongest sales motion is:

1. Offer a free fraud or chargeback audit.
2. Show preventable losses and false-decline opportunity.
3. Run a focused pilot.
4. Convert based on measurable business outcomes.

Do not sell AUC first.
Sell:

- fewer chargebacks
- more approved good customers
- clearer risk decisions
- less manual review overhead

---

## Repository Structure

```text
veridian/
|-- api/
|   |-- __init__.py
|   |-- predictor.py
|   |-- routes.py
|   `-- schemas.py
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- model_trainer.py
|   `-- predictor.py
|-- tests/
|   |-- __init__.py
|   |-- conftest.py
|   |-- test_api.py
|   `-- test_predictor.py
|-- data/
|-- models/
|-- mlruns/
|-- app.py
|-- Dockerfile
|-- docker-compose.yml
|-- pytest.ini
`-- requirements.txt
```

---

## API Surface

### `POST /api/v1/predict`

Scores a single transaction and returns:

- `fraud_probability`
- `is_fraud`
- `risk_tier`
- `threshold_used`
- `model_version`

### `POST /api/v1/predict/explain`

Returns the prediction plus top SHAP feature contributions for explainability.

Use this when you need:

- analyst visibility
- debugging support
- trust and audit context

### `POST /api/v1/predict/batch`

Scores multiple transactions in a single request.

### `GET /api/v1/health`

Returns API and model readiness information.

---

## Example Request

```json
{
  "TransactionAmt": 117.0,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com",
  "DeviceType": "desktop"
}
```

## Example Response

```json
{
  "fraud_probability": 0.823,
  "is_fraud": true,
  "risk_tier": "CRITICAL",
  "threshold_used": 0.5,
  "model_version": "xgb-v1.0"
}
```

Risk tiers:

- `LOW` for lower-risk transactions
- `MEDIUM` for moderate caution
- `HIGH` for strong review signal
- `CRITICAL` for likely fraud or immediate intervention

---

## Local Setup

### Option A: Docker

```bash
docker compose up --build api
```

Then open:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

### Option B: Local Python Environment

Create a Python 3.11 environment and install dependencies:

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

---

## Training the Model

### 1. Download the dataset

Download the IEEE-CIS Fraud Detection files from Kaggle and place them in:

```text
data/train_transaction.csv
data/train_identity.csv
```

### 2. Ingest the data

```bash
python -m src.data_loader
```

### 3. Train

```bash
python -m src.model_trainer
```

### 4. Inspect experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Testing

Run:

```bash
pytest
```

CI is configured through GitHub Actions to validate the core test suite on push and pull request.

---

## Engineering Principles Behind This Project

Veridian is designed around a few practical principles:

### 1. Build for clarity, not only cleverness

A system that cannot be understood cannot be trusted.

### 2. Separate experimentation from serving

Training and inference have different responsibilities and should evolve deliberately.

### 3. Treat explainability as product value

For risk systems, explanations are not decoration. They are part of operational usability.

### 4. Stay honest about maturity

Strong software is built by making sharp distinctions between:

- prototype
- MVP
- production-ready
- world-class

This repository is moving deliberately across those stages.

---

## Current Gaps to Close Next

To move from strong MVP to production candidate, the next priorities are:

1. Persist the preprocessing pipeline so training and inference use the exact same transformations.
2. Add auth, API keys, tenant-aware configs, and secrets management.
3. Introduce structured logging, metrics, tracing, and alerting.
4. Add time-based validation and better evaluation for fraud realism.
5. Build a merchant feedback loop from disputes, refunds, and chargebacks.
6. Add a rule engine and threshold controls by merchant.
7. Build an operator dashboard for review and monitoring.

---

## Why This README Is Written This Way

This project needs to communicate with multiple audiences at once.

### For Engineers

It shows the architecture, repo structure, and execution flow.

### For System Designers

It clarifies boundaries, platform direction, and what still needs to mature.

### For Founders

It frames the real customer problem, ideal niche, and expansion path.

### For CEOs and Decision-Makers

It explains why this problem matters commercially and how the product could be monetized without carrying all infrastructure cost internally.

That is intentional. Great technical products earn trust when the code, the architecture, and the business story all line up.

---

## Final Positioning

Veridian is not just a fraud model.

It is the beginning of a risk intelligence product designed to help digital businesses make faster, safer, and more explainable payment decisions.

If built to its full potential, Veridian can become:

- a serious fraud decisioning engine
- a chargeback reduction platform
- a managed risk operations product
- a high-trust tool for growing digital merchants

That is the standard this project is aiming for.

---

Built for ambitious, practical, high-trust risk systems.
