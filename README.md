# Veridian

Veridian is a production-oriented financial risk intelligence platform designed to detect fraudulent transactions in real time.  

The system processes large-scale transaction data, applies memory-efficient ingestion techniques, and builds a structured pipeline for modeling, explainability, and deployment.

This project focuses on building not just a model, but a complete end-to-end system similar to what real fintech teams maintain internally.

---

## Problem

Fraud detection is inherently difficult because:

- Extreme class imbalance (~3–4% fraud cases)
- Behavioral patterns evolve over time
- High cost of false negatives (missed fraud)
- Need for explainability in financial systems

Veridian addresses these challenges by combining efficient data engineering, scalable ML pipelines, and model interpretability.

---

## System Overview

The pipeline is designed as a layered system:

1. Data Ingestion  
   - Chunk-based CSV loading  
   - Memory optimization using dtype downcasting  
   - SQLite-backed storage for scalable joins  

2. Data Processing  
   - Missing value handling (median + categorical fallback)  
   - Categorical encoding using category codes  
   - Stratified dataset splitting  

3. Modeling (upcoming)  
   - Baseline models: Logistic Regression, Random Forest  
   - Advanced model: XGBoost with hyperparameter tuning  
   - Evaluation using AUC-ROC, AUC-PR, F1-score  

4. Explainability (planned)  
   - SHAP-based feature contribution  
   - Per-transaction reasoning  

5. API Layer (planned)  
   - FastAPI endpoints for prediction and explanation  

---

## Tech Stack

**Core**
- Python
- Pandas, NumPy
- Scikit-learn
- SQLite

**ML & Optimization (planned)**
- XGBoost
- Optuna
- SHAP

**API & Deployment (planned)**
- FastAPI
- Docker
- AWS ECS

---

## Project Structure


veridian/
│
├── data/ # raw data (ignored in git)
├── notebooks/ # EDA & experiments
├── src/
│ ├── data_loader.py # ingestion + DB pipeline
│ ├── preprocessing.py # cleaning + encoding + split
│ └── init.py
│
├── api/ # (planned) FastAPI service
├── tests/ # (planned) unit tests
│
├── requirements.txt
├── README.md
└── .gitignore


---

## Key Engineering Decisions

### 1. Chunk-based Processing
The dataset (~590K rows, 400+ columns) cannot be safely loaded into memory in one go.  
All ingestion is done using chunked reads to prevent memory crashes.

### 2. Database-backed Join
Instead of merging large DataFrames in memory, joins are executed in SQLite:

- Reduces memory usage significantly  
- Mirrors production ETL patterns  
- Improves stability for large datasets  

### 3. Memory Optimization
Numeric columns are downcasted to smaller types (`int32`, `float32`) to reduce footprint.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/veridian.git
cd veridian
2. Create environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Add dataset

Download IEEE-CIS Fraud Detection dataset from Kaggle and place:

data/train_transaction.csv
data/train_identity.csv
5. Run pipeline
python -m src.data_loader
Current Status
Data ingestion pipeline: complete
Memory optimization: implemented
Preprocessing pipeline: complete
Database-backed merge: implemented

Next steps:

Model training (baseline → XGBoost)
MLflow integration
SHAP explainability
FastAPI deployment
Metrics Target (Planned)
Model	AUC-ROC	AUC-PR	F1 Score
Logistic Regression	~0.85	~0.61	~0.72
Random Forest	~0.91	~0.74	~0.81
XGBoost (Target)	0.94+	0.83+	0.87+
Why This Project

This project is intentionally built beyond a typical notebook:

Focus on system design, not just modeling
Handles real-world constraints (memory, scale)
Designed to be deployable as a service
Author

Harsh Thanki
MCA | Machine Learning & Systems Engineering Focus

Note

Dataset is not included due to size constraints.
Download from Kaggle: IEEE-CIS Fraud Detection


---
