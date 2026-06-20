FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.135.2 \
    uvicorn==0.42.0 \
    pydantic==2.12.5 \
    xgboost==3.2.0 \
    scikit-learn==1.7.2 \
    shap==0.49.1 \
    numpy==2.2.6 \
    pandas==2.3.3 \
    python-dotenv==1.2.2

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
