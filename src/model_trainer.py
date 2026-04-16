import os
import tempfile
import pandas as pd
import mlflow
import mlflow.xgboost
import skops.io as sio
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier


# ---------------- MODELS ----------------
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        tol=1e-3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


# ---------------- EVALUATION ----------------
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)

    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_val, y_pred))

    return auc


def show_feature_importance(model, feature_names, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop Features:")
    print(feat_imp.head(top_n))


# ---------------- SAFE LOGGING ----------------
def log_sklearn_model_safe(model, model_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, f"{model_name}.skops")
        sio.dump(model, path)
        mlflow.log_artifact(path, artifact_path="model")


# ---------------- MAIN TRAINING ----------------
def run_training(X_train, X_val, y_train, y_val):

    mlflow.set_experiment("fraud_detection")

    # -------- Logistic Regression --------
    with mlflow.start_run(run_name="Logistic Regression"):
        print("\nTraining Logistic Regression...")

        lr_model = train_logistic_regression(X_train, y_train)
        auc = evaluate_model(lr_model, X_val, y_val)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("auc", auc)

        log_sklearn_model_safe(lr_model, "lr_model")

    # -------- Random Forest --------
    with mlflow.start_run(run_name="Random Forest"):
        print("\nTraining Random Forest...")

        rf_model = train_random_forest(X_train, y_train)
        auc = evaluate_model(rf_model, X_val, y_val)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("auc", auc)

        log_sklearn_model_safe(rf_model, "rf_model")

    # -------- XGBoost (FINAL MODEL) --------
    with mlflow.start_run(run_name="XGBoost"):
        print("\nTraining XGBoost...")

        xgb_model = train_xgboost(X_train, y_train)
        auc = evaluate_model(xgb_model, X_val, y_val)

        mlflow.log_param("model", "XGBoost")
        mlflow.log_metric("auc", auc)

        mlflow.xgboost.log_model(xgb_model, name="model")

        # save for dashboard
        os.makedirs("models", exist_ok=True)
        joblib.dump(xgb_model, "models/xgb_model.pkl")

        # feature importance (safe)
        try:
            feature_names = [f"f{i}" for i in range(X_train.shape[1])]
            show_feature_importance(xgb_model, feature_names)
        except Exception:
            pass

    return xgb_model