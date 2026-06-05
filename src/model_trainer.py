import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import optuna
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
from src.data_loader import load_merged
from src.preprocessing import preprocess, split

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _objective(trial, X_train, X_val, y_train, y_val, scale_pos_weight: float):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5),
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": 42,
    }
    model = XGBClassifier(**params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, preds)


def train(n_trials: int = 50):
    mlflow.set_experiment("veridian-fraud-detection")

    print("Loading data...")
    df = load_merged()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split(X, y)
    X_tr, X_val, y_tr, y_val = split(X_train, y_train, test_size=0.15)

    # Class imbalance ratio for scale_pos_weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Class ratio: {scale_pos_weight:.1f}x  |  Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    print(f"Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, X_tr, X_val, y_tr, y_val, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = {**study.best_params, "scale_pos_weight": scale_pos_weight, "tree_method": "hist", "random_state": 42}
    print(f"Best AUC-PR: {study.best_value:.4f}")

    with mlflow.start_run(run_name="xgb-final"):
        mlflow.log_params(best_params)

        final_model = XGBClassifier(**best_params, early_stopping_rounds=30)
        final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

        proba = final_model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_test, proba),
            "auc_pr": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(final_model, "xgb_model")

        print("\n=== Test Set Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(classification_report(y_test, preds, target_names=["legit", "fraud"]))

        # Persist artifacts
        with open(MODEL_DIR / "xgb_model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(MODEL_DIR / "feature_columns.pkl", "wb") as f:
            pickle.dump(list(X.columns), f)

        final_model.save_model(MODEL_DIR / "xgb_booster.json")
        print(f"\nArtifacts saved to {MODEL_DIR}")

    return final_model, metrics


if __name__ == "__main__":
    train()
