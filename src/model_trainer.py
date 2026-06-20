import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
import optuna
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.data_loader import load_merged
from src.preprocessing import (
    fit_preprocessing_artifacts,
    prepare_target,
    save_preprocessing_artifacts,
    split,
    transform,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.pkl"


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
    predictions = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, predictions)


def train(n_trials: int = 50):
    mlflow.set_experiment("veridian-fraud-detection")

    print("Loading data...")
    df = load_merged()
    X_raw, y = prepare_target(df)

    X_train_raw, X_test_raw, y_train, y_test = split(X_raw, y)
    artifacts = fit_preprocessing_artifacts(X_train_raw)

    X_train = transform(X_train_raw, artifacts)
    X_test = transform(X_test_raw, artifacts)
    X_tr, X_val, y_tr, y_val = split(X_train, y_train, test_size=0.15)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(
        f"Class ratio: {scale_pos_weight:.1f}x  |  Train: {len(X_tr):,}  "
        f"Val: {len(X_val):,}  Test: {len(X_test):,}"
    )

    print(f"Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, X_tr, X_val, y_tr, y_val, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "random_state": 42,
    }
    print(f"Best AUC-PR: {study.best_value:.4f}")

    with mlflow.start_run(run_name="xgb-final"):
        mlflow.log_params(best_params)

        final_model = XGBClassifier(**best_params, early_stopping_rounds=30)
        final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

        probabilities = final_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_test, probabilities),
            "auc_pr": average_precision_score(y_test, probabilities),
            "f1": f1_score(y_test, predictions),
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(final_model, "xgb_model")

        print("\n=== Test Set Results ===")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print(classification_report(y_test, predictions, target_names=["legit", "fraud"]))

        with open(MODEL_DIR / "xgb_model.pkl", "wb") as handle:
            pickle.dump(final_model, handle)
        with open(MODEL_DIR / "feature_columns.pkl", "wb") as handle:
            pickle.dump(artifacts.feature_columns, handle)

        save_preprocessing_artifacts(artifacts, PREPROCESSING_PATH)
        final_model.save_model(MODEL_DIR / "xgb_booster.json")
        print(f"\nArtifacts saved to {MODEL_DIR}")

    return final_model, metrics


if __name__ == "__main__":
    train()
