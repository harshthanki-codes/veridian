import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


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


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)

    print(f"AUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))


def run_training(X_train, X_val, y_train, y_val):
    print("Training Logistic Regression...")

    model = train_logistic_regression(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_val, y_val)

    return model