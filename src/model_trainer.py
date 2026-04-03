import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier


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
    # handle class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val, name="Model"):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)

    print(f"\n{name} Results")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))


def show_feature_importance(model, feature_names, top_n=20):
    importance = model.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print("\nTop Features:")
    print(feat_imp.head(top_n))


def run_training(X_train, X_val, y_train, y_val, feature_names):
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_val, y_val, name="Logistic Regression")

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_val, y_val, name="Random Forest")

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_val, y_val, name="XGBoost")

    print("\nXGBoost Feature Importance:")
    show_feature_importance(xgb_model, feature_names)

    return xgb_model