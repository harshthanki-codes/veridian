import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Fraud Detection", layout="wide")

MODEL_PATH = "models/xgb_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"
DB_PATH = "data/veridian.db"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_columns():
    return joblib.load(FEATURE_PATH)

model = load_model()
feature_columns = load_feature_columns()

# ---------------- LOAD DATA (SAFE) ----------------
@st.cache_data
def load_sample_data(limit=500):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"SELECT * FROM transactions LIMIT {limit}", conn)
        conn.close()
    except:
        df = pd.read_csv(
            "data/train_transaction.csv",
            nrows=limit,
            low_memory=True
        )

    return df.copy()

df = load_sample_data()

# ---------------- HELPERS ----------------
def encode_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def align_features(df):
    df = df.copy()

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns].copy()


def predict(df_input):
    X = encode_categoricals(df_input)
    X = align_features(X)
    return float(model.predict_proba(X)[0][1])


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox("Choose Input", ["Sample Data", "Manual Input"])
threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.3)

# ---------------- DISPLAY COLUMNS ----------------
DISPLAY_COLS = [
    "TransactionAmt", "ProductCD", "card1", "card2",
    "card3", "card4", "card5", "card6",
    "addr1", "addr2"
]

DISPLAY_COLS = [c for c in DISPLAY_COLS if c in df.columns]

# ---------------- SAMPLE MODE ----------------
if mode == "Sample Data":

    st.subheader("📊 Select Transaction")

    idx = st.slider("Row Index", 0, len(df) - 1, 0)
    sample = df.iloc[[idx]].copy()

    st.subheader("📄 Transaction Overview")
    st.dataframe(sample[DISPLAY_COLS], width="stretch")

    if st.button("🔍 Predict"):

        prob = predict(sample)

        # RESULT
        st.subheader("📢 Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fraud Probability", f"{prob:.6f}")

        with col2:
            if prob > threshold:
                st.error("⚠️ Fraud Transaction")
            else:
                st.success("✅ Legitimate Transaction")

        # RISK
        st.subheader("📊 Risk Level")

        if prob < 0.1:
            st.success("🟢 Very Low Risk")
        elif prob < 0.3:
            st.info("🔵 Low Risk")
        elif prob < 0.6:
            st.warning("🟠 Medium Risk")
        else:
            st.error("🔴 High Risk 🚨")

        # FEATURE IMPORTANCE
        st.subheader("📊 Top Features Affecting Prediction")

        if hasattr(model, "feature_importances_"):
            feat_df = pd.DataFrame({
                "feature": feature_columns,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False).head(10)

            st.dataframe(feat_df, width="stretch")
        else:
            st.warning("Feature importance not available")


# ---------------- MANUAL MODE ----------------
else:

    st.subheader("✍️ Manual Input")

    col1, col2 = st.columns(2)

    with col1:
        TransactionAmt = st.number_input("TransactionAmt", value=100.0)
        card1 = st.number_input("card1", value=1000)
        card2 = st.number_input("card2", value=100)

    with col2:
        card3 = st.number_input("card3", value=150)
        addr1 = st.number_input("addr1", value=200)
        addr2 = st.number_input("addr2", value=80)

    if st.button("🔍 Predict"):

        sample = df.iloc[[0]].copy()

        sample["TransactionAmt"] = TransactionAmt
        sample["card1"] = card1
        sample["card2"] = card2
        sample["card3"] = card3
        sample["addr1"] = addr1
        sample["addr2"] = addr2

        prob = predict(sample)

        # RESULT
        st.subheader("📢 Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fraud Probability", f"{prob:.6f}")

        with col2:
            if prob > threshold:
                st.error("⚠️ Fraud Transaction")
            else:
                st.success("✅ Legitimate Transaction")

        # RISK
        st.subheader("📊 Risk Level")

        if prob < 0.1:
            st.success("🟢 Very Low Risk")
        elif prob < 0.3:
            st.info("🔵 Low Risk")
        elif prob < 0.6:
            st.warning("🟠 Medium Risk")
        else:
            st.error("🔴 High Risk 🚨")