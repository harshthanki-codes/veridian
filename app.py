import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Fraud Detection System")

MODEL_PATH = "models/xgb_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"
DB_PATH = "data/veridian.db"

# ---------------- SAFE LOAD ----------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    st.error("❌ Model not found. Run: python -m src.data_loader")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

# ---------------- PREPROCESS ----------------
def encode_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def align_features(df, feature_columns):
    df = df.copy()

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct order
    df = df[feature_columns]

    return df


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_sample_data(limit=1000):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM transactions LIMIT {limit}", conn)
    conn.close()
    return df


df = load_sample_data()

DISPLAY_COLS = [
    "TransactionAmt", "ProductCD", "card1", "card2", "card3",
    "card4", "card5", "card6", "addr1", "addr2"
]

AVAILABLE_DISPLAY_COLS = [c for c in DISPLAY_COLS if c in df.columns]

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox("Choose Input", ["Sample Data", "Manual Input"])

threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.3)

# ---------------- SAMPLE MODE ----------------
if mode == "Sample Data":

    st.subheader("📊 Select Transaction")

    idx = st.slider("Row Index", 0, len(df) - 1, 0)
    sample = df.iloc[[idx]].copy()

    st.write("### 🧾 Transaction Overview")
    st.dataframe(sample[AVAILABLE_DISPLAY_COLS], width="stretch")

    if st.button("🔍 Predict", key="sample_predict"):

        X = sample.drop(columns=["isFraud"], errors="ignore")

        X = encode_categoricals(X)
        X = align_features(X, feature_columns)

        prob = float(model.predict_proba(X)[0][1])  # ✅ FIXED float issue

        st.subheader("📢 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fraud Probability", f"{prob:.6f} ({prob:.2%})")

        with col2:
            if prob > threshold:
                st.error("⚠️ Fraud Detected")
            else:
                st.success("✅ Legitimate Transaction")

        # ---------------- PROGRESS BAR ----------------
        st.progress(min(max(prob, 0.0), 1.0))  # ✅ SAFE float

        # ---------------- RISK LEVEL ----------------
        st.subheader("📊 Risk Level")

        if prob < 0.1:
            st.success("🟢 Very Low Risk")
        elif prob < 0.3:
            st.info("🔵 Low Risk")
        elif prob < 0.6:
            st.warning("🟠 Medium Risk")
        else:
            st.error("🔴 High Risk 🚨")


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

    if st.button("🔍 Predict", key="manual_predict"):

        sample = df.iloc[[0]].copy()

        sample["TransactionAmt"] = TransactionAmt
        sample["card1"] = card1
        sample["card2"] = card2
        sample["card3"] = card3
        sample["addr1"] = addr1
        sample["addr2"] = addr2

        X = sample.drop(columns=["isFraud"], errors="ignore")

        X = encode_categoricals(X)
        X = align_features(X, feature_columns)

        prob = float(model.predict_proba(X)[0][1])  # ✅ FIXED

        st.subheader("📢 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fraud Probability", f"{prob:.6f} ({prob:.2%})")

        with col2:
            if prob > threshold:
                st.error("⚠️ Fraud Detected")
            else:
                st.success("✅ Legitimate Transaction")

        # ---------------- PROGRESS BAR ----------------
        st.progress(min(max(prob, 0.0), 1.0))  # ✅ FIXED

        # ---------------- RISK LEVEL ----------------
        st.subheader("📊 Risk Level")

        if prob < 0.1:
            st.success("🟢 Very Low Risk")
        elif prob < 0.3:
            st.info("🔵 Low Risk")
        elif prob < 0.6:
            st.warning("🟠 Medium Risk")
        else:
            st.error("🔴 High Risk 🚨")