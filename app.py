import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import io

from src.predictor import predict_transaction

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

# ---------------- PATHS ----------------
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

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_sample_data(limit=500):

    try:
        conn = sqlite3.connect(DB_PATH)

        query = f"""
        SELECT *
        FROM transactions
        LIMIT {limit}
        """

        df = pd.read_sql(query, conn)
        conn.close()

    except Exception:

        df = pd.read_csv(
            "data/train_transaction.csv",
            nrows=limit,
            low_memory=True
        )

    return df.copy()


df = load_sample_data()

# ---------------- DISPLAY COLUMNS ----------------
DISPLAY_COLS = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2"
]

DISPLAY_COLS = [c for c in DISPLAY_COLS if c in df.columns]

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Choose Input",
    ["Sample Data", "Manual Input"]
)

threshold = st.sidebar.slider(
    "Fraud Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.01
)

# ---------------- REPORT FUNCTION ----------------
def generate_report(sample, prob):

    report = sample.copy()

    report["Fraud_Probability"] = prob

    buffer = io.BytesIO()

    report.to_csv(buffer, index=False)

    return buffer


# =========================================================
# SAMPLE DATA MODE
# =========================================================
if mode == "Sample Data":

    st.title("💳 Fraud Detection System")

    st.subheader("📊 Select Transaction")

    idx = st.slider(
        "Row Index",
        0,
        len(df) - 1,
        0
    )

    sample = df.iloc[[idx]].copy()

    st.subheader("📄 Transaction Overview")

    st.dataframe(
        sample[DISPLAY_COLS],
        width="stretch"
    )

    if st.button("🔍 Predict"):

        # ---------------- PREDICTION ----------------
        prob = predict_transaction(sample)

        # ---------------- RESULT ----------------
        st.subheader("📢 Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fraud Probability",
                f"{prob:.6f}"
            )

        with col2:

            if prob > threshold:
                st.error("⚠️ Fraud Transaction")
            else:
                st.success("✅ Legitimate Transaction")

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

        # ---------------- FEATURE IMPORTANCE ----------------
        st.subheader("📊 Top Features Affecting Prediction")

        try:

            if hasattr(model, "feature_importances_"):

                feat_df = pd.DataFrame({
                    "feature": feature_columns,
                    "importance": model.feature_importances_
                })

                feat_df = feat_df.sort_values(
                    by="importance",
                    ascending=False
                ).head(10)

                st.dataframe(
                    feat_df,
                    width="stretch"
                )

            else:
                st.warning("Feature importance not available")

        except Exception as e:
            st.warning(f"Unable to load feature importance: {e}")

        # ---------------- FRAUD DISTRIBUTION ----------------
        if "isFraud" in df.columns:

            st.subheader("📈 Fraud Distribution")

            fraud_counts = df["isFraud"].value_counts()

            st.bar_chart(fraud_counts)

        # ---------------- DOWNLOAD REPORT ----------------
        report_buffer = generate_report(sample, prob)

        st.download_button(
            label="📥 Download Report",
            data=report_buffer,
            file_name="fraud_report.csv",
            mime="text/csv"
        )


# =========================================================
# MANUAL INPUT MODE
# =========================================================
else:

    st.title("💳 Fraud Detection System")

    st.subheader("✍️ Manual Input")

    col1, col2 = st.columns(2)

    with col1:

        TransactionAmt = st.number_input(
            "TransactionAmt",
            value=100.0
        )

        card1 = st.number_input(
            "card1",
            value=1000
        )

        card2 = st.number_input(
            "card2",
            value=100
        )

    with col2:

        card3 = st.number_input(
            "card3",
            value=150
        )

        addr1 = st.number_input(
            "addr1",
            value=200
        )

        addr2 = st.number_input(
            "addr2",
            value=80
        )

    if st.button("🔍 Predict"):

        sample = df.iloc[[0]].copy()

        sample["TransactionAmt"] = TransactionAmt
        sample["card1"] = card1
        sample["card2"] = card2
        sample["card3"] = card3
        sample["addr1"] = addr1
        sample["addr2"] = addr2

        # ---------------- PREDICTION ----------------
        prob = predict_transaction(sample)

        # ---------------- RESULT ----------------
        st.subheader("📢 Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{prob:.6f}"
            )

        with col2:

            if prob > threshold:
                st.error("⚠️ Fraud Transaction")

            else:
                st.success("✅ Legitimate Transaction")

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

        # ---------------- DOWNLOAD REPORT ----------------
        report_buffer = generate_report(sample, prob)

        st.download_button(
            label="📥 Download Report",
            data=report_buffer,
            file_name="manual_fraud_report.csv",
            mime="text/csv"
        )