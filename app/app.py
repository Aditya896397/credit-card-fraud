import streamlit as st
import pandas as pd
import joblib
import os
import sys

# -------------------------------------------------
# Resolve project root & fix imports
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# -------------------------------------------------
# Train model if not present (Cloud-safe)
# -------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Trained model not found. Training model...")
    from src.train_model import train
    train()

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = joblib.load(MODEL_PATH)

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Predict fraudulent credit card transactions using a machine learning model. "
    "Upload a CSV file or use sample data."
)

# -------------------------------------------------
# Option 1: Use sample data
# -------------------------------------------------
if st.button("Use Sample Transactions"):
    df = pd.read_csv(DATA_PATH)

    # Drop label if present
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    st.subheader("📄 Sample Data Preview")
    st.dataframe(df.head())

    preds = model.predict(df)
    df["Fraud Prediction"] = preds

    fraud_count = int(df["Fraud Prediction"].sum())

    st.markdown("---")
    st.success(f"🚨 Fraudulent Transactions Detected: **{fraud_count} / {len(df)}**")
    st.dataframe(df)

# -------------------------------------------------
# Option 2: Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop label automatically
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    preds = model.predict(df)
    df["Fraud Prediction"] = preds

    fraud_count = int(df["Fraud Prediction"].sum())

    st.markdown("---")
    st.success(f"🚨 Fraudulent Transactions Detected: **{fraud_count} / {len(df)}**")
    st.dataframe(df)
