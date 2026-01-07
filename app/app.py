import streamlit as st
import pandas as pd
import joblib
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("Upload a CSV file to train the model and detect fraud.")

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("Pre-trained model loaded")

uploaded_file = st.file_uploader("Upload credit card transactions CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Class" not in df.columns:
        st.error("CSV must contain a 'Class' column for training.")
        st.stop()

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    X = df.drop(columns=["Class"])
    y = df["Class"]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    st.success("Model trained successfully")

    preds = model.predict(X_test)
    fraud_count = int(preds.sum())

    st.markdown("---")
    st.success(f"🚨 Fraudulent Transactions Detected in Test Set: {fraud_count}")
