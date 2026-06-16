import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="AI Cyber Log Analyzer", layout="wide")

st.title("AI Cyber Log Analyzer")
st.write("Upload CICIDS2017-style network traffic data to detect possible DDoS activity.")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.write(df.head())

    original_rows = len(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if "Label" in df.columns:
        X = df.drop(columns=["Label"])
    else:
        X = df.copy()

    predictions = model.predict(X)

    results = df.copy()
    results["Prediction"] = predictions
    results["Prediction Label"] = results["Prediction"].map({
        0: "BENIGN",
        1: "DDoS"
    })

    attack_count = (results["Prediction"] == 1).sum()
    benign_count = (results["Prediction"] == 0).sum()
    attack_percentage = attack_count / len(results) * 100

    st.subheader("Scan Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows Scanned", len(results))
    col2.metric("DDoS Alerts", attack_count)
    col3.metric("Attack %", f"{attack_percentage:.2f}%")

    st.subheader("Prediction Breakdown")
    st.write(results["Prediction Label"].value_counts())

    st.subheader("Flagged DDoS Traffic")
    st.write(results[results["Prediction Label"] == "DDoS"].head(50))

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="security_scan_results.csv",
        mime="text/csv"
    )
else:
    st.info("Upload a CSV file to begin scanning.")