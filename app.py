import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------- Page Config ----------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 AI-Powered Customer Churn Prediction")
st.write("Upload customer data to predict churn risk.")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# ---------- File Upload ----------
uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # ---------- Preprocessing ----------
    df_processed = pd.get_dummies(df)

    # Align columns with training data (add missing cols as 0, drop extra cols)
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)

    # ---------- Predictions ----------
    probs = model.predict_proba(df_processed)[:, 1]
    preds = model.predict(df_processed)

    def risk_level(prob):
        if prob >= 0.7:
            return "High Risk"
        elif prob >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    results = df.copy()
    results["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in preds]
    results["Churn_Probability"] = probs.round(2)
    results["Risk_Level"] = [risk_level(p) for p in probs]

    # ---------- Results Table ----------
    st.subheader("Prediction Results")
    st.dataframe(results)

    # ---------- Summary Metrics ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk", (results["Risk_Level"] == "High Risk").sum())
    col2.metric("Medium Risk", (results["Risk_Level"] == "Medium Risk").sum())
    col3.metric("Low Risk", (results["Risk_Level"] == "Low Risk").sum())

    # ---------- Pie Chart ----------
    st.subheader("Risk Distribution")
    risk_counts = results["Risk_Level"].value_counts()
    fig, ax = plt.subplots()
    colors = {"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#2ecc71"}
    ax.pie(
        risk_counts,
        labels=risk_counts.index,
        autopct="%1.1f%%",
        colors=[colors[r] for r in risk_counts.index],
        startangle=90,
    )
    st.pyplot(fig)

    # ---------- Download Results ----------
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload a CSV file to get started.")
