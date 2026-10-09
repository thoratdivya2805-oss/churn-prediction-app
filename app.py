from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# Support both layouts: model files in the repository root OR in a models/ folder.
def find_model_file(filename):
    for candidate in (ROOT / filename, ROOT / "models" / filename):
        if candidate.exists():
            return candidate
    return None

MODEL_PATH = find_model_file("churn_model.pkl")
COLUMNS_PATH = find_model_file("model_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Customer Churn Prediction")
st.caption("Final-year project prototype | Upload a customer CSV to estimate churn risk.")
st.warning("Predictions are estimates, not guarantees. Use columns compatible with the training dataset.")

@st.cache_resource
def load_model():
    if MODEL_PATH is None or COLUMNS_PATH is None:
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(COLUMNS_PATH)

model, model_columns = load_model()
if model is None:
    st.error(
        "Trained model files were not found. Add both churn_model.pkl and "
        "model_columns.pkl either beside app.py or inside a models/ folder."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to see customer preview and predictions.")
    st.subheader("Project workflow")
    st.write("CSV upload → preprocessing → ML prediction → risk summary → downloadable results")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

st.subheader("👥 Customer Data Preview")
st.dataframe(raw_df.head(10), use_container_width=True)

# Keep only the feature columns used for training, in the original training order.
X = raw_df.reindex(columns=model_columns)

try:
    predictions = model.predict(X)
    probabilities_all = model.predict_proba(X)
    classes = list(model.classes_)
    # Find probability corresponding to class 1 (churn); fall back safely if class 1 is absent.
    if 1 in classes:
        churn_index = classes.index(1)
    else:
        churn_index = len(classes) - 1
    probabilities = probabilities_all[:, churn_index]
except Exception as exc:
    st.error(
        "Prediction failed. The uploaded CSV must contain the same feature columns "
        "and compatible data types as the training dataset."
    )
    st.code(str(exc))
    st.stop()

def risk_level(probability):
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"

results = raw_df.copy()
results["Churn Prediction"] = ["Yes" if int(p) == 1 else "No" for p in predictions]
results["Churn Probability (%)"] = (probabilities * 100).round(2)
results["Risk Level"] = [risk_level(float(p)) for p in probabilities]

st.subheader("🔮 Prediction Results")
st.dataframe(results, use_container_width=True)

total = len(results)
churn_count = int((predictions == 1).sum())
active_count = total - churn_count
churn_rate = (churn_count / total * 100) if total else 0.0

a, b, c, d = st.columns(4)
a.metric("Total Customers", total)
b.metric("Predicted Churn", churn_count)
c.metric("Predicted Not Churning", active_count)
d.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")

st.subheader("⚠️ Risk Summary")
risk_counts = results["Risk Level"].value_counts().reindex(
    ["High Risk", "Medium Risk", "Low Risk"], fill_value=0
)
r1, r2, r3 = st.columns(3)
r1.metric("High Risk", int(risk_counts["High Risk"]))
r2.metric("Medium Risk", int(risk_counts["Medium Risk"]))
r3.metric("Low Risk", int(risk_counts["Low Risk"]))

st.subheader("📈 Risk Distribution")
fig, ax = plt.subplots()
ax.pie(risk_counts.values, labels=risk_counts.index, autopct="%1.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)
plt.close(fig)

st.download_button(
    "⬇️ Download prediction results (CSV)",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="churn_predictions.csv",
    mime="text/csv",
)

 
