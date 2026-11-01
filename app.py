from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "credit_risk_model.joblib"
METRICS_PATH = BASE / "reports" / "metrics.json"

st.set_page_config(page_title="Loan & Credit Risk Assessment", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🏦 Bank Loan Approval & Credit Risk Prediction")
st.caption("Educational decision-support prototype | Synthetic data | Human review required")

if not MODEL_PATH.exists():
    st.error("Trained model not found. First run: python generate_data.py && python train_model.py")
    st.stop()

bundle = load_model()
model = bundle["model"]

with st.sidebar:
    st.header("Applicant details")
    st.caption("Enter illustrative applicant information.")
    age = st.slider("Age", 21, 75, 30)
    annual_income = st.number_input("Annual income (₹)", min_value=100000, max_value=10000000, value=600000, step=50000)
    employment_years = st.slider("Years employed", 0, 45, 5)
    loan_amount = st.number_input("Loan amount (₹)", min_value=10000, max_value=10000000, value=500000, step=25000)
    loan_term_months = st.selectbox("Loan term", [12, 24, 36, 48, 60, 84], index=4)
    credit_score = st.slider("Credit score", 300, 850, 700)
    existing_debt = st.number_input("Existing debt (₹)", min_value=0, max_value=10000000, value=100000, step=10000)
    num_credit_lines = st.slider("Number of active credit lines", 0, 25, 4)
    previous_defaults = st.selectbox("Previous defaults recorded?", ["No", "Yes"])
    home_ownership = st.selectbox("Home ownership", ["Rent", "Mortgage", "Own"])
    education = st.selectbox("Education", ["High School", "Graduate", "Postgraduate"])
    loan_purpose = st.selectbox("Loan purpose", ["Home", "Education", "Vehicle", "Medical", "Personal", "Business"])
    run = st.button("Assess credit risk", type="primary", use_container_width=True)

st.info(
    "This prototype estimates the probability of historical default patterns in a synthetic dataset. "
    "It does not approve, decline, price, or otherwise determine a real loan."
)

if run:
    applicant = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "num_credit_lines": num_credit_lines,
        "previous_defaults": 1 if previous_defaults == "Yes" else 0,
        "home_ownership": home_ownership,
        "education": education,
        "loan_purpose": loan_purpose
    }])
    probability = float(model.predict_proba(applicant)[0, 1])
    risk = "Low" if probability < 0.25 else ("Medium" if probability < 0.50 else "High")
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated default probability", f"{probability:.1%}")
    col2.metric("Risk band", risk)
    col3.metric("Loan amount", f"₹{loan_amount:,.0f}")

    if risk == "Low":
        st.success("Lower estimated risk in this demo model. Continue with normal human-led verification.")
    elif risk == "Medium":
        st.warning("Moderate estimated risk. Additional human review may be appropriate.")
    else:
        st.error("Higher estimated risk in this demo model. Refer for careful human review; do not make an automated adverse decision.")

    st.subheader("Applicant summary")
    summary = applicant.T.reset_index()
    summary.columns = ["Field", "Value"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    fig = px.pie(
        names=["Estimated default risk", "Remaining probability"],
        values=[probability, 1-probability],
        hole=0.55,
        title="Model-estimated default probability"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("System modules")
    c1, c2, c3 = st.columns(3)
    c1.metric("Module 1", "Applicant input")
    c2.metric("Module 2", "ML risk estimate")
    c3.metric("Module 3", "Human review")
    st.write("Enter sample applicant details in the sidebar and select **Assess credit risk**.")

if METRICS_PATH.exists():
    import json
    metrics = json.loads(METRICS_PATH.read_text())
    st.divider()
    st.subheader("Model evaluation on held-out synthetic test data")
    cols = st.columns(5)
    for col, key, label in zip(
        cols, ["accuracy", "precision", "recall", "f1", "roc_auc"],
        ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"]
    ):
        col.metric(label, f"{metrics[key]:.3f}")
    st.caption(
        f"Test rows: {metrics['test_rows']}. Metrics describe this synthetic demonstration dataset, "
        "not expected real-world lending performance."
    )

st.divider()
st.caption(
    "Responsible-use note: Do not use this educational prototype for real lending decisions. "
    "Real deployment requires representative and legally obtained data, validation, calibration, "
    "bias/fairness testing, privacy and security controls, explainability, regulatory review, "
    "and accountable human oversight. Protected characteristics should not be used to make decisions."
)
