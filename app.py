from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn Risk", page_icon="📊", layout="centered")

MODEL_PATH = Path(__file__).with_name("churn_model.pkl")
CHURN_THRESHOLD = 0.4
HIGH_RISK_THRESHOLD = 0.7


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run the model-export section in Customer_Churn_01.ipynb first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Telecom Customer Churn Risk")
st.caption(
    "Logistic-regression pipeline · Churn is predicted at a probability threshold of 40%."
)
st.info(
    "Risk bands: Low below 40%, Medium from 40% to below 70%, and High at or above 70%. "
    "The churn prediction cutoff is 40%."
)

with st.form("customer_details"):
    st.subheader("Customer details")
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has a partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, step=1)
    with right:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

    st.subheader("Services")
    left, right = st.columns(2)
    with left:
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )
    with right:
        tech_support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"]
        )
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.subheader("Monthly charges")
    left, right = st.columns(2)
    with left:
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.01
        )
    with right:
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            value=float(tenure * monthly_charges),
            step=0.01,
        )

    submitted = st.form_submit_button("Assess churn risk", type="primary")

if submitted:
    customer = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        }
    )

    churn_probability = float(model.predict_proba(customer)[0, 1])
    churn_prediction = churn_probability >= CHURN_THRESHOLD
    if churn_probability >= HIGH_RISK_THRESHOLD:
        risk_level = "High"
    elif churn_prediction:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    st.subheader("Assessment")
    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Churn probability", f"{churn_probability:.1%}")
    metric_middle.metric("Risk level", risk_level)
    metric_right.metric("Prediction", "Churn" if churn_prediction else "Stay")

    if churn_prediction:
        st.warning(
            "This customer is above the selected churn threshold. Consider reviewing them for retention outreach."
        )
    else:
        st.success("This customer is below the selected churn threshold.")
