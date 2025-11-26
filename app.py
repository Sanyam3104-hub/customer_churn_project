import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/churn_model.pkl")
encoders = joblib.load("model/encoders.pkl")

st.set_page_config(page_title="Customer Churn AI", layout="centered")

st.markdown("""
<style>
.main-box {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.title {
    font-size: 30px;
    font-weight: bold;
    margin-bottom: 10px;
}
.subtitle {
    color: #6b7280;
    margin-bottom: 20px;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    font-size: 18px;
}
.churn {
    background-color: #fee2e2;
    color: #991b1b;
}
.no-churn {
    background-color: #dcfce7;
    color: #166534;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-box'>", unsafe_allow_html=True)
st.markdown("<div class='title'>📊 Customer Churn Prediction AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a customer will churn using advanced ML</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    MonthlyCharges = st.number_input("Monthly Charges", 10.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

ChargesPerMonth = TotalCharges / (tenure + 1)

user_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "ChargesPerMonth": ChargesPerMonth,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod
}])

for col in user_data.columns:
    if col in encoders:
        user_data[col] = encoders[col].transform(user_data[col])

user_data = user_data.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("🔮 Predict Churn"):
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    st.progress(probability)

    if prediction == 1:
        st.markdown(
            f"<div class='result-box churn'>⚠ Customer Likely to Churn<br>Probability: {probability:.2%}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-box no-churn'>✅ Customer Likely to Stay<br>Probability: {1 - probability:.2%}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
