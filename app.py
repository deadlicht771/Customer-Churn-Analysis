import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Customer Retention AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. MODEL PARAMETERS (Cleaned from your Output)
# ==========================================
# These match your XGBoost training data exactly.

MEANS = [
    0.1352, 28.0401, 68.2382, 2088.1978, 0.5732, 0.5057, 0.2911, 0.9201, 
    0.1062, 0.4888, 0.5413, 0.17, 0.17, 0.2887, 0.17, 0.3763, 
    0.17, 0.3767, 0.17, 0.2897, 0.17, 0.4553, 0.17, 0.451, 
    0.1904, 0.1793, 0.6942, 0.2182, 0.4818, 0.2358
]

SCALES = [
    0.3419, 24.0154, 28.7458, 2201.9142, 0.4946, 0.5, 0.4543, 0.2711, 
    0.3081, 0.4999, 0.4983, 0.3756, 0.3756, 0.4532, 0.3756, 0.4845, 
    0.3756, 0.4845, 0.3756, 0.4536, 0.3756, 0.498, 0.3756, 0.4976, 
    0.3926, 0.3836, 0.4607, 0.413, 0.4997, 0.4245
]

EXPECTED_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 
    'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 
    'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check'
]

# Load XGBoost Model
@st.cache_resource
def load_model():
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error: Could not load 'xgb_model.pkl'. Please check if the file is in the same folder as app.py.")
        st.stop()

model = load_model()

# ==========================================
# 3. SIDEBAR - USER INPUTS
# ==========================================
st.sidebar.header("👤 Customer Profile")

# Demographics
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])

# Usage
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 24)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)

st.sidebar.markdown("---")
st.sidebar.subheader("Services & Contract")

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
paperless = st.sidebar.selectbox("Paperless Billing?", ["Yes", "No"])

with st.sidebar.expander("Additional Services"):
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# ==========================================
# 4. PREPROCESSING ENGINE
# ==========================================
def preprocess_input():
    # 1. Map inputs to dictionary
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'gender_Male': 1 if gender == "Male" else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        'PhoneService_Yes': 1 if phone_service == 'Yes' else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaperlessBilling_Yes': 1 if paperless == 'Yes' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if "Credit card" in payment_method else 0,
        'PaymentMethod_Electronic check': 1 if "Electronic check" in payment_method else 0,
        'PaymentMethod_Mailed check': 1 if "Mailed check" in payment_method else 0
    }

    # 2. Create the array in correct order
    row = []
    for col in EXPECTED_COLUMNS:
        row.append(input_dict.get(col, 0))
    
    # 3. Convert to numpy array
    row = np.array(row, dtype=np.float32)
    
    # 4. Apply Scaling
    row_scaled = (row - np.array(MEANS)) / np.array(SCALES)
    
    return row_scaled.reshape(1, -1)

# ==========================================
# 5. MAIN DASHBOARD
# ==========================================
st.title("🛡️ Customer Retention Strategy AI")
st.markdown("### XGBoost Powered Churn Prediction")

if st.button("Analyze Customer Risk", type="primary"):
    
    # Preprocess
    processed_data = preprocess_input()
    
    # Inference
    churn_prob = model.predict_proba(processed_data)[0][1]
    
    # Threshold (0.40 favors retention/recall)
    THRESHOLD = 0.40
    
    st.divider()
    
    # Display Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{churn_prob:.1%}")
    with col2:
        if churn_prob > THRESHOLD:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("✅ LOYAL")

    # ==========================================
    # RETENTION STRATEGY LOGIC
    # ==========================================
    if churn_prob > THRESHOLD:
        st.subheader("📋 Recommended Retention Strategy")
        st.markdown("Based on the specific risk factors identified:")
        
        strategies = []
        
        # Strategy 1: Contract
        if contract == "Month-to-month":
            strategies.append("🔒 **Contract Upgrade:** This customer is on a risky Month-to-Month plan. Offer a **20% discount** for locking in a 1-year contract.")
            
        # Strategy 2: Fiber Optic
        if internet_service == "Fiber optic":
            strategies.append("⚡ **Service Quality:** Fiber customers have high churn rates. Schedule a **free technical checkup** or speed optimization.")
            
        # Strategy 3: Payment Method
        if "Electronic check" in payment_method:
            strategies.append("💳 **Payment Method:** Electronic checks are associated with higher churn. Incentivize switching to **Auto-Pay (Credit Card)** with a $5 bill credit.")
            
        # Strategy 4: Pricing
        if monthly_charges > 80:
            strategies.append("💰 **Price Sensitivity:** Monthly bill is high ($80+). Offer a **downgrade option** to a cheaper plan without cancellation fees.")
            
        # Strategy 5: Tech Support
        if tech_support == "No":
            strategies.append("🛠️ **Support Bundle:** Lack of tech support increases frustration. Offer **3 months of Free Premium Support**.")
            
        # Fallback Strategy
        if not strategies:
            strategies.append("📧 **General Engagement:** Send a personalized 'We miss you' email with a loyalty reward.")
            
        for s in strategies:
            st.markdown(f"- {s}")
            
    else:
        st.subheader("🎉 Customer is Safe")
        st.write("No immediate action required. Consider sending a **Customer Appreciation** email.")