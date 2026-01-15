import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import re

# ---------------------------------------------------------
# 1. Configuration & Model Loading
# ---------------------------------------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

@st.cache_resource
def load_model_and_cols():
    # Load the saved model and column list
    model = joblib.load('best_credit_model.pkl')
    cols = joblib.load('feature_columns.pkl')
    return model, cols

try:
    model, feature_cols = load_model_and_cols()
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure .pkl files are in the same directory.")
    st.stop()

# Initialize SHAP Explainer
explainer = shap.TreeExplainer(model)

# ---------------------------------------------------------
# 2. Sidebar: User Inputs
# ---------------------------------------------------------
st.sidebar.header("📝 Customer Information")

def user_input_features():
    # 1. Checking Account Status
    existing_account = st.sidebar.selectbox("Checking Account Balance", 
        ['< 0 DM', '0 <= x < 200 DM', '>= 200 DM / Salary assignments', 'No checking account'])
    
    # 2. Loan Duration
    duration = st.sidebar.slider("Loan Duration (Months)", 4, 72, 24)
    
    # 3. Credit Amount
    credit_amount = st.sidebar.number_input("Credit Amount (DM)", 250, 20000, 3000)
    
    # 4. Credit History
    credit_history = st.sidebar.selectbox("Credit History",
        ['No credits taken', 'All credits paid back duly', 'Existing credits paid back duly', 
         'Delay in paying off', 'Critical account / Other credits existing'])
    
    # 5. Savings Account
    savings = st.sidebar.selectbox("Savings Account",
        ['< 100 DM', '100 <= x < 500 DM', '500 <= x < 1000 DM', '>= 1000 DM', 'Unknown/No savings account'])
    
    # 6. Age
    age = st.sidebar.slider("Age", 19, 75, 30)
    
    # Create DataFrame
    data = {
        'Existing_account': existing_account,
        'Duration_month': duration,
        'Credit_amount': credit_amount,
        'Credit_history': credit_history,
        'Savings_account': savings,
        'Age': age,
        # Default values for other features
        'Purpose': 'Radio/Television',
        'Employment_since': '1 <= x < 4 years',
        'Installment_rate': 4,
        'Personal_status_sex': 'Male : Single',
        'Guarantors': 'None',
        'Residence_since': 4,
        'Property': 'Real estate',
        'Other_installment_plans': 'None',
        'Housing': 'Own',
        'Num_existing_credits': 1,
        'Job': 'Skilled employee / Official',
        'Num_people_liable': 1,
        'Telephone': 'None',
        'Foreign_worker': 'Yes'
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# ---------------------------------------------------------
# 3. Data Preprocessing
# ---------------------------------------------------------
# One-Hot Encoding
input_df_encoded = pd.get_dummies(input_df)

# Rename columns to match XGBoost requirements (Regex)
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
input_df_encoded.columns = [regex.sub("_", col) for col in input_df_encoded.columns]

# Align columns with training data
final_input = pd.DataFrame(columns=feature_cols)
for col in feature_cols:
    if col in input_df_encoded.columns:
        final_input[col] = input_df_encoded[col]
    else:
        final_input[col] = 0 

# ---------------------------------------------------------
# 4. Main Dashboard
# ---------------------------------------------------------
st.title("🏦 Credit Risk Scoring Dashboard")
st.markdown("This dashboard predicts credit risk using an **Explainable XGBoost Model**.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Prediction Result")
    
    # Model Prediction
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1] # Probability of Class 1 (Bad)
    
    if prediction == 0:
        st.success("✅ **Approved (Low Risk)**")
        st.metric("Probability of Default", f"{probability:.1%}", delta="-Low Risk")
    else:
        st.error("🚫 **Rejected (High Risk)**")
        st.metric("Probability of Default", f"{probability:.1%}", delta="+High Risk", delta_color="inverse")
    
    st.markdown("---")
    st.write("**Customer Data Preview:**")
    st.dataframe(input_df.T, height=250)

with col2:
    st.subheader("🔍 Explainability (SHAP Analysis)")
    
    with st.spinner('Calculating SHAP values...'):
        shap_values = explainer(final_input)
        
        # Waterfall Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        
    st.info("""
    **How to read this graph:**
    - **Red bars (→)**: Factors that **increase** the risk of default.
    - **Blue bars (←)**: Factors that **decrease** the risk (safe factors).
    """)