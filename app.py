import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai

# -----------------------------
# Load and preprocess dataset
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    df = df.drop('customerID', axis=1)
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df, df_encoded

df, df_encoded = load_data()
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train model (Cached)
# -----------------------------
@st.cache_resource
def train_model():
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    return rf

model = train_model()

# -----------------------------
# Configure Gemini LLM
# -----------------------------
API_KEY = "MY KEY"
genai.configure(api_key=API_KEY)

# Use active gemini-3.6-flash (gemini-1.5-flash is retired/deprecated)
llm = genai.GenerativeModel("gemini-3.6-flash")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📉 Telecom Customer Churn Predictor + Advisory")
st.write("Enter customer details below to predict churn probability and retrieve the appropriate retention playbook action.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Prepare input data including TotalCharges
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges,
    "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
    "InternetService_No": 1 if internet == "No" else 0,
}

# Use a DataFrame with feature names to prevent scikit-learn warnings
input_df = pd.DataFrame([input_dict], columns=X.columns).fillna(0)

if st.button("Predict Churn", type="primary"):
    risk = float(model.predict_proba(input_df)[0][1])
    st.write(f"### Predicted Churn Probability: **{risk:.1%}**")

    # -------------------------------------------------------------
    # Retrieval Step (Playbook Clauses 1 - 4)
    # Clause 3 takes precedence for new customers (< 3 months, Any Risk)
    # -------------------------------------------------------------
    if tenure < 3:
        clause = "Clause 3 — New Customer, Any Risk, Tenure < 3 months: Route to the onboarding team instead of the standard retention flow."
    elif risk >= 0.70:
        clause = "Clause 1 — High Risk (probability >= 0.70): Offer a loyalty discount and a callback from a retention specialist within 48 hours."
    elif risk >= 0.40:
        clause = "Clause 2 — Moderate Risk (0.40–0.70): Send a targeted email highlighting an underused service or a contract upgrade offer."
    else:
        clause = "Standard Maintenance — Low Risk (< 0.40): Maintain standard customer service. No aggressive retention discount required."

    st.info(f"📋 **Playbook Policy:** {clause}")

    # LLM system prompt (Strictly excludes demographics per Clause 4)
    system_prompt = f"""
You are a retention assistant at a telecom company.
Produce a concise 3-4 sentence explanation for a retention agent.
Your explanation must be grounded strictly in the retrieved clause and operational features below.
Under Clause 4 (Non-Discrimination Rule), you must NEVER mention or imply gender, SeniorCitizen, Partner, or Dependents.

Retrieved Policy:
{clause}

Customer Features:
- Contract: {contract}
- Internet: {internet}
- Tenure: {tenure} months
- Monthly Charges: ${monthly_charges:.2f}
"""

    with st.spinner("Generating retention guidance..."):
        try:
            response = llm.generate_content(system_prompt)
            st.markdown("### 💡 AI Retention Advisory:")
            st.write(response.text.strip())
        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")
