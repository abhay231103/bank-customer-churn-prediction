import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "bank_churn_random_forest.joblib"
META_PATH = BASE / "model_metadata.json"
IMPORTANCE_PATH = BASE / "permutation_feature_importance.csv"

st.set_page_config(page_title="Bank Customer Churn Risk Scoring", page_icon="🏦", layout="wide")
st.title("Bank Customer Churn Risk Scoring")
st.caption("Predictive analytics demonstration. Scores are model estimates, not guarantees or decisions.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
model = load_model()
default_threshold = float(metadata["threshold_precision_constrained"])

st.sidebar.header("Risk settings")
threshold = st.sidebar.slider("Churn classification threshold", min_value=0.10, max_value=0.90, value=default_threshold, step=0.01)
st.sidebar.caption("Lower thresholds flag more customers but usually create more false positives.")

tab1, tab2, tab3 = st.tabs(["Customer risk calculator", "Feature importance", "What-if simulator"])

with tab1:
    st.subheader("Enter customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=650)
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
    with c2:
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=20, value=5)
        balance = st.number_input("Account balance", min_value=0.0, value=75000.0, step=1000.0)
        products = st.number_input("Number of products", min_value=1, max_value=4, value=2)
        has_card = st.selectbox("Has credit card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with c3:
        active = st.selectbox("Active member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        salary = st.number_input("Estimated annual salary", min_value=0.0, value=80000.0, step=1000.0)

    if st.button("Calculate churn risk", type="primary"):
        row = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": active,
            "EstimatedSalary": salary,
            "BalanceToSalaryRatio": balance / (abs(salary) + 1.0),
            "ProductDensity": products / (tenure + 1.0),
            "EngagementProductInteraction": active * products,
            "AgeTenureInteraction": age * tenure
        }])
        probability = float(model.predict_proba(row)[0, 1])
        risk = "High" if probability >= 0.60 else ("Medium" if probability >= 0.30 else "Low")
        flag = int(probability >= threshold)
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated churn probability", f"{probability:.1%}")
        m2.metric("Risk band", risk)
        m3.metric("Predicted churn flag", "Likely to churn" if flag else "Not flagged")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Classification threshold: {threshold:.0%}. Risk bands are illustrative and can be changed to match validated business policy.")

with tab2:
    st.subheader("Model feature importance")
    st.write("Permutation importance measures how much held-out ROC-AUC declines when one input feature is shuffled. It is useful for interpretation, but it does not prove causation.")
    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH).sort_values("importance_mean", ascending=True).tail(12)
        st.bar_chart(importance.set_index("feature")["importance_mean"])
        st.dataframe(importance.sort_values("importance_mean", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance file not found.")

with tab3:
    st.subheader("What-if scenario simulator")
    st.write("Compare the estimated risk for the same customer under different engagement and product-use scenarios. This is a model sensitivity exercise, not a guarantee that changing a feature will cause churn risk to change.")
    geography_w = st.selectbox("Geography", ["France", "Spain", "Germany"], key="w_geo")
    gender_w = st.selectbox("Gender", ["Female", "Male"], key="w_gender")
    age_w = st.slider("Age", 18, 100, 45, key="w_age")
    credit_w = st.slider("Credit score", 300, 900, 650, key="w_credit")
    tenure_w = st.slider("Tenure", 0, 20, 5, key="w_tenure")
    balance_w = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0, key="w_balance")
    salary_w = st.number_input("Estimated salary", min_value=0.0, value=80000.0, step=1000.0, key="w_salary")
    card_w = st.selectbox("Has credit card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="w_card")
    base_products = st.selectbox("Products — baseline", [1, 2, 3, 4], index=1, key="w_products")
    base_active = st.selectbox("Active member — baseline", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="w_active")
    scenario_products = st.selectbox("Products — scenario", [1, 2, 3, 4], index=1, key="s_products")
    scenario_active = st.selectbox("Active member — scenario", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="s_active")

    def make_row(product_count, is_active):
        return pd.DataFrame([{
            "CreditScore": credit_w, "Geography": geography_w, "Gender": gender_w,
            "Age": age_w, "Tenure": tenure_w, "Balance": balance_w,
            "NumOfProducts": product_count, "HasCrCard": card_w,
            "IsActiveMember": is_active, "EstimatedSalary": salary_w,
            "BalanceToSalaryRatio": balance_w / (abs(salary_w) + 1.0),
            "ProductDensity": product_count / (tenure_w + 1.0),
            "EngagementProductInteraction": is_active * product_count,
            "AgeTenureInteraction": age_w * tenure_w
        }])

    if st.button("Compare scenarios"):
        p_base = float(model.predict_proba(make_row(base_products, base_active))[0, 1])
        p_scenario = float(model.predict_proba(make_row(scenario_products, scenario_active))[0, 1])
        a, b, c = st.columns(3)
        a.metric("Baseline risk", f"{p_base:.1%}")
        b.metric("Scenario risk", f"{p_scenario:.1%}")
        delta = (p_scenario - p_base) * 100
        c.metric("Change", f"{delta:+.1f} percentage points")
        st.caption("A scenario comparison is not a causal estimate. Validate interventions with controlled experiments or suitable causal analysis before claiming retention impact.")

st.divider()
st.caption("Educational prototype. Do not use for real customer decisions without validation, calibration, privacy review, fairness assessment, and human oversight.")
