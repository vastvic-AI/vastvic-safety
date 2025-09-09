import streamlit as st
import numpy as np
import pandas as pd
from insurance.insurrance_recommendation import InsuranceRecommendationEngine, generate_synthetic_data

st.set_page_config(page_title="Insurance Recommendation", layout="centered")
st.title("Insurance Recommendation System")

st.markdown("""
Enter your details to get personalized insurance product recommendations based on your profile, behavior, and walking pattern.
""")

# --- Input form ---
with st.form("insurance_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        occupation = st.selectbox("Occupation", ["Medical", "Legal", "Engineer", "Academic", "Financial", "Other"])
        smoking_habit = st.selectbox("Smoking Habit", ["Smoker", "Non-smoker"])
        policy_time = st.slider("Policy Time (years)", 1, 30, 5)
        premium = st.number_input("Current Premium", min_value=0.0, value=1000.0)
    with col2:
        user_behavior_score = st.slider("User Behavior Score (0=Low, 1=High)", 0.0, 1.0, 0.7)
        exit_strategy = st.selectbox("Exit Strategy", ["Early Exit", "Long-Term", "Churn Risk"])
        walking_pattern_score = st.slider("Walking Pattern Score (0=Low, 1=High)", 0.0, 1.0, 0.8)
        user_type = st.radio("User Type", ["Existing", "New"], horizontal=True)
        user_id = st.number_input("User ID (for existing user)", min_value=0, value=42) if user_type == "Existing" else None
    submitted = st.form_submit_button("Get Recommendation")

# --- Model Initialization (cache for performance) ---
@st.cache_resource(show_spinner=True)
def get_engine():
    df = generate_synthetic_data()
    engine = InsuranceRecommendationEngine()
    engine.train(df)
    return engine

engine = get_engine()

# --- Recommendation Logic ---
if submitted:
    customer = {
        'age': age,
        'gender': gender,
        'occupation': occupation,
        'smoking_habit': smoking_habit,
        'policy_time': policy_time,
        'premium': premium,
        'user_behavior_score': user_behavior_score,
        'exit_strategy': exit_strategy,
        'walking_pattern_score': walking_pattern_score
    }
    with st.spinner("Generating recommendations..."):
        if user_type == "Existing":
            result = engine.recommend(customer, user_id=int(user_id))
        else:
            result = engine.recommend(customer)
    if 'error' in result:
        st.error(f"Error: {result['error']}")
    else:
        st.success(f"Recommendations for {user_type} user:")
        st.write({
            'Source': result['source'],
            'Products': result['recommendations'],
            'Estimated Premium': f"${result['estimated_premium']:.2f}",
            'Request ID': result['request_id'],
            'Timestamp': result['timestamp']
        })
        st.markdown("---")
        st.caption("Model uses your profile, behavior, exit strategy, and walking pattern for personalized suggestions.")
