import streamlit as st
import pandas as pd
import numpy as np
from insurance.insurrance_recommendation import generate_synthetic_data

st.set_page_config(layout="wide")
st.title("Insurance Product Recommendation Demo")

st.sidebar.header("Synthetic Data Controls")
num_users = st.sidebar.slider("Number of Users", min_value=100, max_value=10000, value=1000, step=100)
num_products = st.sidebar.slider("Number of Products", min_value=5, max_value=30, value=22, step=1)

gen_btn = st.sidebar.button("Generate Synthetic User Data")

if 'user_df' not in st.session_state or gen_btn:
    st.session_state['user_df'] = generate_synthetic_data(num_users=num_users, num_products=num_products)

df = st.session_state['user_df']

st.subheader("All Users & Recommended Insurance Product")
st.dataframe(df[['user_id', 'age', 'gender', 'occupation', 'smoking_habit', 'product_name', 'premium']].rename(columns={
    'product_name': 'Recommended Product',
    'premium': 'Recommended Premium'
}))

st.markdown("---")
st.subheader("Select a User to View Details")
user_id = st.selectbox("User ID", df['user_id'])
user_row = df[df['user_id'] == user_id].iloc[0]
st.write(f"**User ID:** {user_row['user_id']}")
st.write(f"**Age:** {user_row['age']}")
st.write(f"**Gender:** {user_row['gender']}")
st.write(f"**Occupation:** {user_row['occupation']}")
st.write(f"**Smoking Habit:** {user_row['smoking_habit']}")
st.write(f"**Recommended Product:** {user_row['product_name']}")
st.write(f"**Recommended Premium:** ₹{user_row['premium']:.2f}")

st.markdown("---")
st.info("This is a demo using synthetic data and a simple rule-based recommendation. For real deployment, integrate your trained model or business logic in insurrance_recommendation.py.")
