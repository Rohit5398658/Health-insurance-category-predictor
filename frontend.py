import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

st.title("🛡️ Insurance Premium Category Predictor")

st.markdown("### Fill the form to get your insurance premium category")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, max_value=200.0, value=70.0)
height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.75)
income_lpa = st.number_input("Annual Income (in LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Smoker", ["yes", "no"])
city = st.text_input("City", value="Delhi")
occupation = st.selectbox("Occupation", [
    "retired", "freelancer", "student", "government_job",
    "business_owner", "unemployed", "private_job"
])

if st.button("Predict Insurance Category"):
    # Prepare the request payload
    payload = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            prediction = response.json()["predicted_category"]
            st.success(f"✅ Predicted Insurance Category: **{prediction}**")
        else:
            st.error("❌ Error: Unable to get prediction from server.")
            st.json(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"🔌 Server Error: {e}")
