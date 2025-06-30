import streamlit as st
import requests
import pandas as pd
import joblib
import os
import json

# Set page config
st.set_page_config(
    page_title="🏡 House Price Predictor",
    layout="wide",
    page_icon="📈"
)

# Load locations from preprocessor
try:
    base_dir = os.path.dirname(__file__)
    preprocessor = joblib.load(os.path.join(base_dir, "..", "model_artifacts", "preprocessor.pkl"))
    ohe = preprocessor.named_transformers_["cat"]
    locations = ohe.categories_[0].tolist()
    locations.sort()
except Exception as e:
    st.error("⚠️ Could not load location options. Check that 'preprocessor.pkl' exists.")
    st.exception(e)
    locations = []

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("""
    Stacking-Based Ensemble Model Integrating **CatBoost**, **Random Forest** and  **Multi-Layer Perceptron (MLP)**

    ###  Highlights
    - **Boosted Accuracy**: Combines three top models for better predictions  
    - **Handles Real-World Data**: Works well with mixed and messy data  
    - **Consistent Results**: Stable even with noisy or unseen inputs  
    - **Business-Ready**: Cross-validated for reliable performance  
    """)

    if st.button("📄 Show Model Metadata"):
        try:
            metadata_path = os.path.join(base_dir, "..", "model_artifacts", "model_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            st.json(metadata)
        except Exception as e:
            st.warning("Metadata could not be loaded.")
            st.exception(e)

# Custom CSS for UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    label[data-testid="stFormLabel"] > div:first-child {
        display: none;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #d3d3d3;
        padding: 10px;
        border-radius: 8px;
        font-size: 16px;
    }
    .stNumberInput input {
        background-color: #ffffff;
        border: 1px solid #d3d3d3;
        border-radius: 8px;
        padding: 8px;
        font-size: 16px;
    }
    .stSelectbox > div {
        background-color: #ffffff;
        border: 1px solid #d3d3d3;
        border-radius: 8px;
        padding: 6px;
        font-size: 16px;
    }
    button[kind="primary"] {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    button[kind="primary"]:hover {
        background-color: #45a049;
    }
    section.main > div {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("🏢 Real Estate Price Predictor")
st.subheader("Predict House Prices in Bangalore")

# Input form
with st.form("prediction_form"):
    st.write("### 🏠 Enter Property Details")

    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox(" Location", options=locations)
        bhk = st.number_input(" BHK (Bedrooms)", min_value=1, max_value=10, value=2)
    with col2:
        total_sqft = st.number_input(" Total Sqft", min_value=300.0, max_value=10000.0, value=1000.0)
        bath = st.number_input(" Bathrooms", min_value=1, max_value=5, value=2)

    submitted = st.form_submit_button("Predict Price")

# Submit and prediction
if submitted:
    if not location:
        st.warning("⚠️ Please select a location.")
    else:
        with st.spinner("⏳ Sending request to model..."):
            try:
                api_url = "http://localhost:8000/predict"
                response = requests.post(api_url, json={
                    "location": location,
                    "total_sqft": total_sqft,
                    "bath": bath,
                    "bhk": bhk
                })

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"💰 Estimated Price: ₹ {result['predicted_price_lakhs']} Lakhs")
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")

# Footer
st.markdown("---")
st.markdown(" • Made with Streamlit & FastAPI • ")
