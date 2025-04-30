import streamlit as st
import pickle
import numpy as np

# Load model and data columns
model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))
columns = pickle.load(open('columns.pickle', 'rb'))['data_columns']
locations = columns[3:]  # first 3 are ['total_sqft', 'bath', 'bhk']

# Title
st.title("🏠 House Price Predictor")

# Input fields
st.header("Enter the Property Details:")
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=50)
bhk = st.selectbox("Number of Bedrooms (BHK)", range(1, 11))
bath = st.selectbox("Number of Bathrooms", range(1, 11))
location = st.selectbox("Select Location", sorted(locations))

# Prediction button
if st.button("Predict Price"):
    # Prepare input vector
    loc_index = columns.index(location) if location in columns else -1
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict and display
    prediction = round(model.predict([x])[0], 2)
    st.success(f"Estimated Price: ₹ {prediction} Lakhs")

