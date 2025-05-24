import streamlit as st
import pickle
import numpy as np

# Load model and data columns
model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))
columns = pickle.load(open('columns.pickle', 'rb'))['data_columns']
locations = columns[3:]

# Page configuration
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stMarkdown h1 {
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            color: gray;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with your info
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2356/2356778.png", width=100)
st.sidebar.title("Rohit Kumar")
st.sidebar.markdown("""
**Contact Info**  
📞 8527274237  
📧 rohitkr7518@gmail.com  
📧 rohit.kmr@iitg.ac.in
""")

st.sidebar.markdown("---")
st.sidebar.info("This app predicts House prices in Bangalore using a trained ML model.")

# Header
st.title("🏠 Bangalore House Price Predictor")
st.caption("Predict the price of your dream home in Bangalore using machine learning!")


# Input form
with st.form("prediction_form"):
    st.subheader("📋 Enter Property Details")

    col1, col2 = st.columns(2)
    with col1:
        sqft = st.number_input("🏗️ Total Square Feet", min_value=300, max_value=10000, step=50)
        bhk = st.selectbox("🛏️ Number of Bedrooms (BHK)", range(1, 11))
    with col2:
        bath = st.selectbox("🛁 Number of Bathrooms", range(1, 11))
        location = st.selectbox("📍 Select Location", sorted(locations))

    submitted = st.form_submit_button("💰 Predict Price")

# Prediction logic
if submitted:
    loc_index = columns.index(location) if location in columns else -1
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    prediction = round(model.predict([x])[0], 2)

    st.success("🏷️ Estimated Property Price")
    st.markdown(f"""
    <div style='font-size: 28px; font-weight: bold; color: #2E8B57;'>
        ₹ {prediction} Lakhs
    </div>
    <br>
    <div style='font-weight: bold;'>💼 Details</div>
    <div style='margin-left: 20px;'>
        📐 Area: {sqft} sq.ft • 🛏️ Bedrooms: {bhk} • 🛁 Bathrooms: {bath} • 📍 Location: {location}
    </div>
""", unsafe_allow_html=True)

    # Show money animation
    st.markdown("![building](https://media.giphy.com/media/W1fFapmqgqEf8RJ9TQ/giphy.gif)")

# Footer
st.markdown("---")
st.markdown("<div class='footer'>Made with ❤️ by Rohit Kumar | © 2025 Real Estate ML</div>", unsafe_allow_html=True)
