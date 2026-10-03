from pathlib import Path
import pickle

import streamlit as st
import pandas as pd

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "label_encoder.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, label_encoders = load_artifacts()

st.title("Demand Forecasting App")

st.divider()

st.header("Input Features")

price = st.number_input("Price", min_value=0.0, value=50.0)
discount = st.number_input("Discount (%)", min_value=0, max_value=100, value=10)
inventory_level = st.number_input("Inventory Level", min_value=0, value=100)
promotion = st.selectbox("Promotion", options=[0, 1])
competitor_pricing = st.number_input("Competitor Pricing", min_value=0.0, value=50.0)
category = st.selectbox(
    "Category", 
    label_encoders['Category'].classes_.tolist()
    )

input_data  = pd.DataFrame(
    {
        "Price": [price],
        "Discount": [discount],
        "Inventory Level": [inventory_level],
        "Promotion": [promotion],
        "Competitor Pricing": [competitor_pricing],
        "Category": [category]
    }
)

for col, encoder in label_encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

st.divider()

if st.button("Predict Demand"):
    prediction = float(model.predict(input_data)[0])
    # Round to the nearest whole unit and never show negative demand
    st.success(f"Predicted Demand: {max(0, round(prediction))} units")

