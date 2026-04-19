import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
except:
    st.error("Model or scaler file not found!")
    st.stop()

st.title("💰 Insurance Charges Prediction App")

st.write("Enter the client details:")

# User Inputs
age = st.slider("Age", 18, 64, 30)
sex = st.radio("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.radio("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Convert input to dataframe
input_data = {
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}

input_df = pd.DataFrame([input_data])

# One-hot encoding
input_df_encoded = pd.get_dummies(input_df, columns=["sex", "smoker", "region"], drop_first=True)

# Expected columns (IMPORTANT)
expected_columns = [
    'age', 'bmi', 'children', 'sex_male', 'smoker_yes',
    'region_northwest', 'region_southeast', 'region_southwest'
]

# Add missing columns
for col in expected_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Arrange columns
input_df_encoded = input_df_encoded[expected_columns]

# Scale input
input_scaled = scaler.transform(input_df_encoded)

# Prediction
if st.button("Predict Charges"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"💵 Predicted Insurance Charges: ${prediction:.2f}")
