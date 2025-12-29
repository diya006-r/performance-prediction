import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("models/trained_model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

model, features = load_model()

# App UI
st.title("🎓 Student Performance Prediction")
st.write("Predict a student's final grade based on academic and personal features.")

st.divider()

# User inputs
user_input = []

for feature in features:
    value = st.number_input(
        f"Enter value for {feature}",
        min_value=0.0,
        step=1.0
    )
    user_input.append(value)

st.divider()

# Prediction
if st.button("Predict Final Grade"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)

    st.success(f"📊 Predicted Final Grade: {round(prediction[0], 2)}")
