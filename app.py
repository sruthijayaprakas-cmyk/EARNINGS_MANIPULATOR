import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Earnings Manipulator Detection",
    layout="centered"
)

st.title("📊 Earnings Manipulator Detection App")
st.write(
    "This application predicts whether a firm is likely engaging in earnings manipulation "
    "based on financial inputs."
)

# -----------------------------
# Load model and feature names
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names


try:
    model, feature_names = load_artifacts()
except Exception as e:
    st.error("Model files not found or failed to load.")
    st.stop()

# -----------------------------
# User input section
# -----------------------------
st.subheader("🔢 Enter Financial Inputs")

user_values = []

for feature in feature_names:
    value = st.number_input(
        label=str(feature),
        value=0.0,
        format="%.4f"
    )
    user_values.append(value)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_array = np.array(user_values).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    st.markdown("---")
    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ **Likely Earnings Manipulator**\n\n"
            f"Confidence Score: **{probability:.2%}**"
        )
    else:
        st.success(
            f"✅ **Not an Earnings Manipulator**\n\n"
            f"Confidence Score: **{(1 - probability):.2%}**"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Model deployed using Streamlit and GitHub")
