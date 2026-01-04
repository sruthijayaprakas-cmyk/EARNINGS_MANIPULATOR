
import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Earnings Manipulator Detection")

st.title("📊 Earnings Manipulator Detection App")
st.write("Enter financial values to predict earnings manipulation risk.")

# ----------------------------
# Load model and features
# ----------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

try:
    model, features = load_artifacts()
except Exception:
    st.error("Model files not found or could not be loaded.")
    st.stop()

# ----------------------------
# User Inputs
# ----------------------------
st.subheader("🔢 Financial Inputs")

inputs = []
for feature in features:
    value = st.number_input(
        label=str(feature),
        value=0.0,
        format="%.4f"
    )
    inputs.append(value)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.markdown("---")
    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Likely Earnings Manipulator (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Not an Earnings Manipulator (Confidence: {(1 - probability):.2%})")
