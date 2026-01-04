import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Earnings Manipulator Detection")

st.title("📊 Earnings Manipulator Detection App")

@st.cache_resource
def load_files():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

try:
    model, features = load_files()
except Exception as e:
    st.error("Model files not found or could not be loaded.")
    st.stop()

inputs = []
for f in features:
    inputs.append(st.number_input(str(f), value=0.0))

if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    if pred == 1:
        st.error(f"Likely Earnings Manipulator (Confidence: {prob:.2%})")
    else:
        st.success(f"Not an Earnings Manipulator (Confidence: {(1-prob):.2%})")
