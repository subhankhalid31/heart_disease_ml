import os
import pickle
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_files():
    with open(os.path.join(BASE, "svc_heart_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE, "encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    return model, enc


model, enc = load_files()

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the four patient attributes below and click **Predict**.")

sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
)
vessels = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy",
    ["Zero", "One", "Two", "Three"],
)
angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

if st.button("Predict"):
    row = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "MajorVessels": [vessels],
        "ExerciseAngina": [angina],
    })
    row_enc = row.copy()
    for col in ["Sex", "ChestPain", "MajorVessels", "ExerciseAngina"]:
        row_enc[col] = enc[col].transform(row[col])

    pred = model.predict(row_enc)[0]
    label = enc["HeartDisease"].inverse_transform([pred])[0]

    if label == "Yes":
        st.error("Prediction: HEART DISEASE LIKELY")
    else:
        st.success("Prediction: NO HEART DISEASE")

st.caption("Student project for AIC354. Not for real medical use.")
