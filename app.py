import streamlit as st
import pickle
import numpy as np # Import numpy as it's used later

# Load the trained model (make sure you save your trained model as model.pkl in notebook)
model = pickle.load(open("adm_2model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🎓 Graduate Admission Prediction App")
st.write("Enter the details below to predict your chance of admission.")

# Input fields
gre_score = st.number_input("GRE Score (out of 340)", min_value=0, max_value=340, step=1)
toefl_score = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, step=1)
university_rating = st.slider("University Rating (1-5)", 1, 5, 3)
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, 3.0, step=0.5)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, 3.0, step=0.5)
cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, step=0.1)
research = st.selectbox("Research Experience", [0, 1])  # 0 = No, 1 = Yes

# Predict button
if st.button("Predict Admission Chance"):
    features = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prediction = max(0, min(1, prediction))
    st.success(f"🎯 Predicted Admission Chance: {prediction*100:.2f}%")

# Display model accuracy
try:
    with open("model_accuracy.txt", "r") as f:
        accuracy = float(f.read())
    st.info(f"Model R² Score: {accuracy:.2f}")  # Shows 0.85 as 85% 
except:
    st.info("Model accuracy not available.")