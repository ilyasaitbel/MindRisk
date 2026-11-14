
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/mental_health_model.joblib")

st.title("MindRisk")
st.subheader("Student Mental Health Risk Prediction")

st.write("Enter student information:")

# Numerical features
age = st.number_input("Age", min_value=15, max_value=100, value=20)

year = st.number_input("Year of Study", min_value=1, max_value=10, value=1)

sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)

study = st.number_input("Study Hours Per Week", min_value=0.0, value=20.0)

gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)

social_media = st.number_input("Social Media Hours", min_value=0.0, value=2.0)

screen_time = st.number_input("Screen Time Hours", min_value=0.0, value=5.0)

financial_stress = st.number_input("Financial Stress", min_value=0.0, value=3.0)

family_support = st.number_input("Family Support", min_value=0.0, value=3.0)

attendance = st.number_input("Attendance Rate", min_value=0.0, max_value=100.0, value=80.0)


# Categorical features
gender = st.selectbox("Gender", ["Male", "Female"])

part_time_job = st.selectbox("Part Time Job", ["Yes", "No"])

previous_history = st.selectbox(
    "Previous Mental Health History",
    ["Yes", "No"]
)

physical_activity = st.selectbox(
    "Physical Activity",
    ["Low", "Moderate", "High"]
)

diet_quality = st.selectbox(
    "Diet Quality",
    ["Poor", "Average", "Good"]
)


# Prediction
if st.button("Predict Risk"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Year_of_Study": year,
        "Sleep_Hours": sleep,
        "Study_Hours_Per_Week": study,
        "GPA": gpa,
        "Social_Media_Hours": social_media,
        "Screen_Time_Hours": screen_time,
        "Financial_Stress": financial_stress,
        "Family_Support": family_support,
        "Attendance_Rate": attendance,
        "Gender": gender,
        "Part_Time_Job": part_time_job,
        "Previous_Mental_Health_History": previous_history,
        "Physical_Activity": physical_activity,
        "Diet_Quality": diet_quality
    }])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    st.success(f"Mental Health Risk: {prediction[0]}")
