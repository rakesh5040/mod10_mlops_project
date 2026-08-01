import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_visit_with_us_pred_purch_model_v1.joblib")
model = joblib.load(model_path)

st.title("Visit With Us Customer Purchase Prediction App")
st.write("""
This application that predicts whether a customer will purchase the newly introduced Wellness Tourism Package .
""")

Gender_str         = st.selectbox("Gender", ["Female", "Male"])
Gender = 0 if Gender_str == "Female" else 1 # Map 'Female' to 0 and 'Male' to 1
CityTier         = st.selectbox("CityTier", ["1", "2", "3"])
Age     = st.number_input("Age", 0, 100, 1)
NumberOfPersonVisiting     = st.number_input("NumberOfPersonVisiting", 1, 10, 1)
PreferredPropertyStar     = st.number_input("PreferredPropertyStar", 1, 7, 1)
NumberOfTrips     = st.number_input("NumberOfTrips", 1, 100, 1)
Passport_str     = st.selectbox("Passport", ["0", "1"])
Passport = int(Passport_str)
OwnCar_str     = st.selectbox("OwnCar", ["0", "1"])
OwnCar = int(OwnCar_str)
NumberOfChildrenVisiting     = st.number_input("NumberOfChildrenVisiting", 0, 100, 1)
MonthlyIncome     = st.number_input("MonthlyIncome", 0, 100000)
PitchSatisfactionScore     = st.number_input("PitchSatisfactionScore", 0, 10, 1)
NumberOfFollowups     = st.number_input("NumberOfFollowups", 0, 100, 1)
DurationOfPitch     = st.number_input("DurationOfPitch", 0, 100, 1)

TypeofContact     = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation     = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
MaritalStatus = st.selectbox("MaritalStatus", ['Single', 'Divorced', 'Married', 'Unmarried'])
Designation    = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
ProductPitched       = st.selectbox("ProductPitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])


input_data = pd.DataFrame([{
    "Age": Age,
    "Gender": Gender,
    "CityTier": int(CityTier),
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer Will Purchase" if prediction == 1 else "Customer Will Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
