import streamlit as st
import pandas as pd
import joblib

# load model and cloumns
model = joblib.load("models/churn_model.pkl")
model_cols = joblib.road("models/model_columns.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn")

# user inputs
CreditScore = st.number_input("Credit Score", 300, 900, 600)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", 18, 100, 40)
Tenure = st.number_input("Tenure (years)", 0, 10, 3)
Balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
NumOfProducts = st.number_input("Number of Products", 1, 4, 2)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", 0.0, 200000.0, 70000.0)

if st.button("predict"):
    input_data = {
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_cols, fill_value= 0)

    prediction = model.predict(df)[0]
    
    if prediction == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")