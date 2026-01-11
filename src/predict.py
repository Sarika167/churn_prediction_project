import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")
model_cols = joblib.load("models/model_columns.pkl")

def predict_customer(input_dict):
    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df)
    df = df.reindex(columns=model_cols, fill_value=0)

    pred = model.predict(df)[0]

    return "Customer will CHURN ❌" if pred == 1 else "Customer will NOT churn ✅"


# Example input (as dictionary)
example = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 3,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 70000
}

print(predict_customer(example))
