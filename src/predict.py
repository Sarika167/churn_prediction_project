import pandas as pd
import joblib
from .data_preprocessing import preprocess_data
from .model_training import load_model


def prepare_input(sample_dict):
    # load reference dataframe (for columns)
    X_train, X_test, y_train, y_test = preprocess_data("data/Churn_data.csv")

    df = pd.DataFrame([sample_dict])

    # apply same dummy encoding
    df = pd.get_dummies(df)

    # missing columns -> fill with 0
    df = df.reindex(columns=X_train.columns, fill_value=0)

    return df


def predict(sample_dict):
    model = load_model()

    X = prepare_input(sample_dict)

    pred = model.predict(X)[0]

    return "Churn" if pred == 1 else "Not Churn"


if __name__ == "__main__":
    customer = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000,
    }

    print(predict(customer))
