from sklearn.linear_model import LogisticRegression
from .data_preprocessing import preprocess_data
import joblib
from pathlib import Path

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def save_model(model, path="models/churn_model.pkl"):
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path="models/churn_model.pkl"):
    model = joblib.load(path)
    print("Model loaded")
    return model