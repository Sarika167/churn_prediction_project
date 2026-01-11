import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# ------------ 1️⃣ LOAD DATA -------------
df = pd.read_csv("data/Churn_data.csv")

# ------------ 2️⃣ PREPROCESS -------------
df['Exited'] = df['Exited'].astype(int)

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Exited', axis=1)
y = df['Exited']

# ------------ 3️⃣ TRAIN TEST SPLIT -------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------ 4️⃣ TRAIN MODEL -------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully")

# ------------ 5️⃣ SAVE MODEL + COLUMNS -------------
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(X.columns, "models/model_columns.pkl")

print("Model & columns saved!")




