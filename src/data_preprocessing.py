import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(path):
    df = pd.read_csv(path)

    print("COLUMNS INSIDE FUNCTION =>")
    print(df.columns)

    # Target column
    target = "Exited"

    y = df[target]
    X = df.drop(target, axis=1)

    # convert categorical
    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y, test_size=0.2, random_state=42)
