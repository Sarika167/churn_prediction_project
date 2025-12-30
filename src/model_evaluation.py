from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .data_preprocessing import preprocess_data
from .model_training import train_model, save_model


def run():
    X_train, X_test, y_train, y_test = preprocess_data("data/Churn_data.csv")

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    save_model(model)


if __name__ == "__main__":
    run()
