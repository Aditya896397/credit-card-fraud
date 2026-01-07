import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from data_preprocessing import preprocess_data

def evaluate():
    _, X_test, _, y_test = preprocess_data("../data/creditcard.csv")

    model = joblib.load("../models/fraud_model.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

if __name__ == "__main__":
    evaluate()
