from sklearn.linear_model import LogisticRegression
import joblib
from data_preprocessing import preprocess_data

def train():
    X_train, X_test, y_train, y_test = preprocess_data("../data/creditcard.csv")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, "../models/fraud_model.pkl")
    print("Model trained and saved successfully")

if __name__ == "__main__":
    train()
