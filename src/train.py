import os, sys, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess

def train(data_path="../data/transactions.csv", model_path="../models/fraud_model.pkl"):
    print("="*45)
    print("  Fraud Detection Model Training")
    print("="*45)

    X, y = preprocess(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    print("\nTraining Logistic Regression...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, random_state=42)
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*45)
    print("  Model Results")
    print("="*45)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
    print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.2f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.2f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.2f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.2f}")
    print("="*45)
    print(classification_report(y_test, y_pred,
          target_names=["Legitimate","Fraud"], zero_division=0))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    return model

if __name__ == "__main__":
    train()