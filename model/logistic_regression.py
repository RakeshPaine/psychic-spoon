import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class LogisticRegressionModel:

  def __init__(self, data_path="../bank-additional-full.csv"):
    self.data_path = data_path
    self.model = LogisticRegression(max_iter=1000, random_state=42)
    self.scaler = StandardScaler()

  def load_and_preprocess(self):
    # Check if file exists in root or parent path
    path = (
        self.data_path
        if os.path.exists(self.data_path)
        else "bank-additional-full.csv"
    )
    df = pd.read_csv(path, sep=";")

    # Encode categorical columns
    categorical_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]
    for col in categorical_cols:
      if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Encode target variable 'y'
    if "y" in df.columns:
      df["y"] = LabelEncoder().fit_transform(df["y"])

    X = df.drop(columns=["y"])
    y = df["y"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric columns
    numeric_cols = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    numeric_cols = [col for col in numeric_cols if col in X.columns]

    X_trainpy = X_train.copy()
    X_testpy = X_test.copy()

    X_trainpy[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
    X_testpy[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

    return X_trainpy, X_testpy, y_train, y_test

  def train_and_evaluate(self):
    X_train, X_test, y_train, y_test = self.load_and_preprocess()

    # Train model
    self.model.fit(X_train, y_train)

    # Predictions
    y_pred = self.model.predict(X_test)
    y_prob = (
        self.model.predict_proba(X_test)[:, 1]
        if hasattr(self.model, "predict_proba")
        else y_pred
    )

    # Calculate the 6 required evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_test, y_pred),
    }

    print("--- Logistic Regression Metrics ---")
    for metric_name, value in metrics.items():
      print(f"{metric_name}: {value:.4f}")

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    return metrics


if __name__ == "__main__":
  # Run test execution directly if executed from inside model/ folder
  lr = LogisticModel()
  lr.train_and_evaluate()
