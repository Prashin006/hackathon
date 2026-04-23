import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import numpy as np


def run_classification(df: pd.DataFrame, target_col: str) -> dict:
    """
    Run Logistic Regression + SVM on any dataframe.
    Automatically encodes categorical columns.
    """
    df = df.copy()

    # Encode categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    results["Logistic Regression"] = {
        "accuracy": round(accuracy_score(y_test, lr_preds), 3),
        "report": classification_report(y_test, lr_preds, output_dict=True)
    }

    # SVM
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    results["SVM"] = {
        "accuracy": round(accuracy_score(y_test, svm_preds), 3),
        "report": classification_report(y_test, svm_preds, output_dict=True)
    }

    return results


def run_regression(df: pd.DataFrame, target_col: str) -> dict:
    """Run Linear Regression on any dataframe."""
    df = df.copy()

    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 2)

    return {
        "Linear Regression": {
            "rmse": rmse,
            "r2_score": round(lr.score(X_test, y_test), 3)
        }
    }