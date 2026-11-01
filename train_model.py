"""Train and evaluate a baseline credit-risk model."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE = Path(__file__).parent
DATA = BASE / "data" / "loan_data.csv"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

NUMERIC = [
    "age", "annual_income", "employment_years", "loan_amount",
    "loan_term_months", "credit_score", "existing_debt",
    "num_credit_lines", "previous_defaults"
]
CATEGORICAL = ["home_ownership", "education", "loan_purpose"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "defaulted"

def main():
    if not DATA.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_data.py")

    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])

    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_rows": int(len(y_test)),
        "positive_class": "defaulted (1)"
    }

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_DIR / "credit_risk_model.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["No default", "Default"])
    ax.set_yticks([0, 1], ["No default", "Default"])
    for (i, j), value in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0)
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_DIR / 'credit_risk_model.joblib'}")

if __name__ == "__main__":
    main()
