# Bank Loan Approval & Credit Risk Prediction System

A complete educational Python project that estimates the probability of loan default using a Random Forest classifier and presents results in a Streamlit web app.

> **Important:** This is a classroom/portfolio prototype using synthetic data. It is not a real credit-underwriting system and must not be used to approve, reject, price, or otherwise determine real loans. A real system needs representative data, legal/compliance review, robust validation, fairness testing, privacy/security controls, and accountable human oversight.

## Features
- Synthetic loan dataset generator (4,000 records)
- Data preprocessing with missing-value handling, scaling, and one-hot encoding
- Random Forest binary classification model
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Streamlit interface for entering applicant details
- Estimated probability of default and Low/Medium/High risk bands
- Evaluation metrics displayed in the app

## Project structure
```text
bank_loan_credit_risk_project/
├── app.py
├── generate_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── loan_data.csv              # generated
├── models/
│   └── credit_risk_model.joblib   # generated
└── reports/
    ├── metrics.json               # generated
    ├── confusion_matrix.png       # generated
    └── classification_report.txt  # generated
```

## Setup

Python 3.10+ is recommended.

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the synthetic dataset
```bash
python generate_data.py
```

### 4. Train and evaluate the model
```bash
python train_model.py
```

### 5. Launch the web app
```bash
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## Dataset fields
| Field | Meaning |
|---|---|
| age | Applicant age (synthetic) |
| annual_income | Annual income in INR |
| employment_years | Years employed |
| loan_amount | Requested loan amount in INR |
| loan_term_months | Requested term |
| credit_score | Synthetic credit score |
| existing_debt | Existing debt in INR |
| num_credit_lines | Number of credit lines |
| previous_defaults | Previous-default indicator |
| home_ownership | Rent / Mortgage / Own |
| education | Education category |
| loan_purpose | Loan purpose |
| defaulted | Target: 1 = simulated default, 0 = no simulated default |

The synthetic target is generated from a simple probabilistic formula for demonstration. It does not reflect actual bank data, actual borrower behavior, or a validated risk model.

## Model design
A scikit-learn Pipeline applies:
1. Median imputation and standard scaling to numeric fields.
2. Most-frequent imputation and one-hot encoding to categorical fields.
3. Random Forest classification with class weighting.

The target class is `defaulted=1`. The app displays `predict_proba` as an estimated default probability. The Low/Medium/High thresholds are illustrative only and are not lending policy.

## Suggested extensions
- Add cross-validation and probability calibration.
- Add SHAP or another validated explanation method.
- Add data drift and model monitoring.
- Perform subgroup performance and fairness assessments using legally permissible, appropriately governed data.
- Add authentication, audit logs, encryption, access control, and secure deployment.
- Replace synthetic data only with properly authorized, high-quality data and obtain legal/compliance approval.

## Troubleshooting
- `ModuleNotFoundError`: activate the virtual environment and run `pip install -r requirements.txt`.
- Model missing: run `python generate_data.py` followed by `python train_model.py`.
- Port busy: run `streamlit run app.py --server.port 8502`.
