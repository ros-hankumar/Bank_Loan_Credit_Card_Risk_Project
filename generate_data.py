"""Generate a reproducible synthetic loan dataset for learning/demo purposes."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N = 4000

def main():
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.integers(21, 66, N)
    annual_income = np.clip(rng.lognormal(11.0, 0.55, N), 120000, 3000000).astype(int)
    employment_years = np.minimum(rng.integers(0, 31, N), age - 18)
    loan_amount = np.clip(rng.lognormal(12.0, 0.65, N), 50000, 2500000).astype(int)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 84], N, p=[.05,.1,.3,.15,.3,.1])
    credit_score = np.clip(rng.normal(690, 65, N), 300, 850).astype(int)
    existing_debt = np.clip(rng.lognormal(10.3, 0.8, N), 0, 1500000).astype(int)
    num_credit_lines = rng.integers(0, 13, N)
    previous_defaults = rng.binomial(1, 0.12, N)
    home_ownership = rng.choice(["Rent", "Mortgage", "Own"], N, p=[.42,.38,.20])
    education = rng.choice(["High School", "Graduate", "Postgraduate"], N, p=[.35,.45,.20])
    purpose = rng.choice(["Home", "Education", "Vehicle", "Medical", "Personal", "Business"], N)

    debt_to_income = (existing_debt + loan_amount / np.maximum(loan_term_months / 12, 1)) / np.maximum(annual_income, 1)
    loan_to_income = loan_amount / np.maximum(annual_income, 1)
    # Synthetic target: deliberately simplified and not representative of a real lender.
    logit = (
        -2.3
        + 2.0 * debt_to_income
        + 1.0 * loan_to_income
        - 0.010 * (credit_score - 650)
        + 1.25 * previous_defaults
        - 0.035 * employment_years
        + 0.12 * (num_credit_lines > 8)
    )
    probability = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, np.clip(probability, 0.02, 0.85))

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "num_credit_lines": num_credit_lines,
        "previous_defaults": previous_defaults,
        "home_ownership": home_ownership,
        "education": education,
        "loan_purpose": purpose,
        "defaulted": defaulted,
    })
    out = Path(__file__).parent / "data" / "loan_data.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Created {len(df):,} synthetic records at {out}")
    print(f"Default rate: {df.defaulted.mean():.1%}")

if __name__ == "__main__":
    main()
