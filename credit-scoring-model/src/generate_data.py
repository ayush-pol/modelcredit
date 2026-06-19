"""
generate_data.py
----------------
Generates a synthetic credit scoring dataset and saves it to data/credit_data.csv.

Usage:
    python src/generate_data.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 2000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'credit_data.csv')


def generate_credit_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic credit scoring dataset.

    Features:
        income               : Annual income in USD
        debt                 : Total outstanding debt in USD
        payment_history      : Payment history score (0–100)
        num_credit_accounts  : Number of open credit accounts
        credit_utilization   : Credit utilization ratio (0.0–1.0)
        employment_years     : Years at current/last employment
        num_late_payments    : Late payments in last 2 years
        loan_amount          : Requested / existing loan amount
        age                  : Applicant age (22–70)
        debt_to_income_ratio : Derived feature (debt / income)
        creditworthy         : Target (1 = creditworthy, 0 = not)
    """
    rng = np.random.default_rng(seed)

    income = rng.normal(60_000, 25_000, n_samples).clip(15_000, 250_000)
    debt = rng.exponential(15_000, n_samples).clip(0, 150_000)
    payment_history = rng.normal(72, 18, n_samples).clip(0, 100)
    num_credit_accounts = rng.integers(1, 15, n_samples)
    credit_utilization = rng.beta(2, 5, n_samples)           # skewed toward low utilization
    employment_years = rng.exponential(5, n_samples).clip(0, 40)
    num_late_payments = rng.integers(0, 12, n_samples)
    loan_amount = rng.normal(25_000, 15_000, n_samples).clip(1_000, 100_000)
    age = rng.integers(22, 71, n_samples)

    debt_to_income_ratio = (debt / income).clip(0, 5)

    # ----- Creditworthiness label (rule-based probability) -----
    score = (
        0.30 * (payment_history / 100)
        + 0.25 * (1 - credit_utilization)
        + 0.20 * (1 - debt_to_income_ratio / 5)
        + 0.15 * (1 - num_late_payments / 12)
        + 0.10 * (employment_years / 40)
    )
    # Add noise
    score += rng.normal(0, 0.08, n_samples)
    score = score.clip(0, 1)

    # Threshold at 0.50 for binary label
    creditworthy = (score >= 0.50).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income.round(2),
        'debt': debt.round(2),
        'loan_amount': loan_amount.round(2),
        'payment_history': payment_history.round(1),
        'num_credit_accounts': num_credit_accounts,
        'credit_utilization': credit_utilization.round(4),
        'employment_years': employment_years.round(1),
        'num_late_payments': num_late_payments,
        'debt_to_income_ratio': debt_to_income_ratio.round(4),
        'creditworthy': creditworthy,
    })

    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = generate_credit_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved → {os.path.abspath(OUTPUT_PATH)}")
    print(f"Shape        : {df.shape}")
    print(f"Class balance:\n{df['creditworthy'].value_counts(normalize=True).round(3)}")


if __name__ == '__main__':
    main()
