"""
preprocess.py
-------------
Feature engineering and preprocessing pipeline for the credit scoring model.

Usage (as module):
    from src.preprocess import load_and_preprocess

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'credit_data.csv')
TEST_SIZE = 0.20
RANDOM_SEED = 42

FEATURE_COLS = [
    'age',
    'income',
    'debt',
    'loan_amount',
    'payment_history',
    'num_credit_accounts',
    'credit_utilization',
    'employment_years',
    'num_late_payments',
    'debt_to_income_ratio',
]
TARGET_COL = 'creditworthy'


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python src/generate_data.py` first."
        )
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features on top of the raw columns.

    New features:
        loan_to_income_ratio   : loan_amount / income
        payment_utilization    : payment_history * (1 - credit_utilization)
        risk_score             : composite heuristic risk indicator
    """
    df = df.copy()
    df['loan_to_income_ratio'] = (df['loan_amount'] / df['income']).clip(0, 10)
    df['payment_utilization'] = df['payment_history'] * (1 - df['credit_utilization'])
    df['risk_score'] = (
        df['num_late_payments'] * 5
        + df['credit_utilization'] * 50
        + df['debt_to_income_ratio'] * 10
    )
    return df


def load_and_preprocess(
    path: str = DATA_PATH,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
):
    """
    Full preprocessing pipeline:
        1. Load CSV
        2. Engineer features
        3. Train/test split
        4. Return arrays + feature names

    Returns
    -------
    X_train, X_test : np.ndarray  (scaled)
    y_train, y_test : np.ndarray
    feature_names   : list[str]
    scaler          : fitted StandardScaler
    """
    df = load_data(path)
    df = engineer_features(df)

    all_features = FEATURE_COLS + ['loan_to_income_ratio', 'payment_utilization', 'risk_score']
    X = df[all_features].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, all_features, scaler


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, features, scaler = load_and_preprocess()
    print("Preprocessing complete.")
    print(f"  Train shape : {X_train.shape}")
    print(f"  Test shape  : {X_test.shape}")
    print(f"  Features    : {features}")
