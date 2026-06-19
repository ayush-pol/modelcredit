"""
test_pipeline.py
----------------
Unit tests for the credit scoring pipeline.

Run:
    pytest tests/
"""

import os
import sys
import numpy as np
import pytest

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generate_data import generate_credit_data
from src.preprocess import engineer_features, load_and_preprocess


# ─── Data generation tests ────────────────────────────────────────────────────

class TestGenerateData:

    def test_output_shape(self):
        df = generate_credit_data(n_samples=500)
        assert df.shape[0] == 500
        assert df.shape[1] >= 10

    def test_target_binary(self):
        df = generate_credit_data(n_samples=500)
        assert set(df['creditworthy'].unique()).issubset({0, 1})

    def test_no_nulls(self):
        df = generate_credit_data(n_samples=500)
        assert df.isnull().sum().sum() == 0

    def test_income_positive(self):
        df = generate_credit_data(n_samples=500)
        assert (df['income'] > 0).all()

    def test_credit_utilization_range(self):
        df = generate_credit_data(n_samples=500)
        assert (df['credit_utilization'] >= 0).all()
        assert (df['credit_utilization'] <= 1).all()

    def test_payment_history_range(self):
        df = generate_credit_data(n_samples=500)
        assert (df['payment_history'] >= 0).all()
        assert (df['payment_history'] <= 100).all()

    def test_reproducibility(self):
        df1 = generate_credit_data(n_samples=100, seed=1)
        df2 = generate_credit_data(n_samples=100, seed=1)
        assert df1.equals(df2)

    def test_different_seeds_differ(self):
        df1 = generate_credit_data(n_samples=100, seed=1)
        df2 = generate_credit_data(n_samples=100, seed=2)
        assert not df1.equals(df2)


# ─── Feature engineering tests ────────────────────────────────────────────────

class TestFeatureEngineering:

    def setup_method(self):
        self.df = generate_credit_data(n_samples=300)

    def test_new_columns_added(self):
        out = engineer_features(self.df)
        for col in ['loan_to_income_ratio', 'payment_utilization', 'risk_score']:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_inf_values(self):
        out = engineer_features(self.df)
        assert not np.isinf(out.select_dtypes(include='number').values).any()

    def test_no_nulls_after_engineering(self):
        out = engineer_features(self.df)
        assert out.isnull().sum().sum() == 0

    def test_input_unchanged(self):
        """Engineer features should not mutate the original DataFrame."""
        original_cols = set(self.df.columns)
        _ = engineer_features(self.df)
        assert set(self.df.columns) == original_cols

    def test_loan_to_income_non_negative(self):
        out = engineer_features(self.df)
        assert (out['loan_to_income_ratio'] >= 0).all()


# ─── Preprocessing pipeline tests ─────────────────────────────────────────────

class TestPreprocessing:

    def setup_method(self):
        # Generate data into the expected path
        from src.generate_data import generate_credit_data, OUTPUT_PATH
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df = generate_credit_data(n_samples=500)
        df.to_csv(OUTPUT_PATH, index=False)

    def test_output_types(self):
        X_train, X_test, y_train, y_test, features, scaler = load_and_preprocess()
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_train_test_proportions(self):
        X_train, X_test, y_train, y_test, _, _ = load_and_preprocess(test_size=0.2)
        total = X_train.shape[0] + X_test.shape[0]
        assert abs(X_test.shape[0] / total - 0.2) < 0.05

    def test_feature_count_matches(self):
        X_train, X_test, y_train, y_test, features, _ = load_and_preprocess()
        assert X_train.shape[1] == len(features)

    def test_no_nulls_in_arrays(self):
        X_train, X_test, _, _, _, _ = load_and_preprocess()
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_scaler_applied(self):
        """Scaled data should have approximately zero mean on training set."""
        X_train, _, _, _, _, _ = load_and_preprocess()
        col_means = X_train.mean(axis=0)
        assert np.allclose(col_means, 0, atol=1e-6)

    def test_labels_binary(self):
        _, _, y_train, y_test, _, _ = load_and_preprocess()
        assert set(np.unique(y_train)).issubset({0, 1})
        assert set(np.unique(y_test)).issubset({0, 1})
