"""
train.py
--------
Trains three classification models (Logistic Regression, Decision Tree,
Random Forest) and saves them to the models/ directory.

Usage:
    python src/train.py
"""

import os
import sys
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Allow running as script from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import load_and_preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RANDOM_SEED = 42
CV_FOLDS = 5


CLASSIFIERS = {
    'logistic_regression': LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0,
        solver='lbfgs',
    ),
    'decision_tree': DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
}


def train_and_save_models():
    """Train all classifiers, print cross-val scores, and save to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading & preprocessing data …")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()

    # Also save the scaler so it can be used at inference time
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")

    results = {}

    for name, clf in CLASSIFIERS.items():
        print(f"\nTraining  : {name} …")

        # Cross-validation on training set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, scoring='roc_auc')
        print(f"  CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training set
        clf.fit(X_train, y_train)

        # Persist
        model_path = os.path.join(MODELS_DIR, f'{name}.pkl')
        joblib.dump(clf, model_path)
        print(f"  Model saved → {model_path}")

        results[name] = {'clf': clf, 'cv_auc': cv_scores.mean()}

    print("\n✅ All models trained and saved.")
    return results, X_train, X_test, y_train, y_test, feature_names


if __name__ == '__main__':
    train_and_save_models()
