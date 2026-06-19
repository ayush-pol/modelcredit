"""
evaluate.py
-----------
Loads trained models and evaluates them on the held-out test set.
Generates and saves:
  - Classification reports (Precision, Recall, F1, Accuracy)
  - ROC curves (all models on one plot)
  - Confusion matrices
  - Feature importance plot (Random Forest)

Usage:
    python src/evaluate.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import load_and_preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
RANDOM_SEED = 42

MODEL_NAMES = ['logistic_regression', 'decision_tree', 'random_forest']
DISPLAY_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
}
COLORS = {
    'logistic_regression': '#4C72B0',
    'decision_tree': '#DD8452',
    'random_forest': '#55A868',
}


def load_models():
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}. Run `python src/train.py` first."
            )
        models[name] = joblib.load(path)
    return models


def compute_metrics(clf, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'report': classification_report(y_test, y_pred, target_names=['Not Creditworthy', 'Creditworthy']),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(models, metrics_dict, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random (AUC = 0.50)')

    for name, clf in models.items():
        fpr, tpr, _ = roc_curve(y_test, metrics_dict[name]['y_prob'])
        auc = metrics_dict[name]['roc_auc']
        ax.plot(fpr, tpr, color=COLORS[name], lw=2,
                label=f"{DISPLAY_NAMES[name]} (AUC = {auc:.3f})")

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves – Credit Scoring Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'roc_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_confusion_matrices(models, metrics_dict, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ['Not\nCreditworthy', 'Creditworthy']

    for ax, (name, clf) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, metrics_dict[name]['y_pred'])
        sns.heatmap(
            cm, annot=True, fmt='d', ax=ax,
            cmap='Blues', xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='white',
        )
        ax.set_title(DISPLAY_NAMES[name], fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)

    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'confusion_matrices.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_metrics_comparison(metrics_dict):
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    x = np.arange(len(metric_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics[k] for k in metric_keys]
        bars = ax.bar(
            x + (i - 1) * width, values, width,
            label=DISPLAY_NAMES[name],
            color=COLORS[name], alpha=0.85, edgecolor='white', linewidth=0.5,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'metrics_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_feature_importance(rf_clf, feature_names):
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sorted_features[::-1], sorted_importance[::-1],
                   color='#55A868', alpha=0.85, edgecolor='white')
    ax.set_xlabel('Feature Importance (Mean Decrease Impurity)', fontsize=11)
    ax.set_title('Random Forest – Feature Importances', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'feature_importance.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def save_summary_table(metrics_dict):
    rows = []
    for name, m in metrics_dict.items():
        rows.append({
            'Model': DISPLAY_NAMES[name],
            'Accuracy': round(m['accuracy'], 4),
            'Precision': round(m['precision'], 4),
            'Recall': round(m['recall'], 4),
            'F1-Score': round(m['f1'], 4),
            'ROC-AUC': round(m['roc_auc'], 4),
        })
    df = pd.DataFrame(rows).set_index('Model')
    path = os.path.join(FIGURES_DIR, '..', 'metrics_summary.csv')
    df.to_csv(path)
    print(f"\n📊 Metrics Summary:\n{df.to_string()}")
    print(f"\n  Table saved → {os.path.abspath(path)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data …")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()

    print("Loading models …")
    models = load_models()

    # ── Compute all metrics
    metrics_dict = {}
    for name, clf in models.items():
        metrics_dict[name] = compute_metrics(clf, X_test, y_test)

    # ── Print classification reports
    print("\n" + "=" * 60)
    for name, m in metrics_dict.items():
        print(f"\n{'─' * 60}")
        print(f"  {DISPLAY_NAMES[name]}")
        print(f"{'─' * 60}")
        print(m['report'])

    # ── Generate plots
    print("\nGenerating plots …")
    plot_roc_curves(models, metrics_dict, y_test)
    plot_confusion_matrices(models, metrics_dict, y_test)
    plot_metrics_comparison(metrics_dict)
    plot_feature_importance(models['random_forest'], feature_names)

    # ── Save summary CSV
    save_summary_table(metrics_dict)

    print("\n✅ Evaluation complete. Figures saved to reports/figures/")


if __name__ == '__main__':
    main()
