# 💳 Credit Scoring Model

A machine learning project to predict individual **creditworthiness** using past financial data. Implements and compares multiple classification algorithms with full evaluation metrics.

---

## 📌 Objective

Predict whether an individual is creditworthy (`1 = Good`, `0 = Bad`) using features derived from financial history.

---

## 🧠 Approach

Three classifiers are trained and compared:
- **Logistic Regression** — linear baseline
- **Decision Tree** — interpretable non-linear model
- **Random Forest** — ensemble method for best accuracy

---

## 📊 Dataset

Uses a synthetic dataset (auto-generated via `src/generate_data.py`) with features modeled on real-world credit data:

| Feature | Description |
|---|---|
| `income` | Annual income (USD) |
| `debt` | Total outstanding debt (USD) |
| `payment_history` | Score 0–100 (100 = perfect payments) |
| `num_credit_accounts` | Number of open credit accounts |
| `credit_utilization` | % of credit limit being used |
| `employment_years` | Years at current/last job |
| `num_late_payments` | Late payments in last 2 years |
| `loan_amount` | Requested/existing loan amount |
| `age` | Applicant age |
| `debt_to_income_ratio` | Derived: debt / income |
| `creditworthy` | **Target** (1 = creditworthy, 0 = not) |

---

## 📁 Project Structure

```
credit-scoring-model/
├── data/
│   └── credit_data.csv          # Synthetic dataset
├── notebooks/
│   └── EDA_and_Modeling.ipynb   # Exploratory analysis + model training
├── src/
│   ├── generate_data.py         # Data generation script
│   ├── preprocess.py            # Feature engineering & preprocessing
│   ├── train.py                 # Model training
│   └── evaluate.py              # Evaluation metrics & plots
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
├── reports/
│   └── figures/                 # Saved plots (ROC, confusion matrix, etc.)
├── tests/
│   └── test_pipeline.py         # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/credit-scoring-model.git
cd credit-scoring-model
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
python src/generate_data.py
```

### 3. Train all models
```bash
python src/train.py
```

### 4. Evaluate & generate reports
```bash
python src/evaluate.py
```

---

## 📈 Evaluation Metrics

Each model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

Confusion matrices and ROC curves are saved to `reports/figures/`.

---

## 🧪 Run Tests
```bash
pytest tests/
```

---

## 📋 Sample Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.82 | ~0.83 | ~0.88 | ~0.85 | ~0.89 |
| Decision Tree | ~0.85 | ~0.86 | ~0.89 | ~0.87 | ~0.87 |
| Random Forest | ~0.91 | ~0.92 | ~0.93 | ~0.92 | ~0.96 |

> *Exact numbers vary by random seed.*

---

## 🛠️ Tech Stack

- Python 3.9+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- joblib (model persistence)
- pytest

---

## 📄 License

MIT License
