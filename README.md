credit-scoring-model/
├── src/
│   ├── generate_data.py   → synthetic dataset (2000 samples, 10 features)
│   ├── preprocess.py      → feature engineering + train/test split + scaling
│   ├── train.py           → trains LR, DT, RF with cross-validation
│   └── evaluate.py        → all metrics + 4 plots saved to reports/
├── models/                → pre-trained .pkl files (ready to load)
├── data/credit_data.csv   → generated dataset
├── notebooks/EDA_and_Modeling.ipynb  → full walkthrough notebook
├── reports/figures/       → ROC curves, confusion matrices, feature importance
├── tests/test_pipeline.py → 19 unit tests
├── requirements.txt
├── .gitignore
└── README.md              → proper GitHub README with badges, table, quickstart
