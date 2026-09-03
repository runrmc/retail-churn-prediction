# Retail Customer Churn Prediction

Predicting customer churn for a UK-based online gift retailer using real, messy transaction data — with a self-engineered churn label (no pre-built target variable) and a business-facing evaluation, not just ML metrics.

## Problem

Customer retention is far cheaper than acquisition, but knowing *who's actually at risk* of leaving is non-trivial when a business has no explicit "churned" flag in its data. This project builds a full pipeline — from raw transactions to a deployable churn model — to identify at-risk customers and quantify the revenue tied to them.

## Dataset

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (accessed via Kaggle mirror through `kagglehub`): ~1.07M transactions from a UK-based online retailer between Dec 2009 and Dec 2011, covering ~5,900 customers. Deliberately chosen *without* a pre-built churn label, requiring the label itself to be engineered — a closer analog to real-world business problems than most tutorial datasets.

## Approach

1. **EDA**: missing values, cancellation patterns, date range, customer distribution
2. **Cleaning**: dropped ~243K rows missing `Customer ID`; split cancelled orders from purchases to preserve return behavior as a feature rather than discarding it
3. **Churn labeling**: tested 90-day and 180-day windows against a fixed reference date; selected 180 days based on the dataset's observed ~180-day median purchase cycle (90 days over-flagged normal infrequent buyers as "churned")
4. **Feature engineering**: RFM (recency, frequency, monetary) + return rate, computed strictly on pre-reference-date data to avoid leakage
5. **Modeling**: logistic regression baseline, then XGBoost — tuned via grid search (5-fold CV) after untuned XGBoost initially underperformed the baseline
6. **Evaluation**: standard ML metrics (ROC-AUC, precision/recall) plus a business-facing "revenue at risk" metric

## Results

| Metric | Logistic Regression | XGBoost (default) | XGBoost (tuned) |
|---|---|---|---|
| ROC-AUC | 0.791 | 0.773 | **0.806** |
| Accuracy | 0.70 | 0.68 | **0.73** |
| Recall (churned) | 0.68 | 0.68 | **0.77** |
| Precision (churned) | 0.70 | 0.67 | 0.69 |

**Final model**: tuned XGBoost (`max_depth=2`, `n_estimators=50`, `learning_rate=0.1`) — a deliberately shallow model, appropriate given the small feature set and dataset size.

**Feature importance**: `frequency` (0.47) was the strongest predictor, ahead of `recency_days` (0.28) and `monetary` (0.23) — despite recency showing the most visually obvious relationship with churn during EDA. `return_rate` contributed minimally (0.02).

**Key finding — revenue at risk**: while the model catches 77% of churners by customer count, it only captures 49.9% of the associated revenue. Missed churners (false negatives) have an average historical spend of **$2,163**, over 3x the $649 average of correctly identified churners — meaning the model systematically underperforms on high-value accounts, likely because large/infrequent wholesale-style purchasing patterns resemble the "about to churn" signal. This is flagged as a limitation: for high-value customers specifically, model output should supplement — not replace — manual account review.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, matplotlib, Jupyter

## Project Structure

```
retail-churn-prediction/
├── notebooks/
│   └── 01_eda.ipynb       # Full pipeline: EDA → cleaning → labeling → features → modeling
├── reports/
│   └── feature_importance.png
├── data/                  # Not tracked (raw fetched via kagglehub; processed data gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dataset is fetched automatically via `kagglehub` on first notebook run (requires a free Kaggle account and API token).