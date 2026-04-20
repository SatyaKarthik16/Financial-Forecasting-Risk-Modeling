# Financial Forecasting and Credit Risk Modeling

End-to-end credit risk assessment and revenue forecasting system for a consumer lending portfolio.

---

## Overview

This project solves two connected business problems for consumer lenders:

1. **Predict customer default risk** before losses occur using machine learning.
2. **Forecast near-term revenue** for planning, reserves, and portfolio strategy.

The repository delivers a complete workflow covering data generation, ETL, feature engineering, model training, explainability, customer risk scoring, ARIMA forecasting, and business impact reporting — both as standalone scripts and as a unified Jupyter notebook.

---

## Key Features

| Area | Details |
|---|---|
| Data Pipeline | Schema validation, cleaning, and feature engineering |
| Credit Modeling | Logistic Regression, Random Forest, XGBoost with SMOTE |
| Evaluation | ROC-AUC and PR-AUC optimized for class imbalance |
| Explainability | SHAP values (XGBoost) with fallback feature importance |
| Risk Scoring | Per-customer default probability and risk tier assignment |
| Forecasting | ARIMA with AIC-based order selection and confidence intervals |
| Business Impact | Model-vs-baseline cost and loss comparison |
| SQL Analytics | Six production-style queries for portfolio monitoring |

---

## Repository Structure

```text
Financial-Forecasting-Risk-Modeling/
├── Financial_Risk_Analysis.ipynb   # End-to-end notebook
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── synthetic_transactions.csv  # Sample dataset
├── scripts/
│   ├── generate_data.py            # Synthetic data generator
│   ├── etl_pipeline.py             # Data validation and feature engineering
│   ├── risk_model.py               # Model training and evaluation
│   ├── forecasting.py              # ARIMA revenue forecasting
│   └── risk_scoring.py             # Customer risk scoring
├── sql/
│   └── risk_segmentation.sql       # SQL analytics queries
├── models/                         # Generated during runs (not tracked)
└── images/                         # Generated during runs
```

---

## Visual Results

### Model Evaluation

**Confusion Matrices**
![Confusion Matrices](images/confusion_matrices.png)

**ROC and Precision-Recall Curves**
![ROC and PR Curves](images/roc_pr_curves.png)

### Explainability

**SHAP Feature Importance**
![Feature Importance](images/shap_feature_importance.png)

### Forecasting

**ARIMA Revenue Forecast**
![Revenue Forecast](images/revenue_forecast_arima.png)

### Risk Portfolio Dashboard

**Customer Risk Tiers and Expected Loss**
![Portfolio Risk Dashboard](images/portfolio_risk_dashboard.png)

### Business Impact

**Model vs. Baseline Cost Comparison**
![Business Impact](images/business_impact.png)

---

## Core Components

### 1. ETL and Feature Engineering — `scripts/etl_pipeline.py`

- Validates required schema columns
- Removes invalid rows and out-of-range values
- Engineers model features:
  - `AmountLog` — log-scaled transaction amount
  - `IsHighValue` — binary flag for high-value transactions
  - `DayOfWeek` — day of week from transaction date
  - `PaymentType_*` — one-hot encoded payment type features

### 2. Credit Risk Modeling — `scripts/risk_model.py`

- Trains Logistic Regression, Random Forest, and XGBoost classifiers
- Applies SMOTE to handle class imbalance
- Selects best model by PR-AUC
- Saves best model to `models/credit_risk_model.pkl`
- Exports metrics to `credit_risk_model_report.csv`
- Generates visualizations:
  - `images/confusion_matrices.png`
  - `images/roc_pr_curves.png`
  - `images/shap_feature_importance.png`
  - `images/business_impact.png`

### 3. Revenue Forecasting — `scripts/forecasting.py`

- Aggregates transactions to a monthly revenue series
- Runs ADF stationarity test
- Selects ARIMA order via AIC grid search
- Evaluates holdout MAE and MAPE
- Exports `revenue_forecast.csv` and `images/revenue_forecast_arima.png`

### 4. Risk Scoring and Segmentation — `scripts/risk_scoring.py`

- Loads the saved model and scores default probability per transaction
- Aggregates to customer level and assigns risk tiers: **Low / Medium / High / Very High**
- Computes expected credit loss by tier
- Exports:
  - `customer_risk_scores.csv`
  - `portfolio_risk_summary.csv`
  - `images/portfolio_risk_dashboard.png`

### 5. SQL Analytics — `sql/risk_segmentation.sql`

Six production-style queries for portfolio monitoring:

| Query | Purpose |
|---|---|
| Monthly Revenue & Default Summary | Revenue trend and default rate over time |
| Customer Risk Tier Segmentation | Count and exposure by risk tier |
| Default Rate by Payment Type | Segment-level default risk |
| High-Risk Watchlist | Customers exceeding default threshold |
| Expected Credit Loss by Tier | ECL-style loss estimate |
| Rolling 3-Month Default Trend | Moving average of default rate |

---

## Notebook

**`Financial_Risk_Analysis.ipynb`** mirrors the full scripted pipeline in a single interactive document:

- Exploratory data analysis and feature review
- Model training, benchmarking, and evaluation
- Explainability section with SHAP charts
- Customer risk scoring and tier assignment
- ARIMA revenue forecasting
- Business impact analysis

---

## How to Run

**Step 1 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2 — Generate synthetic data** *(optional — sample data already included)*

```bash
python scripts/generate_data.py --rows 10000 --customers 2000 --seed 42 --output data/synthetic_transactions.csv
```

**Step 3 — Train models and generate model artifacts**

```bash
python scripts/risk_model.py --data data/synthetic_transactions.csv
```

**Step 4 — Run ARIMA revenue forecasting**

```bash
python scripts/forecasting.py --data data/synthetic_transactions.csv --horizon 6
```

**Step 5 — Run customer risk scoring**

```bash
python scripts/risk_scoring.py --data data/synthetic_transactions.csv --model models/credit_risk_model.pkl
```

**Step 6 — Run the notebook**

Open `Financial_Risk_Analysis.ipynb` and run all cells to reproduce the full analysis interactively.

---

## Notes

The `models/` directory and CSV output files are excluded from version control. Run the scripts to regenerate them locally.

---

## Contact

**Satya Karthik**  
satyakarthik.y@gmail.com
