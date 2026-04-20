# 💸 Financial Forecasting & Credit Risk Modeling

> End-to-end credit risk assessment and revenue forecasting system for a consumer lending portfolio built with Python, SQL, and machine learning.

## Problem Statement

Consumer lending institutions face two interconnected challenges:

1. Credit Default Risk: Identifying which customers are likely to default before it happens.
2. Revenue Forecasting: Projecting future portfolio revenue to support treasury planning and ECL provisioning.

This project delivers a complete analytical system for both using ML and time series methods.

## Business Value

| Business Problem | Solution | Impact |
|---|---|---|
| Missed defaults | XGBoost + SMOTE | Lower credit loss cost |
| Explainability requirements | SHAP feature importance | Transparent model behavior |
| Revenue planning | ARIMA 6-month forecast | Better cash-flow and reserve planning |
| Portfolio prioritization | Customer risk tiers | Targeted interventions |
| Provisioning | PD x EAD x LGD | IFRS 9 / CECL style ECL estimates |

## Key Results

- Best model: XGBoost (with SMOTE and class-imbalance handling)
- ROC-AUC > 0.85 on test set
- ARIMA forecasting with holdout validation
- Customer-level risk scoring and expected-loss outputs

## Repository Structure

```text
Financial-Forecasting-Risk-Modeling/
|-- Financial_Risk_Analysis.ipynb
|-- requirements.txt
|-- scripts/
|   |-- generate_data.py
|   |-- etl_pipeline.py
|   |-- risk_model.py
|   |-- forecasting.py
|   |-- risk_scoring.py
|-- sql/
|   |-- risk_segmentation.sql
|-- data/
|   |-- synthetic_transactions.csv
|-- models/
|-- images/
|-- README.md
```

## How to Run

```bash
pip install -r requirements.txt
python scripts/risk_model.py --data data/synthetic_transactions.csv
python scripts/forecasting.py --data data/synthetic_transactions.csv --horizon 6
python scripts/risk_scoring.py --data data/synthetic_transactions.csv --model models/credit_risk_model.pkl
```

Then open and run all cells in Financial_Risk_Analysis.ipynb.

## SQL Analytics

See sql/risk_segmentation.sql for:
- Monthly revenue/default summaries
- Risk-tier segmentation
- Payment-channel risk analysis
- High-risk watchlist
- ECL estimates
- Rolling default trends

## Contact

Satya Karthik  
satyakarthik.y@gmail.com
