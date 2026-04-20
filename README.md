# Financial Forecasting and Credit Risk Modeling

End-to-end credit risk assessment and revenue forecasting system for a lending portfolio, built with Python, SQL, and machine learning.

## Problem Statement

Lenders face two linked challenges:
1. Predicting customer default risk before losses happen.
2. Forecasting monthly revenue for planning and provisioning.

This project provides a complete analytics workflow for both.

## Business Value

- Better default detection through imbalance-aware ML.
- Explainable risk drivers via SHAP-oriented workflow.
- Monthly revenue forecasts with uncertainty bands.
- Customer risk tiers and expected credit loss outputs.

## Key Results

- Multi-model risk pipeline: Logistic Regression, Random Forest, XGBoost
- SMOTE-based class imbalance handling
- ARIMA monthly forecasting with holdout error tracking
- Customer-level probability scoring and portfolio segmentation

## Repository Structure

```text
Financial-Forecasting-Risk-Modeling/
|-- Financial_Risk_Analysis.ipynb
|-- README.md
|-- requirements.txt
|-- data/
|   |-- synthetic_transactions.csv
|-- scripts/
|   |-- generate_data.py
|   |-- etl_pipeline.py
|   |-- risk_model.py
|   |-- forecasting.py
|   |-- risk_scoring.py
|-- sql/
|   |-- risk_segmentation.sql
|-- models/              # generated at runtime
|-- images/              # generated at runtime
```

## How to Run

```bash
pip install -r requirements.txt
python scripts/risk_model.py --data data/synthetic_transactions.csv
python scripts/forecasting.py --data data/synthetic_transactions.csv --horizon 6
python scripts/risk_scoring.py --data data/synthetic_transactions.csv --model models/credit_risk_model.pkl
```

Then run the notebook:

- Financial_Risk_Analysis.ipynb

## Outputs

Generated after running scripts:
- models/credit_risk_model.pkl
- credit_risk_model_report.csv
- revenue_forecast.csv
- customer_risk_scores.csv
- portfolio_risk_summary.csv
- images/*.png

## Contact

Satya Karthik
satyakarthik.y@gmail.com
