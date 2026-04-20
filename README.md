# Financial Forecasting and Credit Risk Modeling

End-to-end credit risk assessment and revenue forecasting project for a consumer lending portfolio.

## Problem Statement

Lenders need to solve two connected problems:
1. Predict customer default risk before losses happen.
2. Forecast near-term revenue for planning, reserves, and portfolio strategy.

This repository provides a complete workflow that covers ETL, modeling, explainability, risk scoring, forecasting, and business impact reporting.

## What Makes This Project Complete

- Data pipeline with validation and feature engineering
- Imbalance-aware credit risk modeling (SMOTE + class weighting)
- Multi-model benchmark (Logistic Regression, Random Forest, XGBoost)
- Proper evaluation metrics for imbalanced data (ROC-AUC and PR-AUC)
- Customer-level probability scoring and risk tiers
- Revenue forecasting with ARIMA and confidence intervals
- Business impact section in notebook (cost-benefit framing)
- SQL analytics pack for portfolio monitoring and ECL-style analysis

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
|-- models/      # generated during runs
|-- images/      # generated during runs
```

## Core Components

### 1) ETL and Feature Engineering
File: scripts/etl_pipeline.py

- Validates required columns
- Cleans invalid rows and values
- Adds engineered features used by models:
  - AmountLog
  - IsHighValue
  - DayOfWeek
  - PaymentType one-hot features

### 2) Credit Risk Modeling
File: scripts/risk_model.py

- Trains Logistic Regression, Random Forest, and XGBoost
- Uses SMOTE for class imbalance
- Reports ROC-AUC, PR-AUC, and classification report
- Saves best model to models/credit_risk_model.pkl
- Writes comparison table to credit_risk_model_report.csv

### 3) Revenue Forecasting
File: scripts/forecasting.py

- Aggregates to monthly revenue
- Performs ADF stationarity test
- Selects ARIMA order via AIC search
- Evaluates holdout error (MAE, MAPE)
- Saves forecast to revenue_forecast.csv
- Saves forecast chart to images/revenue_forecast_arima.png

### 4) Risk Scoring and Segmentation
File: scripts/risk_scoring.py

- Scores customer default probability using saved model
- Assigns risk tiers (Low, Medium, High, Very High)
- Computes expected loss summary
- Saves:
  - customer_risk_scores.csv
  - portfolio_risk_summary.csv

### 5) SQL Analytics
File: sql/risk_segmentation.sql

Includes six production-style analyses:
- monthly revenue/default summary
- customer risk tier segmentation
- default rate by payment type
- high-risk watchlist
- expected credit loss by tier
- rolling 3-month default trend

## Notebook

File: Financial_Risk_Analysis.ipynb

The notebook includes:
- EDA and feature analysis
- model training/evaluation
- explainability section
- customer scoring section
- ARIMA forecast section
- Section 8: Business Impact Analysis

## How to Run

```bash
pip install -r requirements.txt
python scripts/risk_model.py --data data/synthetic_transactions.csv
python scripts/forecasting.py --data data/synthetic_transactions.csv --horizon 6
python scripts/risk_scoring.py --data data/synthetic_transactions.csv --model models/credit_risk_model.pkl
```

Then open and run all cells in Financial_Risk_Analysis.ipynb.

## Notes on Generated Artifacts

The following are runtime outputs and are intentionally ignored from version control:
- models/
- images/
- customer_risk_scores.csv
- portfolio_risk_summary.csv
- revenue_forecast.csv

## Contact

Satya Karthik
satyakarthik.y@gmail.com
