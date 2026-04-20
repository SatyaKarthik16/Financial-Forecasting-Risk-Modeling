"""
Customer-level risk scoring and expected credit loss reporting.
"""

import argparse
import os
import sys

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_COLS = [
    "Amount", "AmountLog", "CreditScore", "IsHighValue", "DayOfWeek",
    "PaymentType_Debit Card", "PaymentType_PayPal", "PaymentType_Wire Transfer",
]
LGD = 0.45


def load_data(filepath: str) -> pd.DataFrame:
    sys.path.insert(0, os.path.dirname(__file__))
    from etl_pipeline import run_etl
    df = run_etl(filepath)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    return df


def assign_tier(prob: float) -> str:
    if prob < 0.10:
        return "Low Risk"
    if prob < 0.25:
        return "Medium Risk"
    if prob < 0.50:
        return "High Risk"
    return "Very High Risk"


def main(data_path: str, model_path: str):
    os.makedirs("images", exist_ok=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run scripts/risk_model.py first.")

    model = joblib.load(model_path)
    df = load_data(data_path)
    df["default_probability"] = model.predict_proba(df[FEATURE_COLS])[:, 1]

    cust = df.groupby("CustomerID").agg(
        transaction_count=("Amount", "count"),
        total_exposure=("Amount", "sum"),
        avg_credit_score=("CreditScore", "mean"),
        avg_default_prob=("default_probability", "mean"),
        max_default_prob=("default_probability", "max"),
        actual_defaults=("Defaulted", "sum"),
    ).reset_index()

    cust["risk_tier"] = cust["max_default_prob"].apply(assign_tier)
    cust["expected_loss"] = (cust["avg_default_prob"] * cust["total_exposure"] * LGD).round(2)

    tier_order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
    cust["risk_tier"] = pd.Categorical(cust["risk_tier"], categories=tier_order, ordered=True)

    summary = cust.groupby("risk_tier", observed=False).agg(
        customer_count=("CustomerID", "count"),
        total_exposure=("total_exposure", "sum"),
        avg_default_prob=("avg_default_prob", "mean"),
        total_expected_loss=("expected_loss", "sum"),
    ).reset_index().sort_values("risk_tier")

    cust.to_csv("customer_risk_scores.csv", index=False)
    summary.to_csv("portfolio_risk_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=summary, x="risk_tier", y="customer_count", color="#4c78a8", ax=axes[0])
    axes[0].set_title("Customers by Risk Tier")
    axes[0].set_xlabel("Risk Tier")
    axes[0].set_ylabel("Customer Count")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=summary, x="risk_tier", y="total_expected_loss", color="#f58518", ax=axes[1])
    axes[1].set_title("Expected Credit Loss by Tier")
    axes[1].set_xlabel("Risk Tier")
    axes[1].set_ylabel("Expected Loss")
    axes[1].tick_params(axis="x", rotation=20)

    plt.suptitle("Portfolio Risk Dashboard", fontsize=12)
    plt.tight_layout()
    plt.savefig("images/portfolio_risk_dashboard.png", dpi=150)
    plt.close()

    print("Saved customer_risk_scores.csv, portfolio_risk_summary.csv, and images/portfolio_risk_dashboard.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_transactions.csv")
    p.add_argument("--model", default="models/credit_risk_model.pkl")
    a = p.parse_args()
    main(a.data, a.model)
