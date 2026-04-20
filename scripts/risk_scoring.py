import argparse
import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_COLS = ["Amount","AmountLog","CreditScore","IsHighValue","DayOfWeek","PaymentType_Debit Card","PaymentType_PayPal","PaymentType_Wire Transfer"]
LGD = 0.45


def load_data(filepath):
    sys.path.insert(0, os.path.dirname(__file__))
    from etl_pipeline import run_etl
    df = run_etl(filepath)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    return df


def assign_tier(p):
    if p < 0.10:
        return "Low Risk"
    if p < 0.25:
        return "Medium Risk"
    if p < 0.50:
        return "High Risk"
    return "Very High Risk"


def main(data_path, model_path):
    os.makedirs("images", exist_ok=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Run scripts/risk_model.py first to create models/credit_risk_model.pkl")
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
    cust.to_csv("customer_risk_scores.csv", index=False)
    summary = cust.groupby("risk_tier").agg(
        customer_count=("CustomerID", "count"),
        total_exposure=("total_exposure", "sum"),
        total_expected_loss=("expected_loss", "sum"),
    ).reset_index()
    summary.to_csv("portfolio_risk_summary.csv", index=False)
    print("Saved customer_risk_scores.csv and portfolio_risk_summary.csv")

    plt.figure(figsize=(7,4))
    summary.set_index("risk_tier")["total_expected_loss"].plot(kind="bar", color="#1f77b4")
    plt.title("Expected Loss by Risk Tier")
    plt.ylabel("Expected Loss ($)")
    plt.tight_layout()
    plt.savefig("images/expected_loss_by_tier.png", dpi=140)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_transactions.csv")
    p.add_argument("--model", default="models/credit_risk_model.pkl")
    a = p.parse_args()
    main(a.data, a.model)
