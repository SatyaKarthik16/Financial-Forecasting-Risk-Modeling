"""
ETL pipeline for financial transactions.
Loads raw data, validates required columns, cleans values, and engineers model features.
"""

import os
import sys
import numpy as np
import pandas as pd


def load_raw(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    print(f"[ETL] Loaded {len(df):,} rows from {filepath}")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "TransactionID", "CustomerID", "Date", "Amount",
        "PaymentType", "CreditScore", "Defaulted"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[ETL] Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=list(required_cols))
    df = df.drop_duplicates(subset=["TransactionID"])
    df = df[df["Amount"] > 0]
    after = len(df)
    if before != after:
        print(f"[ETL] Dropped {before - after:,} invalid rows; {after:,} remain")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CreditScore"] = df["CreditScore"].clip(300, 850)
    df["PaymentType"] = df["PaymentType"].str.strip().str.title()
    df["Defaulted"] = df["Defaulted"].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["AmountLog"] = np.log1p(df["Amount"])
    df["IsHighValue"] = (df["Amount"] > df["Amount"].quantile(0.75)).astype(int)
    df = pd.get_dummies(df, columns=["PaymentType"], drop_first=True, dtype=int)
    return df


def run_etl(filepath: str) -> pd.DataFrame:
    df = load_raw(filepath)
    df = validate(df)
    df = clean(df)
    df = engineer_features(df)
    print(f"[ETL] Pipeline complete. Output shape: {df.shape}")
    print(f"[ETL] Default rate: {df['Defaulted'].mean():.2%}")
    return df


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_transactions.csv"
    out = run_etl(path)
    print(out.head())
