import argparse
import uuid
import numpy as np
import pandas as pd

def generate_transactions(n_rows=50000, seed=42):
    rng = np.random.default_rng(seed)
    tx_ids = [str(uuid.UUID(int=int(rng.integers(0, 2**128)))) for _ in range(n_rows)]
    cust_ids = [str(uuid.UUID(int=int(rng.integers(0, 2**128)))) for _ in range(n_rows)]
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2023-12-31")
    days = (end_date - start_date).days
    dates = [start_date + pd.Timedelta(days=int(d)) for d in rng.integers(0, days, n_rows)]
    payment_types = ["Credit Card", "Debit Card", "PayPal", "Wire Transfer"]
    payment = rng.choice(payment_types, size=n_rows, p=[0.4,0.3,0.15,0.15])
    amt_map = {"Credit Card":(250,150),"Debit Card":(180,100),"PayPal":(120,80),"Wire Transfer":(500,300)}
    amounts = np.array([max(10.0, rng.normal(amt_map[p][0], amt_map[p][1])) for p in payment]).round(2)
    scores = np.clip(rng.normal(650, 80, n_rows), 300, 850).astype(int)
    log_odds = -0.015 * (scores - 580) - 2.2
    probs = 1 / (1 + np.exp(-log_odds))
    defaulted = rng.binomial(1, probs).astype(int)
    df = pd.DataFrame({
        "TransactionID":tx_ids,"CustomerID":cust_ids,"Date":dates,"Amount":amounts,
        "PaymentType":payment,"CreditScore":scores,"Defaulted":defaulted
    }).sort_values("Date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic_transactions.csv")
    args = parser.parse_args()
    df = generate_transactions(args.rows, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df):,} rows -> {args.output}")
