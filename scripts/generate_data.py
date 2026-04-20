import argparse
import uuid

import numpy as np
import pandas as pd


def random_uuid(rng: np.random.Generator) -> str:
    return str(uuid.UUID(bytes=rng.bytes(16)))


def generate_transactions(n_rows: int = 50000, n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Create stable customer profiles so customer-level scoring has repeat behavior.
    customer_ids = [random_uuid(rng) for _ in range(n_customers)]
    base_scores = np.clip(rng.normal(655, 85, n_customers), 300, 850)
    income_band = rng.choice([1.0, 1.4, 1.9], size=n_customers, p=[0.45, 0.40, 0.15])

    sampled_customer_idx = rng.choice(np.arange(n_customers), size=n_rows, replace=True)
    cust_ids = [customer_ids[i] for i in sampled_customer_idx]
    cust_scores = base_scores[sampled_customer_idx]
    cust_income = income_band[sampled_customer_idx]

    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2023-12-31")
    total_days = (end_date - start_date).days + 1
    day_offsets = rng.integers(0, total_days, size=n_rows)
    dates = start_date + pd.to_timedelta(day_offsets, unit="D")
    months_from_start = ((dates.year - start_date.year) * 12 + (dates.month - start_date.month)).astype(float)

    payment_types = np.array(["Credit Card", "Debit Card", "PayPal", "Wire Transfer"])
    payment = rng.choice(payment_types, size=n_rows, p=[0.42, 0.30, 0.16, 0.12])
    payment_multiplier = np.where(payment == "Wire Transfer", 1.8, np.where(payment == "Credit Card", 1.2, 1.0))

    seasonal = 1.0 + 0.09 * np.sin((2 * np.pi * (dates.month - 1)) / 12.0)
    amount_base = rng.lognormal(mean=5.0, sigma=0.55, size=n_rows)
    amounts = np.clip(amount_base * payment_multiplier * cust_income * seasonal, 10, None).round(2)

    score_noise = rng.normal(0, 18, size=n_rows)
    scores = np.clip(cust_scores + score_noise, 300, 850).round().astype(int)

    # Encode risk signal with multiple drivers and mild macro stress over time.
    high_amount_flag = (amounts > np.quantile(amounts, 0.78)).astype(float)
    wire_flag = (payment == "Wire Transfer").astype(float)
    paypal_flag = (payment == "PayPal").astype(float)
    macro_stress = np.clip((months_from_start - 18) / 10.0, 0, 0.6)

    logits = (
        -4.1
        + 0.012 * (620 - scores)
        + 0.95 * high_amount_flag
        + 0.55 * wire_flag
        + 0.30 * paypal_flag
        + 0.20 * macro_stress
        + rng.normal(0, 0.35, size=n_rows)
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    defaulted = rng.binomial(1, probs).astype(int)

    tx_ids = [random_uuid(rng) for _ in range(n_rows)]
    df = pd.DataFrame(
        {
            "TransactionID": tx_ids,
            "CustomerID": cust_ids,
            "Date": dates,
            "Amount": amounts,
            "PaymentType": payment,
            "CreditScore": scores,
            "Defaulted": defaulted,
        }
    ).sort_values("Date").reset_index(drop=True)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--customers", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic_transactions.csv")
    args = parser.parse_args()
    df = generate_transactions(args.rows, args.customers, args.seed)
    df.to_csv(args.output, index=False)
    print(
        f"Generated {len(df):,} rows across {args.customers:,} customers -> {args.output} "
        f"(default rate={df['Defaulted'].mean():.2%})"
    )
