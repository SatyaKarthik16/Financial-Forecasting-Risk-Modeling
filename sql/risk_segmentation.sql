-- 1) Monthly revenue and default summary
SELECT
    DATE_TRUNC('month', "Date") AS month,
    COUNT(*) AS total_transactions,
    ROUND(SUM("Amount")::NUMERIC, 2) AS total_revenue,
    ROUND(AVG("Amount")::NUMERIC, 2) AS avg_transaction_amount,
    SUM("Defaulted") AS total_defaults,
    ROUND(100.0 * SUM("Defaulted") / COUNT(*), 2) AS default_rate_pct
FROM synthetic_transactions
GROUP BY 1
ORDER BY 1;

-- 2) Customer risk-tier segmentation by average credit score
SELECT
    "CustomerID",
    AVG("CreditScore") AS avg_credit_score,
    COUNT(*) AS transaction_count,
    ROUND(SUM("Amount")::NUMERIC, 2) AS total_spend,
    MAX("Defaulted") AS ever_defaulted,
    CASE
        WHEN AVG("CreditScore") >= 740 THEN 'Excellent (740+)'
        WHEN AVG("CreditScore") >= 670 THEN 'Good (670-739)'
        WHEN AVG("CreditScore") >= 580 THEN 'Fair (580-669)'
        ELSE 'Poor (<580)'
    END AS risk_tier
FROM synthetic_transactions
GROUP BY "CustomerID"
ORDER BY avg_credit_score DESC;

-- 3) Default rate by payment type
SELECT
    "PaymentType",
    COUNT(*) AS total_transactions,
    SUM("Defaulted") AS defaults,
    ROUND(100.0 * SUM("Defaulted") / COUNT(*), 2) AS default_rate_pct,
    ROUND(AVG("Amount")::NUMERIC, 2) AS avg_amount
FROM synthetic_transactions
GROUP BY "PaymentType"
ORDER BY default_rate_pct DESC;

-- 4) High-risk customer watchlist
SELECT
    "CustomerID",
    COUNT(*) AS transaction_count,
    ROUND(SUM("Amount")::NUMERIC, 2) AS total_exposure,
    AVG("CreditScore") AS avg_credit_score,
    SUM("Defaulted") AS default_events,
    ROUND(100.0 * SUM("Defaulted") / COUNT(*), 2) AS personal_default_rate_pct
FROM synthetic_transactions
GROUP BY "CustomerID"
HAVING AVG("CreditScore") < 580 AND SUM("Defaulted") >= 1
ORDER BY total_exposure DESC
LIMIT 100;

-- 5) Expected credit loss by risk tier (LGD=45%)
WITH risk_tiers AS (
    SELECT
        CASE
            WHEN "CreditScore" >= 740 THEN 'Excellent (740+)'
            WHEN "CreditScore" >= 670 THEN 'Good (670-739)'
            WHEN "CreditScore" >= 580 THEN 'Fair (580-669)'
            ELSE 'Poor (<580)'
        END AS risk_tier,
        "Amount",
        "Defaulted"
    FROM synthetic_transactions
)
SELECT
    risk_tier,
    COUNT(*) AS transactions,
    ROUND(SUM("Amount")::NUMERIC, 2) AS total_exposure,
    ROUND(100.0 * SUM("Defaulted") / COUNT(*), 2) AS pd_pct,
    ROUND(SUM("Amount") * (SUM("Defaulted")::FLOAT / COUNT(*)) * 0.45, 2) AS expected_credit_loss
FROM risk_tiers
GROUP BY risk_tier
ORDER BY pd_pct DESC;

-- 6) Rolling 3-month default rate
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', "Date") AS month,
        COUNT(*) AS txn_count,
        SUM("Defaulted") AS defaults
    FROM synthetic_transactions
    GROUP BY 1
)
SELECT
    month,
    txn_count,
    defaults,
    ROUND(100.0 * defaults / txn_count, 2) AS monthly_default_rate_pct,
    ROUND(
        100.0 * SUM(defaults) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
              / SUM(txn_count) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
        2
    ) AS rolling_3m_default_rate_pct
FROM monthly
ORDER BY month;
