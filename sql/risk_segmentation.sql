SELECT DATE_TRUNC(''month'', "Date") AS month,
       COUNT(*) AS total_transactions,
       ROUND(SUM("Amount")::NUMERIC,2) AS total_revenue,
       ROUND(100.0 * SUM("Defaulted") / COUNT(*),2) AS default_rate_pct
FROM synthetic_transactions
GROUP BY 1
ORDER BY 1;

SELECT "PaymentType",
       COUNT(*) AS total_transactions,
       SUM("Defaulted") AS defaults,
       ROUND(100.0 * SUM("Defaulted") / COUNT(*),2) AS default_rate_pct
FROM synthetic_transactions
GROUP BY "PaymentType"
ORDER BY default_rate_pct DESC;
