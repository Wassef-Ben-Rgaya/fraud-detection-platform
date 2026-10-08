-- Vue analytique principale

CREATE OR REPLACE VIEW fraud_summary AS

SELECT
    class,
    COUNT(*) AS total_transactions,
    ROUND(AVG(amount)::numeric, 2) AS avg_amount,
    ROUND(MIN(amount)::numeric, 2) AS min_amount,
    ROUND(MAX(amount)::numeric, 2) AS max_amount,
    ROUND(STDDEV(amount)::numeric, 2) AS std_amount
FROM transactions
GROUP BY class;


-- Vue des transactions à haut risque

CREATE OR REPLACE VIEW high_risk_transactions AS

SELECT
    id,
    amount,
    time,
    class
FROM transactions
WHERE amount > 1000 OR class = 1
ORDER BY amount DESC;