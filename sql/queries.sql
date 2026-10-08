-- ============================================
-- FRAUD DETECTION — Advanced SQL Queries
-- ============================================

-- 1. Distribution des fraudes par tranche de montant
SELECT
    CASE
        WHEN amount < 10    THEN '< 10'
        WHEN amount < 100   THEN '10 - 100'
        WHEN amount < 1000  THEN '100 - 1000'
        ELSE '> 1000'
    END AS amount_range,
    COUNT(*) AS total,
    SUM(class) AS frauds,
    ROUND(AVG(class) * 100, 2) AS fraud_rate_pct
FROM transactions
GROUP BY amount_range
ORDER BY fraud_rate_pct DESC;

-- 2. Transactions suspectes (montant élevé + haute probabilité)
SELECT
    id, amount, class,
    RANK() OVER (ORDER BY amount DESC) AS amount_rank
FROM transactions
WHERE class = 1
ORDER BY amount DESC
LIMIT 20;

-- 3. Taux de fraude par heure de la journée
SELECT
    FLOOR(MOD(time, 86400) / 3600) AS hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(class) AS fraud_count,
    ROUND(AVG(class) * 100, 4) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- 4. Statistiques globales avec Window Functions
SELECT
    id,
    amount,
    class,
    AVG(amount) OVER () AS global_avg_amount,
    AVG(amount) OVER (PARTITION BY class) AS class_avg_amount,
    amount - AVG(amount) OVER (PARTITION BY class) AS deviation_from_class_avg,
    NTILE(4) OVER (ORDER BY amount) AS amount_quartile
FROM transactions
LIMIT 100;

-- 5. CTE — Top transactions frauduleuses par quartile
WITH quartiles AS (
    SELECT
        id, amount, class,
        NTILE(4) OVER (ORDER BY amount) AS quartile
    FROM transactions
),
fraud_by_quartile AS (
    SELECT
        quartile,
        COUNT(*) AS total,
        SUM(class) AS frauds,
        ROUND(AVG(class) * 100, 2) AS fraud_rate
    FROM quartiles
    GROUP BY quartile
)
SELECT * FROM fraud_by_quartile ORDER BY quartile;

-- 6. Détection d'anomalies — montants inhabituels
WITH stats AS (
    SELECT
        AVG(amount) AS mean_amount,
        STDDEV(amount) AS std_amount
    FROM transactions
)
SELECT
    t.id, t.amount, t.class,
    ROUND((t.amount - s.mean_amount) / s.std_amount, 2) AS z_score
FROM transactions t, stats s
WHERE ABS((t.amount - s.mean_amount) / s.std_amount) > 3
ORDER BY z_score DESC
LIMIT 50;