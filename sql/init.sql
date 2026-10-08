-- Table principale des transactions
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    time FLOAT NOT NULL,
    v1 FLOAT, v2 FLOAT, v3 FLOAT, v4 FLOAT,
    v5 FLOAT, v6 FLOAT, v7 FLOAT, v8 FLOAT,
    v9 FLOAT, v10 FLOAT, v11 FLOAT, v12 FLOAT,
    v13 FLOAT, v14 FLOAT, v15 FLOAT, v16 FLOAT,
    v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT,
    v21 FLOAT, v22 FLOAT, v23 FLOAT, v24 FLOAT,
    v25 FLOAT, v26 FLOAT, v27 FLOAT, v28 FLOAT,
    amount FLOAT NOT NULL,
    class INTEGER NOT NULL,  -- 0: légitime, 1: fraude
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des prédictions (résultats du modèle)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES transactions(id),
    predicted_class INTEGER NOT NULL,
    fraud_probability FLOAT NOT NULL,
    model_version VARCHAR(50),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des métriques des modèles
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour performance
CREATE INDEX idx_transactions_class ON transactions(class);
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE INDEX idx_predictions_transaction ON predictions(transaction_id);