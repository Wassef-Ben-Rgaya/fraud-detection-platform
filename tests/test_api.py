import requests


API_URL = "http://localhost:5000/api"


# ============================================================
# TEST API HEALTH
# ============================================================

def test_api_health():
    response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "RandomForest"
    assert data["version"] == "1.0.0"


# ============================================================
# TEST MODEL INFORMATION
# ============================================================

def test_model_info():
    response = requests.get(
        f"{API_URL}/model",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "RandomForest"
    assert data["version"] == "1.0.0"
    assert data["features"] == 32

    assert data["metrics"]["f1_score"] == 0.8404
    assert data["metrics"]["precision"] == 0.8778
    assert data["metrics"]["recall"] == 0.8061
    assert data["metrics"]["auc_roc"] == 0.9796


# ============================================================
# TEST DATABASE HEALTH
# ============================================================

def test_database_health():
    response = requests.get(
        f"{API_URL}/database/health",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ============================================================
# TEST PREDICTION
# ============================================================

def test_prediction():

    features = [
        -2.3122265423263,
        1.95199201064158,
        -1.60985073229769,
        3.9979055875468,
        -0.522187864667764,
        -1.42654531920595,
        -2.53738730624579,
        1.39165724829804,
        -2.77008927719433,
        -2.77227214465915,
        3.20203320709635,
        -2.89990738849473,
        -0.595221881324605,
        -4.28925378244217,
        0.389724120274487,
        -1.14074717980657,
        -2.83005567450437,
        -0.0168224681808257,
        0.416955705037907,
        0.126910559061474,
        0.517232370861764,
        -0.0350493686052974,
        -0.465211076182388,
        0.320198198514526,
        0.0445191674731724,
        0.177839798284401,
        0.261145002567677,
        -0.143275874698919,
        -0.35322939296682354,
        -1.9880335064229064,
        0.0,
        -0.35322877284564846
    ]

    assert len(features) == 32

    response = requests.post(
        f"{API_URL}/predict",
        json={
            "features": features
        },
        timeout=30
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "label" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "model" in data
    assert "version" in data
    assert "prediction_id" in data

    assert data["model"] == "RandomForest"
    assert data["version"] == "1.0.0"

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["fraud_probability"] <= 1.0

    assert data["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    assert data["prediction_id"] is not None


# ============================================================
# TEST PREDICTIONS HISTORY
# ============================================================

def test_predictions_history():

    response = requests.get(
        f"{API_URL}/predictions",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data

    assert isinstance(
        data["predictions"],
        list
    )

    assert len(data["predictions"]) >= 1