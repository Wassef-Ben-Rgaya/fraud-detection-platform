import json
import pickle
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "mlflow"
    / "best_model.pkl"
)

MODEL_INFO_PATH = (
    PROJECT_ROOT
    / "mlflow"
    / "model_info.json"
)

PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing_params.pkl"
)


# ============================================================
# TEST MODEL FILE
# ============================================================

def test_model_file_exists():

    assert MODEL_PATH.exists(), (
        f"Model not found: {MODEL_PATH}"
    )


# ============================================================
# TEST MODEL LOAD
# ============================================================

def test_model_load():

    model = joblib.load(
        MODEL_PATH
    )

    assert model is not None

    assert hasattr(
        model,
        "predict"
    )

    assert hasattr(
        model,
        "predict_proba"
    )


# ============================================================
# TEST MODEL INFO
# ============================================================

def test_model_info_exists():

    assert MODEL_INFO_PATH.exists(), (
        f"Model info not found: {MODEL_INFO_PATH}"
    )


def test_model_info():

    with open(
        MODEL_INFO_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        info = json.load(file)

    assert info["model"] == "RandomForest"

    assert info["version"] == "1.0.0"

    assert info["features"] == 32

    assert info["selection_metric"] == "f1_score"

    assert info["metrics"]["f1_score"] == 0.8404

    assert info["metrics"]["precision"] == 0.8778

    assert info["metrics"]["recall"] == 0.8061

    assert info["metrics"]["auc_roc"] == 0.9796


# ============================================================
# TEST PREPROCESSING FILE
# ============================================================

def test_preprocessing_file_exists():

    assert PREPROCESSING_PATH.exists(), (
        f"Preprocessing file not found: "
        f"{PREPROCESSING_PATH}"
    )


def test_preprocessing_parameters():

    with open(
        PREPROCESSING_PATH,
        "rb"
    ) as file:

        params = pickle.load(file)

    required_parameters = [
        "feature_names",
        "amount_mean",
        "amount_std",
        "time_mean",
        "time_std",
        "amount_zscore_mean",
        "amount_zscore_std"
    ]

    for parameter in required_parameters:

        assert parameter in params


    assert len(
        params["feature_names"]
    ) == 32


# ============================================================
# TEST MODEL FEATURE COUNT
# ============================================================

def test_model_feature_count():

    model = joblib.load(
        MODEL_PATH
    )

    assert model.n_features_in_ == 32


# ============================================================
# TEST MODEL PREDICTION
# ============================================================

def test_model_prediction():

    model = joblib.load(
        MODEL_PATH
    )

    features = np.array(
        [[
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
        ]]
    )

    assert features.shape == (1, 32)

    prediction = model.predict(
        features
    )

    probability = model.predict_proba(
        features
    )

    assert prediction.shape == (1,)

    assert prediction[0] in [0, 1]

    assert probability.shape == (1, 2)

    assert 0.0 <= probability[0][1] <= 1.0