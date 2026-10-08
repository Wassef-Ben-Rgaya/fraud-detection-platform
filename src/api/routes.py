from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import os
import json
from datetime import datetime

import psycopg2

from src.api.schemas import validate_features, validate_batch_features


# ============================================================
# API BLUEPRINT
# ============================================================

api_bp = Blueprint(
    "api",
    __name__
)


# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Racine du projet
# D:\fraud-detection-platform

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../.."
    )
)


# ============================================================
# CHEMINS DU MODÈLE
# ============================================================

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow",
    "best_model.pkl"
)

DEFAULT_MODEL_INFO_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow",
    "model_info.json"
)


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    DEFAULT_MODEL_PATH
)

MODEL_INFO_PATH = os.getenv(
    "MODEL_INFO_PATH",
    DEFAULT_MODEL_INFO_PATH
)


# ============================================================
# CONFIGURATION POSTGRESQL
# ============================================================

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost"
)

POSTGRES_PORT = os.getenv(
    "POSTGRES_PORT",
    "5432"
)

POSTGRES_DB = os.getenv(
    "POSTGRES_DB",
    "fraud_db"
)

POSTGRES_USER = os.getenv(
    "POSTGRES_USER",
    "fraud_user"
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD",
    "admin"
)


def get_db_connection():
    """
    Create a PostgreSQL connection.
    """

    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )


# ============================================================
# SAUVEGARDE D'UNE PRÉDICTION
# ============================================================

def save_prediction(
    transaction_id,
    predicted_class,
    fraud_probability,
    model_version
):
    """
    Save a prediction in PostgreSQL.
    """

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO predictions (
                transaction_id,
                predicted_class,
                fraud_probability,
                model_version
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id, predicted_at;
        """

        cursor.execute(
            query,
            (
                transaction_id,
                predicted_class,
                fraud_probability,
                model_version
            )
        )

        prediction_id, predicted_at = cursor.fetchone()

        connection.commit()

        return {
            "id": prediction_id,
            "predicted_at": predicted_at.isoformat()
        }

    except Exception:

        if connection:
            connection.rollback()

        raise

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# LOGS DE DÉMARRAGE
# ============================================================

print("=" * 70)
print("FRAUD DETECTION API")
print("=" * 70)

print(
    f"BASE_DIR: {BASE_DIR}"
)

print(
    f"PROJECT_ROOT: {PROJECT_ROOT}"
)

print(
    f"MODEL_PATH: {MODEL_PATH}"
)

print(
    f"MODEL_INFO_PATH: {MODEL_INFO_PATH}"
)

print(
    f"POSTGRES_HOST: {POSTGRES_HOST}"
)

print(
    f"POSTGRES_PORT: {POSTGRES_PORT}"
)

print(
    f"POSTGRES_DB: {POSTGRES_DB}"
)

print(
    f"POSTGRES_USER: {POSTGRES_USER}"
)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        "Modèle introuvable.\n"
        f"Chemin recherché : {MODEL_PATH}"
    )


model = joblib.load(
    MODEL_PATH
)


print(
    f"Model loaded successfully: "
    f"{type(model).__name__}"
)


# ============================================================
# CHARGEMENT DE model_info.json
# ============================================================

if not os.path.isfile(MODEL_INFO_PATH):

    raise FileNotFoundError(
        "Fichier model_info.json introuvable.\n"
        f"Chemin recherché : {MODEL_INFO_PATH}"
    )


with open(
    MODEL_INFO_PATH,
    "r",
    encoding="utf-8"
) as f:

    model_info = json.load(f)


print(
    "model_info.json loaded successfully."
)


# ============================================================
# INFORMATIONS DU MODÈLE
# ============================================================

MODEL_NAME = model_info.get(
    "model",
    type(model).__name__
)

MODEL_VERSION = model_info.get(
    "version",
    "1.0.0"
)

MODEL_FEATURES = model_info.get(
    "features",
    32
)

MODEL_METRICS = model_info.get(
    "metrics",
    {}
)

DATASET_INFO = model_info.get(
    "dataset",
    {}
)

SELECTION_METRIC = model_info.get(
    "selection_metric",
    "f1_score"
)


# ============================================================
# VÉRIFICATION DES MÉTRIQUES
# ============================================================

print(
    f"Model name: {MODEL_NAME}"
)

print(
    f"Model version: {MODEL_VERSION}"
)

print(
    f"Model features: {MODEL_FEATURES}"
)

print(
    f"Selection metric: {SELECTION_METRIC}"
)

print(
    f"Model metrics: {MODEL_METRICS}"
)

print(
    f"Dataset info: {DATASET_INFO}"
)


REQUIRED_METRICS = [
    "f1_score",
    "precision",
    "recall",
    "auc_roc"
]


missing_metrics = [
    metric
    for metric in REQUIRED_METRICS
    if metric not in MODEL_METRICS
]


if missing_metrics:

    print(
        "WARNING: Missing metrics: "
        + ", ".join(missing_metrics)
    )

else:

    print(
        "All required metrics loaded successfully."
    )


print("=" * 70)


# ============================================================
# HEALTH
# ============================================================

@api_bp.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "healthy",

        "timestamp":
            datetime.utcnow().isoformat(),

        "model":
            MODEL_NAME,

        "version":
            MODEL_VERSION
    })


# ============================================================
# PREDICT
# ============================================================

@api_bp.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        # ----------------------------------------------------
        # Vérification JSON
        # ----------------------------------------------------

        if not data:

            return jsonify({

                "error":
                    "JSON body manquant."

            }), 400

        # ----------------------------------------------------
        # Vérification features
        # ----------------------------------------------------

        features = data.get(
            "features"
        )

        valid, error = validate_features(
            features
        )

        if not valid:

            return jsonify({
                "error": error
            }), 400

        # ----------------------------------------------------
        # Transaction ID optionnel
        # ----------------------------------------------------

        transaction_id = data.get(
            "transaction_id"
        )

        if transaction_id is not None:

            try:

                transaction_id = int(
                    transaction_id
                )

            except (TypeError, ValueError):

                return jsonify({
                    "error":
                        "transaction_id doit être un entier."
                }), 400

        # ----------------------------------------------------
        # Conversion NumPy
        # ----------------------------------------------------

        features_array = np.array(
            features,
            dtype=float
        ).reshape(
            1,
            -1
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = int(
            model.predict(
                features_array
            )[0]
        )

        probability = float(
            model.predict_proba(
                features_array
            )[0][1]
        )

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability > 0.7:

            risk_level = "HIGH"

        elif probability > 0.3:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        # ----------------------------------------------------
        # SAUVEGARDE POSTGRESQL
        # ----------------------------------------------------

        saved_prediction = save_prediction(

            transaction_id=transaction_id,

            predicted_class=prediction,

            fraud_probability=probability,

            model_version=MODEL_VERSION
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "prediction_id":
                saved_prediction["id"],

            "transaction_id":
                transaction_id,

            "prediction":
                prediction,

            "label": (
                "FRAUD"
                if prediction == 1
                else "LEGITIMATE"
            ),

            "fraud_probability":
                round(
                    probability,
                    4
                ),

            "risk_level":
                risk_level,

            "model":
                MODEL_NAME,

            "version":
                MODEL_VERSION,

            "predicted_at":
                saved_prediction["predicted_at"],

            "timestamp":
                datetime.utcnow().isoformat()
        })

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# PREDICT BATCH
# ============================================================

@api_bp.route(
    "/predict/batch",
    methods=["POST"]
)
def predict_batch():

    try:

        data = request.get_json()

        # ----------------------------------------------------
        # Vérification JSON
        # ----------------------------------------------------

        if not data:

            return jsonify({

                "error":
                    "JSON body manquant."

            }), 400

        # ----------------------------------------------------
        # Vérification features
        # ----------------------------------------------------

        features = data.get(
            "features"
        )

        valid, error = validate_batch_features(
            features
        )

        if not valid:

            return jsonify({
                "error": error
            }), 400

        # ----------------------------------------------------
        # Conversion NumPy
        # ----------------------------------------------------

        features_array = np.array(
            features,
            dtype=float
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = model.predict(
            features_array
        ).tolist()

        probabilities = model.predict_proba(
            features_array
        )[:, 1].tolist()

        # ----------------------------------------------------
        # CONSTRUCTION DES RÉSULTATS
        # ----------------------------------------------------

        results = []

        for i, (
            prediction,
            probability
        ) in enumerate(
            zip(
                predictions,
                probabilities
            )
        ):

            if probability > 0.7:

                risk_level = "HIGH"

            elif probability > 0.3:

                risk_level = "MEDIUM"

            else:

                risk_level = "LOW"

            results.append({

                "index":
                    i,

                "prediction":
                    int(prediction),

                "label": (
                    "FRAUD"
                    if prediction == 1
                    else "LEGITIMATE"
                ),

                "fraud_probability":
                    round(
                        float(probability),
                        4
                    ),

                "risk_level":
                    risk_level
            })

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "total":
                len(results),

            "fraud_count":
                sum(
                    r["prediction"]
                    for r in results
                ),

            "model":
                MODEL_NAME,

            "version":
                MODEL_VERSION,

            "results":
                results
        })

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# METRICS
# ============================================================

@api_bp.route(
    "/metrics",
    methods=["GET"]
)
def metrics():

    return jsonify({

        "model":
            MODEL_NAME,

        "version":
            MODEL_VERSION,

        "selection_metric":
            SELECTION_METRIC,

        "metrics":
            MODEL_METRICS,

        "dataset":
            DATASET_INFO
    })


# ============================================================
# MODEL INFORMATION
# ============================================================

@api_bp.route(
    "/model",
    methods=["GET"]
)
def model_information():

    return jsonify({

        "model":
            MODEL_NAME,

        "version":
            MODEL_VERSION,

        "features":
            MODEL_FEATURES,

        "selection_metric":
            SELECTION_METRIC,

        "metrics":
            MODEL_METRICS,

        "dataset":
            DATASET_INFO
    })


# ============================================================
# PREDICTION HISTORY
# ============================================================

@api_bp.route(
    "/predictions",
    methods=["GET"]
)
def prediction_history():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        query = """
            SELECT
                id,
                transaction_id,
                predicted_class,
                fraud_probability,
                model_version,
                predicted_at
            FROM predictions
            ORDER BY predicted_at DESC
            LIMIT 100;
        """

        cursor.execute(
            query
        )

        rows = cursor.fetchall()

        predictions = []

        for row in rows:

            predictions.append({

                "id":
                    row[0],

                "transaction_id":
                    row[1],

                "predicted_class":
                    row[2],

                "label": (
                    "FRAUD"
                    if row[2] == 1
                    else "LEGITIMATE"
                ),

                "fraud_probability":
                    round(
                        float(row[3]),
                        4
                    ),

                "model_version":
                    row[4],

                "predicted_at":
                    row[5].isoformat()
            })

        return jsonify({

            "total":
                len(predictions),

            "predictions":
                predictions
        })

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500

    finally:

        if cursor:

            cursor.close()

        if connection:

            connection.close()


# ============================================================
# DATABASE HEALTH
# ============================================================

@api_bp.route(
    "/database/health",
    methods=["GET"]
)
def database_health():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            "SELECT 1;"
        )

        result = cursor.fetchone()

        return jsonify({

            "status":
                "healthy",

            "database":
                POSTGRES_DB,

            "host":
                POSTGRES_HOST,

            "result":
                result[0]
        })

    except Exception as e:

        return jsonify({

            "status":
                "unhealthy",

            "error":
                str(e)

        }), 500

    finally:

        if cursor:

            cursor.close()

        if connection:

            connection.close()