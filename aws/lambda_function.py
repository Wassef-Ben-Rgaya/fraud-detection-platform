import json
import boto3
import joblib
import numpy as np
import os


# ============================================================
# AWS S3 CLIENT
# ============================================================

s3 = boto3.client("s3")


# ============================================================
# GLOBAL MODEL
# ============================================================

model = None


# ============================================================
# LOAD MODEL FROM S3
# ============================================================

def load_model():
    global model

    if model is None:

        bucket = os.environ["S3_BUCKET_NAME"]

        model_key = os.getenv(
            "S3_MODEL_KEY",
            "models/best_model.pkl"
        )

        tmp_path = "/tmp/best_model.pkl"

        print(
            f"☁️ Téléchargement : "
            f"s3://{bucket}/{model_key}"
        )

        s3.download_file(
            bucket,
            model_key,
            tmp_path
        )

        print(
            f"📦 Modèle téléchargé : "
            f"{tmp_path}"
        )

        print("🔄 Chargement du modèle...")

        model = joblib.load(tmp_path)

        print(
            f"✅ Modèle chargé : "
            f"{type(model).__name__}"
        )

    return model


# ============================================================
# LAMBDA HANDLER
# ============================================================

def lambda_handler(event, context):

    try:

        print("📥 Event reçu :")
        print(json.dumps(event))

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

        body = event.get(
            "body",
            "{}"
        )

        if isinstance(body, str):
            body = json.loads(body)

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        if "features" not in body:

            raise ValueError(
                "Le champ 'features' est obligatoire."
            )

        features = body["features"]

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if len(features) != 32:

            raise ValueError(
                f"32 features attendues, "
                f"{len(features)} reçues."
            )

        # ----------------------------------------------------
        # NUMPY
        # ----------------------------------------------------

        features_array = np.array(
            features,
            dtype=float
        ).reshape(1, -1)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        m = load_model()

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = int(
            m.predict(features_array)[0]
        )

        # ----------------------------------------------------
        # FRAUD PROBABILITY
        # ----------------------------------------------------

        probability = float(
            m.predict_proba(
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
        # MODEL INFORMATION
        # ----------------------------------------------------

        model_name = "RandomForest"
        model_version = "1.0.0"

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "prediction": prediction,

            "label": (
                "FRAUD"
                if prediction == 1
                else "LEGITIMATE"
            ),

            "fraud_probability": round(
                probability,
                4
            ),

            "risk_level": risk_level,

            "model": model_name,

            "version": model_version
        }

        print("✅ Résultat :")
        print(json.dumps(response))

        # ----------------------------------------------------
        # RESPONSE API GATEWAY
        # ----------------------------------------------------

        return {

            "statusCode": 200,

            "headers": {
                "Content-Type": "application/json"
            },

            "body": json.dumps(response)
        }

    # ========================================================
    # ERROR
    # ========================================================

    except Exception as e:

        print(
            f"❌ Erreur Lambda : {str(e)}"
        )

        return {

            "statusCode": 400,

            "headers": {
                "Content-Type": "application/json"
            },

            "body": json.dumps({
                "error": str(e)
            })
        }