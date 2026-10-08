import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"D:\fraud-detection-platform"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "data_splits.pkl"
)

PREPROCESSING_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "preprocessing_params.pkl"
)

MODELS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "models"
)

MLFLOW_DIR = (
    BASE_DIR
    / "mlflow"
)

BEST_MODEL_PATH = (
    MLFLOW_DIR
    / "best_model.pkl"
)

MODEL_INFO_PATH = (
    MLFLOW_DIR
    / "model_info.json"
)

MLFLOW_TRACKING_URI = (
    "http://localhost:5001"
)

MLFLOW_EXPERIMENT = (
    "fraud-detection"
)

RANDOM_STATE = 42


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MLFLOW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

print("\n" + "=" * 70)
print("📂 CHARGEMENT DES DONNÉES")
print("=" * 70)

print(f"\n{DATA_PATH}")


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"❌ Fichier introuvable : {DATA_PATH}"
    )


data = joblib.load(
    DATA_PATH
)


# ============================================================
# CHARGEMENT DES PARAMÈTRES DE PREPROCESSING
# ============================================================

print("\n📐 Chargement des paramètres de preprocessing...")

if not PREPROCESSING_PATH.exists():

    raise FileNotFoundError(
        f"❌ Paramètres de preprocessing introuvables : "
        f"{PREPROCESSING_PATH}"
    )


preprocessing_params = joblib.load(
    PREPROCESSING_PATH
)


print(
    "✅ Paramètres de preprocessing chargés."
)


# ============================================================
# AFFICHAGE PARAMÈTRES
# ============================================================

print("\n📐 PARAMÈTRES")

print(
    f"   Amount mean : "
    f"{preprocessing_params['amount_mean']:.6f}"
)

print(
    f"   Amount std  : "
    f"{preprocessing_params['amount_std']:.6f}"
)

print(
    f"   Time mean   : "
    f"{preprocessing_params['time_mean']:.6f}"
)

print(
    f"   Time std    : "
    f"{preprocessing_params['time_std']:.6f}"
)

print(
    f"   Amount Z mean : "
    f"{preprocessing_params['amount_zscore_mean']:.6f}"
)

print(
    f"   Amount Z std  : "
    f"{preprocessing_params['amount_zscore_std']:.6f}"
)


# ============================================================
# RÉCUPÉRATION DES DONNÉES
# ============================================================

if isinstance(data, dict):

    X_train = np.asarray(
        data["X_train"]
    )

    X_test = np.asarray(
        data["X_test"]
    )

    y_train = np.asarray(
        data["y_train"]
    )

    y_test = np.asarray(
        data["y_test"]
    )

    metadata = data.get(
        "metadata",
        {}
    )

else:

    raise ValueError(
        "❌ data_splits.pkl ne contient pas "
        "le format attendu."
    )


print("\n✅ Données chargées.")

print(
    f"   X_train : "
    f"{X_train.shape}"
)

print(
    f"   X_test  : "
    f"{X_test.shape}"
)

print(
    f"   y_train : "
    f"{y_train.shape}"
)

print(
    f"   y_test  : "
    f"{y_test.shape}"
)


# ============================================================
# FEATURES
# ============================================================

N_FEATURES = X_train.shape[1]

print(
    f"\n🔢 Nombre de features : "
    f"{N_FEATURES}"
)


if N_FEATURES != 32:

    raise ValueError(
        f"❌ Le modèle doit utiliser 32 features. "
        f"{N_FEATURES} trouvées."
    )


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = preprocessing_params.get(
    "feature_names",
    []
)


print("\n📋 Ordre des features :")

for index, feature in enumerate(
    FEATURE_NAMES,
    start=1
):

    print(
        f"   {index:02d}. {feature}"
    )


# ============================================================
# DATASET ORIGINAL
# ============================================================

original_dataset = metadata.get(
    "original_dataset",
    {}
)

original_total = original_dataset.get(
    "total_transactions",
    0
)

original_fraud = original_dataset.get(
    "fraud_transactions",
    0
)

original_legitimate = original_dataset.get(
    "legitimate_transactions",
    0
)

original_fraud_rate = original_dataset.get(
    "fraud_rate",
    "0.000%"
)


# ============================================================
# AFFICHAGE DATASET
# ============================================================

print("\n" + "=" * 70)
print("📊 DATASET ORIGINAL")
print("=" * 70)

print(
    f"\n   Total transactions : "
    f"{original_total:,}"
)

print(
    f"   Fraudes : "
    f"{original_fraud:,}"
)

print(
    f"   Légitimes : "
    f"{original_legitimate:,}"
)

print(
    f"   Taux de fraude : "
    f"{original_fraud_rate}"
)


# ============================================================
# MLFLOW
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)


# ============================================================
# MODÈLES
# ============================================================

models = {

    "LogisticRegression":

        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "RandomForest":

        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "XGBoost":

        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
}


# ============================================================
# TYPES SKOPS AUTORISÉS
# ============================================================

TRUSTED_TYPES = [

    "sklearn.tree._tree.Tree",

    "xgboost.core.Booster",

    "xgboost.sklearn.XGBClassifier"
]


# ============================================================
# ENTRAÎNEMENT
# ============================================================

results = {}

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 60)

    print(
        f"🚀 Entraînement : "
        f"{model_name}"
    )

    print("=" * 60)

    print(
        "\n⏳ Entraînement en cours..."
    )


    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        print(
            "✅ Entraînement terminé."
        )


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        y_proba = (
            model.predict_proba(
                X_test
            )[:, 1]
        )


        # ----------------------------------------------------
        # MÉTRIQUES
        # ----------------------------------------------------

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            y_proba
        )


        # ----------------------------------------------------
        # AFFICHAGE MÉTRIQUES
        # ----------------------------------------------------

        print("\n📊 MÉTRIQUES")

        print(
            f"   F1        : "
            f"{f1:.4f}"
        )

        print(
            f"   Precision : "
            f"{precision:.4f}"
        )

        print(
            f"   Recall    : "
            f"{recall:.4f}"
        )

        print(
            f"   AUC-ROC   : "
            f"{auc:.4f}"
        )


        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        print(
            "\n📋 Classification Report\n"
        )

        print(
            classification_report(
                y_test,
                y_pred,
                target_names=[
                    "Légitime",
                    "Fraude"
                ],
                zero_division=0
            )
        )


        # ----------------------------------------------------
        # MLFLOW PARAMETERS
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            model_name
        )

        mlflow.log_param(
            "features",
            N_FEATURES
        )

        mlflow.log_param(
            "original_dataset_size",
            original_total
        )

        mlflow.log_param(
            "original_fraud_count",
            original_fraud
        )

        mlflow.log_param(
            "train_size_after_smote",
            len(y_train)
        )

        mlflow.log_param(
            "preprocessing_file",
            str(PREPROCESSING_PATH)
        )


        # ----------------------------------------------------
        # MLFLOW METRICS
        # ----------------------------------------------------

        mlflow.log_metric(
            "f1_score",
            float(f1)
        )

        mlflow.log_metric(
            "precision",
            float(precision)
        )

        mlflow.log_metric(
            "recall",
            float(recall)
        )

        mlflow.log_metric(
            "auc_roc",
            float(auc)
        )


        # ----------------------------------------------------
        # LOG MODEL MLFLOW
        # ----------------------------------------------------

        try:

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=TRUSTED_TYPES
            )

        except Exception as e:

            print(
                "\n⚠️ Impossible de logger "
                "le modèle dans MLflow :"
            )

            print(e)


        # ----------------------------------------------------
        # SAVE LOCAL MODEL
        # ----------------------------------------------------

        model_path = (
            MODELS_DIR
            / f"{model_name.lower()}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"\n💾 {model_name} : "
            f"{model_path}"
        )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        results[model_name] = {

            "f1_score":
                round(
                    float(f1),
                    4
                ),

            "precision":
                round(
                    float(precision),
                    4
                ),

            "recall":
                round(
                    float(recall),
                    4
                ),

            "auc_roc":
                round(
                    float(auc),
                    4
                )
        }


        trained_models[
            model_name
        ] = model


# ============================================================
# COMPARAISON
# ============================================================

print("\n\n" + "=" * 70)
print("📊 COMPARAISON DES MODÈLES")
print("=" * 70)


for model_name, metrics in results.items():

    print(
        f"\n🔹 {model_name}"
    )

    print(
        f"   F1        : "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"   Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"   Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"   AUC-ROC   : "
        f"{metrics['auc_roc']:.4f}"
    )


# ============================================================
# MEILLEUR MODÈLE
# ============================================================

best_model_name = max(
    results,
    key=lambda name:
        results[name]["f1_score"]
)

best_model = trained_models[
    best_model_name
]

best_metrics = results[
    best_model_name
]


# ============================================================
# AFFICHAGE BEST MODEL
# ============================================================

print("\n\n" + "=" * 70)
print("🏆 MODÈLE RETENU POUR LE DÉPLOIEMENT")
print("=" * 70)

print(
    f"\nModèle : "
    f"{best_model_name}"
)

print(
    f"F1 : "
    f"{best_metrics['f1_score']:.4f}"
)

print(
    f"Precision : "
    f"{best_metrics['precision']:.4f}"
)

print(
    f"Recall : "
    f"{best_metrics['recall']:.4f}"
)

print(
    f"AUC-ROC : "
    f"{best_metrics['auc_roc']:.4f}"
)


# ============================================================
# SAUVEGARDE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    BEST_MODEL_PATH
)

print(
    "\n💾 Meilleur modèle sauvegardé dans :"
)

print(
    BEST_MODEL_PATH
)


# ============================================================
# MODEL INFO
# ============================================================

model_info = {

    "model":
        best_model_name,

    "version":
        "1.0.0",

    "selection_metric":
        "f1_score",

    "features":
        N_FEATURES,

    "feature_names":
        FEATURE_NAMES,

    "metrics":
        best_metrics,

    "dataset": {

        "total_transactions":
            original_total,

        "fraud_transactions":
            original_fraud,

        "legitimate_transactions":
            original_legitimate,

        "fraud_rate":
            original_fraud_rate
    },

    "training": {

        "train_transactions_before_smote":
            metadata
            .get("split", {})
            .get(
                "train_transactions",
                0
            ),

        "test_transactions":
            metadata
            .get("split", {})
            .get(
                "test_transactions",
                0
            ),

        "train_transactions_after_smote":
            len(y_train),

        "fraud_transactions_after_smote":
            int(y_train.sum())
    },

    "preprocessing": {

        "amount_mean":
            preprocessing_params[
                "amount_mean"
            ],

        "amount_std":
            preprocessing_params[
                "amount_std"
            ],

        "time_mean":
            preprocessing_params[
                "time_mean"
            ],

        "time_std":
            preprocessing_params[
                "time_std"
            ],

        "amount_zscore_mean":
            preprocessing_params[
                "amount_zscore_mean"
            ],

        "amount_zscore_std":
            preprocessing_params[
                "amount_zscore_std"
            ]
    }
}


# ============================================================
# SAVE MODEL INFO
# ============================================================

with open(
    MODEL_INFO_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_info,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    "\n💾 Informations du modèle sauvegardées dans :"
)

print(
    MODEL_INFO_PATH
)


# ============================================================
# FIN
# ============================================================

print("\n\n" + "=" * 70)
print("📦 RÉSUMÉ")
print("=" * 70)

print(
    f"\n🤖 Modèle actif : "
    f"{best_model_name}"
)

print(
    f"📈 F1 : "
    f"{best_metrics['f1_score']:.4f}"
)

print(
    f"📊 Dataset original : "
    f"{original_total:,}"
)

print(
    f"🚨 Fraudes originales : "
    f"{original_fraud:,}"
)

print(
    f"📉 Taux de fraude : "
    f"{original_fraud_rate}"
)

print(
    f"\n📁 Best model : "
    f"{BEST_MODEL_PATH}"
)

print(
    f"📁 Model info : "
    f"{MODEL_INFO_PATH}"
)

print(
    f"📁 Preprocessing : "
    f"{PREPROCESSING_PATH}"
)

print("\n" + "=" * 70)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("=" * 70)