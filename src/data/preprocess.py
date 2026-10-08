import os

import joblib
import pandas as pd

from dotenv import load_dotenv

from sqlalchemy import create_engine

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"D:\fraud-detection-platform"

PROCESSED_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

SPLIT_PATH = os.path.join(
    PROCESSED_PATH,
    "data_splits.pkl"
)

PREPROCESSING_PATH = os.path.join(
    PROCESSED_PATH,
    "preprocessing_params.pkl"
)

os.makedirs(
    PROCESSED_PATH,
    exist_ok=True
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    os.path.join(
        PROJECT_ROOT,
        ".env"
    )
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    engine = create_engine(
        f"postgresql://"
        f"{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:"
        f"{os.getenv('POSTGRES_PORT')}/"
        f"{os.getenv('POSTGRES_DB')}"
    )

    return pd.read_sql(
        "SELECT * FROM transactions",
        engine
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    print("🔧 Feature engineering...")

    # --------------------------------------------------------
    # AMOUNT SCALER
    # --------------------------------------------------------

    amount_scaler = StandardScaler()

    df["amount_scaled"] = (
        amount_scaler.fit_transform(
            df[["amount"]]
        )
    )

    # --------------------------------------------------------
    # TIME SCALER
    # --------------------------------------------------------

    time_scaler = StandardScaler()

    df["time_scaled"] = (
        time_scaler.fit_transform(
            df[["time"]]
        )
    )

    # --------------------------------------------------------
    # HEURE DE LA JOURNÉE
    # --------------------------------------------------------

    df["hour_of_day"] = (
        df["time"] % 86400
    ) / 3600

    # --------------------------------------------------------
    # Z-SCORE DU MONTANT
    # --------------------------------------------------------

    mean_amt = df["amount"].mean()

    std_amt = df["amount"].std()

    df["amount_zscore"] = (
        (df["amount"] - mean_amt)
        / std_amt
    )

    # --------------------------------------------------------
    # PARAMÈTRES DE PREPROCESSING
    # --------------------------------------------------------

    preprocessing_params = {

        "amount": {
            "mean": float(
                amount_scaler.mean_[0]
            ),
            "std": float(
                amount_scaler.scale_[0]
            )
        },

        "time": {
            "mean": float(
                time_scaler.mean_[0]
            ),
            "std": float(
                time_scaler.scale_[0]
            )
        },

        "amount_zscore": {
            "mean": float(mean_amt),
            "std": float(std_amt)
        }
    }

    # --------------------------------------------------------
    # SUPPRESSION DES COLONNES INUTILES
    # --------------------------------------------------------

    df = df.drop(
        columns=[
            "id",
            "time",
            "amount",
            "created_at"
        ],
        errors="ignore"
    )

    return df, preprocessing_params


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data():

    print("\n")

    print("=" * 60)
    print("🔧 FRAUD DETECTION - DATA PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\n📥 Chargement des données...")

    df = load_data()

    print(
        f"✅ {len(df):,} transactions chargées."
    )

    # ========================================================
    # DATASET ORIGINAL
    # ========================================================

    original_total = len(df)

    original_fraud = int(
        (df["class"] == 1).sum()
    )

    original_legitimate = (
        original_total - original_fraud
    )

    original_fraud_rate = (
        original_fraud
        / original_total
        * 100
    )

    print("\n📊 DATASET ORIGINAL")

    print(
        f"   Transactions : "
        f"{original_total:,}"
    )

    print(
        f"   Fraudes      : "
        f"{original_fraud:,}"
    )

    print(
        f"   Légitimes    : "
        f"{original_legitimate:,}"
    )

    print(
        f"   Taux fraude  : "
        f"{original_fraud_rate:.3f}%"
    )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    df, preprocessing_params = feature_engineering(
        df
    )

    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X = df.drop(
        columns=["class"]
    )

    y = df["class"]

    print(
        f"\n📊 Features : "
        f"{X.shape[1]}"
    )

    print(
        f"   Transactions : "
        f"{len(X):,}"
    )

    print(
        f"   Fraudes      : "
        f"{y.sum():,}"
    )

    # --------------------------------------------------------
    # AFFICHAGE ORDRE DES FEATURES
    # --------------------------------------------------------

    print("\n📋 Ordre des features :")

    for index, feature in enumerate(
        X.columns,
        start=1
    ):
        print(
            f"   {index:02d}. {feature}"
        )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print(
        "\n✂️ Séparation train/test..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        f"   Train : "
        f"{X_train.shape}"
    )

    print(
        f"   Test  : "
        f"{X_test.shape}"
    )

    print(
        f"   Fraudes train : "
        f"{y_train.sum():,}"
    )

    print(
        f"   Fraudes test  : "
        f"{y_test.sum():,}"
    )

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    print(
        "\n⚖️ Application de SMOTE..."
    )

    smote = SMOTE(
        random_state=42
    )

    (
        X_train_res,
        y_train_res
    ) = smote.fit_resample(
        X_train,
        y_train
    )

    print(
        f"✅ Train après SMOTE : "
        f"{X_train_res.shape}"
    )

    print(
        f"   Fraudes train après SMOTE : "
        f"{y_train_res.sum():,}"
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "original_dataset": {

            "total_transactions":
                original_total,

            "fraud_transactions":
                original_fraud,

            "legitimate_transactions":
                original_legitimate,

            "fraud_rate":
                f"{original_fraud_rate:.3f}%"
        },

        "split": {

            "train_transactions":
                len(X_train),

            "test_transactions":
                len(X_test),

            "train_fraud_transactions":
                int(y_train.sum()),

            "test_fraud_transactions":
                int(y_test.sum())
        },

        "smote": {

            "train_transactions_after_smote":
                len(X_train_res),

            "fraud_transactions_after_smote":
                int(y_train_res.sum())
        },

        "features":
            X.shape[1],

        "feature_names":
            list(X.columns)
    }

    # ========================================================
    # SAVE DATA SPLITS
    # ========================================================

    print(
        "\n💾 Sauvegarde des données..."
    )

    data_to_save = {

        "X_train":
            X_train_res,

        "X_test":
            X_test,

        "y_train":
            y_train_res,

        "y_test":
            y_test,

        "metadata":
            metadata
    }

    joblib.dump(
        data_to_save,
        SPLIT_PATH
    )

    print(
        "✅ Fichier créé :"
    )

    print(
        SPLIT_PATH
    )

    # ========================================================
    # SAVE PREPROCESSING PARAMETERS
    # ========================================================

    print(
        "\n💾 Sauvegarde des paramètres de preprocessing..."
    )

    preprocessing_data = {

        "feature_names":
            list(X.columns),

        "amount_mean":
            preprocessing_params["amount"]["mean"],

        "amount_std":
            preprocessing_params["amount"]["std"],

        "time_mean":
            preprocessing_params["time"]["mean"],

        "time_std":
            preprocessing_params["time"]["std"],

        "amount_zscore_mean":
            preprocessing_params[
                "amount_zscore"
            ]["mean"],

        "amount_zscore_std":
            preprocessing_params[
                "amount_zscore"
            ]["std"]
    }

    joblib.dump(
        preprocessing_data,
        PREPROCESSING_PATH
    )

    print(
        "✅ Paramètres sauvegardés :"
    )

    print(
        PREPROCESSING_PATH
    )

    # ========================================================
    # DISPLAY PARAMETERS
    # ========================================================

    print(
        "\n📐 PARAMÈTRES DE PREPROCESSING"
    )

    print(
        f"   Amount mean : "
        f"{preprocessing_data['amount_mean']:.6f}"
    )

    print(
        f"   Amount std  : "
        f"{preprocessing_data['amount_std']:.6f}"
    )

    print(
        f"   Time mean   : "
        f"{preprocessing_data['time_mean']:.6f}"
    )

    print(
        f"   Time std    : "
        f"{preprocessing_data['time_std']:.6f}"
    )

    print(
        f"   Amount Z mean : "
        f"{preprocessing_data['amount_zscore_mean']:.6f}"
    )

    print(
        f"   Amount Z std  : "
        f"{preprocessing_data['amount_zscore_std']:.6f}"
    )

    # ========================================================
    # FIN
    # ========================================================

    print("\n")

    print("=" * 60)
    print("✅ PREPROCESSING TERMINÉ")
    print("=" * 60)

    return (
        X_train_res,
        X_test,
        y_train_res,
        y_test
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepare_data()