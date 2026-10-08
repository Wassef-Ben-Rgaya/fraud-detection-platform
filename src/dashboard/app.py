import streamlit as st

# ============================================================
# STREAMLIT PAGE CONFIG
# IMPORTANT : doit être la première commande Streamlit
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# IMPORTS
# ============================================================

import os
import pickle
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sqlalchemy import create_engine
from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PREPROCESSING_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "preprocessing_params.pkl"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

IN_DOCKER = os.path.exists("/.dockerenv")


# ============================================================
# API CONFIGURATION
# ============================================================

# AWS API
AWS_API_URL = os.getenv(
    "AWS_API_URL",
    "https://u85vpnda58.execute-api.eu-north-1.amazonaws.com"
).rstrip("/")

USE_AWS_API = os.getenv(
    "USE_AWS_API",
    "true"
).lower() == "true"


# Local/Docker API
if IN_DOCKER:

    LOCAL_API_URL = os.getenv(
        "API_URL_DOCKER",
        "http://api:5000/api"
    )

else:

    LOCAL_API_URL = os.getenv(
        "API_URL",
        "http://localhost:5000/api"
    )


if USE_AWS_API:

    API_URL = AWS_API_URL

else:

    API_URL = LOCAL_API_URL


# ============================================================
# POSTGRESQL CONFIGURATION
# ============================================================

if IN_DOCKER:

    POSTGRES_HOST = os.getenv(
        "POSTGRES_HOST_DOCKER",
        "postgres"
    )

else:

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


# ============================================================
# PREPROCESSING PARAMETERS
# ============================================================

DEFAULT_PREPROCESSING_PARAMS = {
    "amount_mean": 88.349619,
    "amount_std": 250.119670,
    "time_mean": 94813.859575,
    "time_std": 47488.062585,
    "amount_zscore_mean": 88.349619,
    "amount_zscore_std": 250.120109
}


@st.cache_data
def load_preprocessing_params():

    if PREPROCESSING_PATH.exists():

        try:

            with open(
                PREPROCESSING_PATH,
                "rb"
            ) as f:

                params = pickle.load(f)

            return params

        except Exception:

            return DEFAULT_PREPROCESSING_PARAMS

    return DEFAULT_PREPROCESSING_PARAMS


PREPROCESSING_PARAMS = load_preprocessing_params()


# ============================================================
# PREPROCESSING VALUES
# ============================================================

AMOUNT_MEAN = float(
    PREPROCESSING_PARAMS.get(
        "amount_mean",
        DEFAULT_PREPROCESSING_PARAMS["amount_mean"]
    )
)

AMOUNT_STD = float(
    PREPROCESSING_PARAMS.get(
        "amount_std",
        DEFAULT_PREPROCESSING_PARAMS["amount_std"]
    )
)

TIME_MEAN = float(
    PREPROCESSING_PARAMS.get(
        "time_mean",
        DEFAULT_PREPROCESSING_PARAMS["time_mean"]
    )
)

TIME_STD = float(
    PREPROCESSING_PARAMS.get(
        "time_std",
        DEFAULT_PREPROCESSING_PARAMS["time_std"]
    )
)

AMOUNT_ZSCORE_MEAN = float(
    PREPROCESSING_PARAMS.get(
        "amount_zscore_mean",
        DEFAULT_PREPROCESSING_PARAMS["amount_zscore_mean"]
    )
)

AMOUNT_ZSCORE_STD = float(
    PREPROCESSING_PARAMS.get(
        "amount_zscore_std",
        DEFAULT_PREPROCESSING_PARAMS["amount_zscore_std"]
    )
)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [

    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
    "v17",
    "v18",
    "v19",
    "v20",
    "v21",
    "v22",
    "v23",
    "v24",
    "v25",
    "v26",
    "v27",
    "v28",

    "amount_scaled",
    "time_scaled",
    "hour_of_day",
    "amount_zscore"
]


# ============================================================
# FEATURE COUNT VALIDATION
# ============================================================

if len(FEATURE_NAMES) != 32:

    raise RuntimeError(
        f"Invalid feature configuration: "
        f"{len(FEATURE_NAMES)} features instead of 32."
    )


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def scale_amount(amount):

    return (
        float(amount) - AMOUNT_MEAN
    ) / AMOUNT_STD


def scale_time(time_value):

    return (
        float(time_value) - TIME_MEAN
    ) / TIME_STD


def calculate_amount_zscore(amount):

    return (
        float(amount) - AMOUNT_ZSCORE_MEAN
    ) / AMOUNT_ZSCORE_STD


def reconstruct_time_from_hour(hour):

    """
    Approximation du champ original 'time'.

    Le dataset original utilise le nombre de secondes
    depuis le début de la période.

    Avec uniquement l'heure, on ne peut pas reconstruire
    exactement le timestamp original.
    """

    seconds_per_day = 24 * 60 * 60

    mean_day = int(
        TIME_MEAN // seconds_per_day
    )

    reconstructed_time = (
        mean_day * seconds_per_day
        + int(hour) * 60 * 60
    )

    return reconstructed_time


# ============================================================
# BUILD 32 FEATURES
# ============================================================

def build_features(
    amount,
    hour,
    v1,
    v2,
    v14
):

    features = [

        # ----------------------------------------------------
        # V1 - V28
        # ----------------------------------------------------

        float(v1),
        float(v2),

        0.0,  # v3
        0.0,  # v4
        0.0,  # v5
        0.0,  # v6
        0.0,  # v7
        0.0,  # v8
        0.0,  # v9
        0.0,  # v10
        0.0,  # v11
        0.0,  # v12
        0.0,  # v13

        float(v14),

        0.0,  # v15
        0.0,  # v16
        0.0,  # v17
        0.0,  # v18
        0.0,  # v19
        0.0,  # v20
        0.0,  # v21
        0.0,  # v22
        0.0,  # v23
        0.0,  # v24
        0.0,  # v25
        0.0,  # v26
        0.0,  # v27
        0.0,  # v28

        # ----------------------------------------------------
        # ENGINEERED FEATURES
        # ----------------------------------------------------

        scale_amount(amount),

        scale_time(
            reconstruct_time_from_hour(hour)
        ),

        float(hour),

        calculate_amount_zscore(amount)
    ]

    if len(features) != 32:

        raise ValueError(
            f"Feature count = {len(features)}. "
            f"Expected 32."
        )

    return features


# ============================================================
# DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_engine():

    database_url = (
        "postgresql+psycopg2://"
        f"{POSTGRES_USER}:"
        f"{POSTGRES_PASSWORD}@"
        f"{POSTGRES_HOST}:"
        f"{POSTGRES_PORT}/"
        f"{POSTGRES_DB}"
    )

    return create_engine(
        database_url,
        pool_pre_ping=True
    )


# ============================================================
# DATABASE QUERIES
# ============================================================

@st.cache_data(ttl=60)
def load_summary():

    engine = get_engine()

    return pd.read_sql(
        "SELECT * FROM fraud_summary",
        engine
    )


@st.cache_data(ttl=60)
def load_sample():

    engine = get_engine()

    return pd.read_sql(
        """
        SELECT
            amount,
            class,
            time
        FROM transactions
        ORDER BY RANDOM()
        LIMIT 5000
        """,
        engine
    )


@st.cache_data(ttl=10)
def load_predictions(limit=100):

    engine = get_engine()

    query = f"""
        SELECT
            id AS prediction_id,
            transaction_id,
            predicted_class,
            fraud_probability,
            model_version,
            predicted_at
        FROM predictions
        ORDER BY predicted_at DESC
        LIMIT {int(limit)}
    """

    return pd.read_sql(query, engine)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔍 Fraud Detection"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Fraud Detection Platform"
)

st.sidebar.caption(
    f"Environment: "
    f"{'Docker' if IN_DOCKER else 'Windows'}"
)

st.sidebar.caption(
    f"API: "
    f"{'AWS' if USE_AWS_API else 'Local'}"
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🔎 Predict Transaction",
        "🧾 Prediction History",
        "📈 Model Metrics"
    ]
)


# ============================================================
# PAGE 1 : DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.title(
        "📊 Fraud Detection Dashboard"
    )

    st.markdown("---")

    try:

        summary = load_summary()

        df = load_sample()

        # ----------------------------------------------------
        # VALIDATE DATABASE
        # ----------------------------------------------------

        if summary.empty:

            st.warning(
                "No summary data available."
            )

            st.stop()

        # ----------------------------------------------------
        # KPIs
        # ----------------------------------------------------

        total = int(
            summary[
                "total_transactions"
            ].sum()
        )

        fraud_rows = summary[
            summary["class"] == 1
        ]

        legitimate_rows = summary[
            summary["class"] == 0
        ]

        if fraud_rows.empty:

            frauds = 0

            avg_fraud_amount = 0.0

        else:

            frauds = int(
                fraud_rows[
                    "total_transactions"
                ].iloc[0]
            )

            avg_fraud_amount = float(
                fraud_rows[
                    "avg_amount"
                ].iloc[0]
            )

        if total > 0:

            rate = round(
                frauds / total * 100,
                3
            )

        else:

            rate = 0.0

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            f"{total:,}"
        )

        col2.metric(
            "Fraud Cases",
            f"{frauds:,}"
        )

        col3.metric(
            "Fraud Rate",
            f"{rate}%"
        )

        col4.metric(
            "Avg Fraud Amount",
            f"${avg_fraud_amount:.2f}"
        )

        st.markdown("---")

        # ----------------------------------------------------
        # PIE CHART
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(

                values=[
                    total - frauds,
                    frauds
                ],

                names=[
                    "Legitimate",
                    "Fraud"
                ],

                color_discrete_sequence=[
                    "#2ecc71",
                    "#e74c3c"
                ],

                title="Transaction Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # HISTOGRAM
        # ----------------------------------------------------

        with col2:

            df["class_label"] = df[
                "class"
            ].map(
                {
                    0: "Legitimate",
                    1: "Fraud"
                }
            )

            fig = px.histogram(

                df,

                x="amount",

                color="class_label",

                nbins=50,

                title=(
                    "Amount Distribution "
                    "by Class"
                ),

                color_discrete_map={
                    "Legitimate": "#2ecc71",
                    "Fraud": "#e74c3c"
                },

                barmode="overlay",

                opacity=0.7
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # SCATTER PLOT
        # ----------------------------------------------------

        scatter_df = df.sample(
            min(1000, len(df)),
            random_state=42
        )

        fig = px.scatter(

            scatter_df,

            x="time",

            y="amount",

            color="class_label",

            color_discrete_map={
                "Legitimate": "#2ecc71",
                "Fraud": "#e74c3c"
            },

            title="Transactions over Time",

            opacity=0.6
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # DATA TABLE
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📋 Sample Transactions"
        )

        display_df = df.copy()

        display_df = display_df[
            [
                "amount",
                "class",
                "time"
            ]
        ]

        display_df["class"] = (
            display_df["class"]
            .map(
                {
                    0: "LEGITIMATE",
                    1: "FRAUD"
                }
            )
        )

        st.dataframe(
            display_df.head(20),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Dashboard Error: {e}"
        )


# ============================================================
# PAGE 2 : PREDICT TRANSACTION
# ============================================================

elif page == "🔎 Predict Transaction":

    st.title(
        "🔎 Predict a Transaction"
    )

    st.markdown("---")

    st.info(
        "Enter transaction features to predict "
        "if the transaction is fraudulent."
    )

    # --------------------------------------------------------
    # PREPROCESSING INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Preprocessing parameters",
        expanded=False
    ):

        st.write(
            "The dashboard uses the same preprocessing "
            "parameters as the trained model."
        )

        preprocessing_df = pd.DataFrame(
            {
                "Parameter": [
                    "Amount Mean",
                    "Amount Std",
                    "Time Mean",
                    "Time Std",
                    "Amount Z-Score Mean",
                    "Amount Z-Score Std"
                ],

                "Value": [
                    AMOUNT_MEAN,
                    AMOUNT_STD,
                    TIME_MEAN,
                    TIME_STD,
                    AMOUNT_ZSCORE_MEAN,
                    AMOUNT_ZSCORE_STD
                ]
            }
        )

        st.dataframe(
            preprocessing_df,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(

            "Transaction Amount ($)",

            min_value=0.0,

            max_value=100000.0,

            value=100.0,

            step=1.0
        )

        hour = st.slider(

            "Hour of Day",

            min_value=0,

            max_value=23,

            value=12
        )

    with col2:

        v1 = st.number_input(

            "V1",

            min_value=-10.0,

            max_value=10.0,

            value=0.0,

            step=0.1
        )

        v2 = st.number_input(

            "V2",

            min_value=-10.0,

            max_value=10.0,

            value=0.0,

            step=0.1
        )

        v14 = st.number_input(

            "V14",

            min_value=-10.0,

            max_value=10.0,

            value=0.0,

            step=0.1
        )

    # --------------------------------------------------------
    # FEATURE PREVIEW
    # --------------------------------------------------------

    try:

        preview_features = build_features(
            amount=amount,
            hour=hour,
            v1=v1,
            v2=v2,
            v14=v14
        )

        with st.expander(
            "🔧 Feature vector sent to the model",
            expanded=False
        ):

            feature_df = pd.DataFrame(
                {
                    "Feature": FEATURE_NAMES,
                    "Value": preview_features
                }
            )

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:

        st.error(
            f"Feature preprocessing error: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict",
        type="primary",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # BUILD FEATURES
            # ------------------------------------------------

            features = build_features(

                amount=amount,

                hour=hour,

                v1=v1,

                v2=v2,

                v14=v14
            )

            # ------------------------------------------------
            # VALIDATE FEATURE COUNT
            # ------------------------------------------------

            if len(features) != 32:

                st.error(
                    f"Invalid feature count: "
                    f"{len(features)}. "
                    f"Expected 32."
                )

                st.stop()

            # ------------------------------------------------
            # API REQUEST
            # ------------------------------------------------

            with st.spinner(
                "Analyzing transaction..."
            ):

                api_response = requests.post(

                    f"{API_URL}/predict",

                    json={
                        "features": features
                    },

                    timeout=30
                )

            # ------------------------------------------------
            # HTTP VALIDATION
            # ------------------------------------------------

            api_response.raise_for_status()

            response = api_response.json()

            # ------------------------------------------------
            # RESPONSE DATA
            # ------------------------------------------------

            prediction = response.get(
                "prediction"
            )

            risk_level = response.get(
                "risk_level",
                "UNKNOWN"
            )

            fraud_probability = float(
                response.get(
                    "fraud_probability",
                    0.0
                )
            )

            model_name = response.get(
                "model",
                "Unknown"
            )

            model_version = response.get(
                "version",
                "Unknown"
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown("---")

            if prediction == 1:

                st.error(
                    f"🚨 FRAUD DETECTED — "
                    f"Risk: {risk_level}"
                )

            else:

                st.success(
                    f"✅ LEGITIMATE — "
                    f"Risk: {risk_level}"
                )

            # ------------------------------------------------
            # MODEL INFORMATION
            # ------------------------------------------------

            st.info(
                f"Model: **{model_name}** | "
                f"Version: **{model_version}**"
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            col1.metric(

                "Fraud Probability",

                f"{fraud_probability * 100:.2f}%"
            )

            col2.metric(
                "Risk Level",
                risk_level
            )

            prediction_id = response.get("prediction_id")

            if prediction_id is not None:

                st.caption(
                    f"Prediction ID: **{prediction_id}**"
                )

            # ------------------------------------------------
            # GAUGE
            # ------------------------------------------------

            fig = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=fraud_probability * 100,

                    title={
                        "text":
                        "Fraud Probability (%)"
                    },

                    gauge={

                        "axis": {
                            "range": [0, 100]
                        },

                        "bar": {
                            "color": "#e74c3c"
                        },

                        "steps": [

                            {
                                "range": [0, 30],
                                "color": "#2ecc71"
                            },

                            {
                                "range": [30, 70],
                                "color": "#f39c12"
                            },

                            {
                                "range": [70, 100],
                                "color": "#e74c3c"
                            }
                        ]
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ------------------------------------------------
            # API RAW RESPONSE
            # ------------------------------------------------

            with st.expander(
                "🔧 API Response"
            ):

                st.json(response)

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ API timeout. "
                "The prediction service took too long."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                f"❌ Cannot connect to API:\n\n"
                f"{API_URL}"
            )

        except requests.exceptions.HTTPError as e:

            st.error(
                f"❌ API HTTP error: {e}"
            )

            try:

                st.json(
                    api_response.json()
                )

            except Exception:

                st.text(
                    api_response.text
                )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )


# ============================================================
# PAGE 3 : PREDICTION HISTORY
# ============================================================

elif page == "🧾 Prediction History":

    st.title("🧾 Prediction History")
    st.markdown("---")

    st.info(
        "Historique réel des prédictions enregistrées dans PostgreSQL "
        "via l'API Flask."
    )

    try:

        if st.button("🔄 Refresh History", use_container_width=True):
            load_predictions.clear()

        predictions = load_predictions(100)

        if predictions.empty:

            st.warning("Aucune prédiction enregistrée pour le moment.")
            st.stop()

        # ----------------------------------------------------
        # NORMALIZE DATA
        # ----------------------------------------------------

        predictions["predicted_class"] = predictions[
            "predicted_class"
        ].astype(int)

        predictions["fraud_probability"] = predictions[
            "fraud_probability"
        ].astype(float)

        predictions["label"] = predictions[
            "predicted_class"
        ].map({0: "LEGITIMATE", 1: "FRAUD"})

        predictions["risk_level"] = pd.cut(
            predictions["fraud_probability"],
            bins=[-0.001, 0.30, 0.70, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"]
        )

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            label_filter = st.selectbox(
                "Prediction",
                ["All", "LEGITIMATE", "FRAUD"]
            )

        with col2:
            risk_filter = st.selectbox(
                "Risk Level",
                ["All", "LOW", "MEDIUM", "HIGH"]
            )

        with col3:
            min_probability = st.slider(
                "Minimum Fraud Probability",
                0.0,
                1.0,
                0.0,
                0.01
            )

        filtered = predictions.copy()

        if label_filter != "All":
            filtered = filtered[filtered["label"] == label_filter]

        if risk_filter != "All":
            filtered = filtered[filtered["risk_level"] == risk_filter]

        filtered = filtered[
            filtered["fraud_probability"] >= min_probability
        ]

        # ----------------------------------------------------
        # KPIs
        # ----------------------------------------------------

        total_predictions = len(predictions)
        fraud_predictions = int((predictions["predicted_class"] == 1).sum())
        high_risk = int((predictions["risk_level"] == "HIGH").sum())
        avg_probability = float(predictions["fraud_probability"].mean())

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Predictions", f"{total_predictions:,}")
        k2.metric("Fraud Predictions", f"{fraud_predictions:,}")
        k3.metric("High Risk", f"{high_risk:,}")
        k4.metric("Avg Fraud Probability", f"{avg_probability * 100:.2f}%")

        st.markdown("---")

        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:
            count_df = (
                predictions["label"]
                .value_counts()
                .rename_axis("label")
                .reset_index(name="count")
            )

            fig = px.bar(
                count_df,
                x="label",
                y="count",
                title="Prediction Distribution",
                text="count"
            )

            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                predictions,
                x="fraud_probability",
                nbins=20,
                title="Fraud Probability Distribution"
            )

            fig.update_xaxes(
                title="Fraud Probability",
                tickformat=".0%"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.subheader(
            f"📋 Predictions ({len(filtered)})"
        )

        display = filtered[[
            "prediction_id",
            "transaction_id",
            "predicted_class",
            "label",
            "fraud_probability",
            "risk_level",
            "model_version",
            "predicted_at"
        ]].copy()

        display["fraud_probability"] = (
            display["fraud_probability"] * 100
        ).round(2).astype(str) + "%"

        display["predicted_at"] = pd.to_datetime(
            display["predicted_at"]
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # LATEST PREDICTION DETAILS
        # ----------------------------------------------------

        latest = predictions.iloc[0]

        with st.expander("🔎 Latest Prediction Details", expanded=False):

            st.write({
                "prediction_id": int(latest["prediction_id"]),
                "transaction_id": (
                    None if pd.isna(latest["transaction_id"])
                    else int(latest["transaction_id"])
                ),
                "label": latest["label"],
                "fraud_probability": float(latest["fraud_probability"]),
                "risk_level": str(latest["risk_level"]),
                "model_version": latest["model_version"],
                "predicted_at": str(latest["predicted_at"])
            })

    except Exception as e:

        st.error(
            f"Prediction History Error: {e}"
        )


# ============================================================
# PAGE 4 : MODEL METRICS
# ============================================================

elif page == "📈 Model Metrics":

    st.title(
        "📈 Model Performance"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # L'API AWS actuelle expose POST /predict.
    # Elle n'expose pas GET /metrics.
    #
    # Les métriques ci-dessous correspondent au modèle
    # RandomForest entraîné.
    #
    # --------------------------------------------------------

    model_name = "RandomForest"

    model_version = "1.0.0"

    selection_metric = "f1_score"

    model_metrics = {

        "f1_score": 0.8404,

        "auc_roc": 0.9796,

        "precision": 0.8778,

        "recall": 0.8061
    }

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.info(

        f"Model: **{model_name}** | "
        f"Version: **{model_version}** | "
        f"Selection metric: **{selection_metric}**"
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(

        "F1-Score",

        f"{model_metrics['f1_score']:.4f}"
    )

    col2.metric(

        "AUC-ROC",

        f"{model_metrics['auc_roc']:.4f}"
    )

    col3.metric(

        "Precision",

        f"{model_metrics['precision']:.4f}"
    )

    col4.metric(

        "Recall",

        f"{model_metrics['recall']:.4f}"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # BAR CHART
    # --------------------------------------------------------

    chart_df = pd.DataFrame(

        {
            "Metric": [
                "F1-Score",
                "AUC-ROC",
                "Precision",
                "Recall"
            ],

            "Score": [
                model_metrics["f1_score"],
                model_metrics["auc_roc"],
                model_metrics["precision"],
                model_metrics["recall"]
            ]
        }
    )

    fig = px.bar(

        chart_df,

        x="Metric",

        y="Score",

        title="Model Metrics Overview",

        range_y=[0, 1],

        text="Score"
    )

    fig.update_traces(

        texttemplate="%{text:.4f}",

        textposition="outside"
    )

    st.plotly_chart(

        fig,

        use_container_width=True
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📊 Dataset Information"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Transactions",
        "284,807"
    )

    col2.metric(
        "Fraud Transactions",
        "492"
    )

    col3.metric(
        "Fraud Rate",
        "0.173%"
    )

    # --------------------------------------------------------
    # TRAINING INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🤖 Training Information"
    )

    training_data = pd.DataFrame(

        {
            "Parameter": [

                "Train transactions",

                "Test transactions",

                "Train after SMOTE",

                "Frauds after SMOTE",

                "Number of features",

                "Best model"
            ],

            "Value": [

                "227,845",

                "56,962",

                "454,902",

                "227,451",

                "32",

                "RandomForest"
            ]
        }
    )

    st.dataframe(

        training_data,

        use_container_width=True,

        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔧 Model Features"
    )

    feature_df = pd.DataFrame(

        {
            "Index": range(
                1,
                len(FEATURE_NAMES) + 1
            ),

            "Feature": FEATURE_NAMES
        }
    )

    st.dataframe(

        feature_df,

        use_container_width=True,

        hide_index=True
    )

    # --------------------------------------------------------
    # PREPROCESSING INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "⚙️ Preprocessing Parameters"
    )

    preprocessing_df = pd.DataFrame(

        {
            "Parameter": [

                "Amount mean",

                "Amount std",

                "Time mean",

                "Time std",

                "Amount Z-score mean",

                "Amount Z-score std"
            ],

            "Value": [

                AMOUNT_MEAN,

                AMOUNT_STD,

                TIME_MEAN,

                TIME_STD,

                AMOUNT_ZSCORE_MEAN,

                AMOUNT_ZSCORE_STD
            ]
        }
    )

    st.dataframe(

        preprocessing_df,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Fraud Detection Platform"
)

st.sidebar.caption(
    "RandomForest • v1.0.0"
)