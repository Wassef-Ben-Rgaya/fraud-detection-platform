import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv("D:/fraud-detection-platform/.env")


def get_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "fraud_db")
    user = os.getenv("POSTGRES_USER", "fraud_user")
    password = os.getenv("POSTGRES_PASSWORD", "admin")

    database_url = (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )

    return create_engine(database_url, pool_pre_ping=True)


def ingest_csv(filepath: str):
    """Charger le CSV Kaggle dans PostgreSQL."""

    print(f"📥 Chargement du dataset : {filepath}")

    df = pd.read_csv(filepath)

    # Renommer les colonnes en minuscules
    df.columns = [c.lower() for c in df.columns]

    print(f"✅ {len(df)} transactions chargées")
    print(
        f"   Fraudes : {df['class'].sum()} "
        f"({df['class'].mean() * 100:.2f}%)"
    )

    engine = get_engine()

    print("🔌 Connexion à PostgreSQL...")
    print(
        f"   Host: {os.getenv('POSTGRES_HOST')}\n"
        f"   Port: {os.getenv('POSTGRES_PORT')}\n"
        f"   Database: {os.getenv('POSTGRES_DB')}\n"
        f"   User: {os.getenv('POSTGRES_USER')}"
    )

    # Ingérer dans PostgreSQL
    df.to_sql(
        "transactions",
        engine,
        if_exists="append",
        index=False,
        chunksize=5000
    )

    print("✅ Données insérées dans PostgreSQL avec succès !")


if __name__ == "__main__":
    ingest_csv(
        "D:/fraud-detection-platform/data/raw/creditcard.csv"
    )