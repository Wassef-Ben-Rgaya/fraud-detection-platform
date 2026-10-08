import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    """Créer la connexion PostgreSQL via SQLAlchemy."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "fraud_db")
    user = os.getenv("POSTGRES_USER", "fraud_user")
    pwd  = os.getenv("POSTGRES_PASSWORD", "admin")

    url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url)

engine = get_engine()
