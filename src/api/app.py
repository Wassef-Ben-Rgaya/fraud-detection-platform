from flask import Flask, jsonify
from flask_cors import CORS

from src.api.routes import api_bp


app = Flask(__name__)

CORS(app)

app.register_blueprint(api_bp, url_prefix="/api")


@app.route("/")
def index():
    return jsonify({
        "message": "Fraud Detection API is running 🚀",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict [POST]",
            "batch": "/api/predict/batch [POST]",
            "metrics": "/api/metrics",
            "model": "/api/model",
            "predictions": "/api/predictions",
            "database_health": "/api/database/health"
        }
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )