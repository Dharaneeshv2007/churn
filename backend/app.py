import logging
import os
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model

from utils.preprocessing import preprocess_data, load_and_prepare_data
from utils.shap_explainer import (
    SHAP_BACKGROUND_SIZE,
    ShapExplanationError,
    build_prediction_explanation,
)
from utils.recommendation import get_recommendation
from utils.clv import calculate_clv

# Training is intentionally kept available only when explicitly enabled.
from model.lstm_model import build_lstm_model, train_lstm_model, evaluate_model as eval_lstm
from model.gru_model import build_gru_model, train_gru_model, evaluate_model as eval_gru

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================
# CORS
# ============================================================
# Set FRONTEND_ORIGIN on Render to your exact Vercel URL.
# Multiple origins may be separated by commas.
default_origins = [
    "https://churn-chi.vercel.app",
    "https://churn-git-main-dharaneesh-s-projects4.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
env_origins = [x.strip().rstrip("/") for x in os.getenv("FRONTEND_ORIGIN", "").split(",") if x.strip()]
ALLOWED_ORIGINS = list(dict.fromkeys(default_origins + env_origins))

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "churnprediction.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# ============================================================
# CACHED ARTIFACTS
# ============================================================
MODEL = None
SCALER = None
ENCODER = None
REFERENCE_DATA = None
REFERENCE_X = None
REFERENCE_STATS = {}
FEATURE_NAMES = None
SHAP_BACKGROUND_X = None

# Prevent two simultaneous SHAP calculations from competing for
# the limited CPU/memory of a small Render instance.
SHAP_LOCK = Lock()

# Lazily created by utils.shap_explainer and reused.
SHAP_EXPLAINER_READY = False


def _load_artifacts():
    """Load all inference artifacts once at process startup."""
    global MODEL, SCALER, ENCODER
    global REFERENCE_DATA, REFERENCE_X, REFERENCE_STATS
    global FEATURE_NAMES, SHAP_BACKGROUND_X

    try:
        logger.info("Loading model...")
        MODEL = load_model(MODEL_PATH, compile=False)

        logger.info("Loading preprocessing artifacts...")
        SCALER = joblib.load(SCALER_PATH)
        ENCODER = joblib.load(ENCODER_PATH)

        REFERENCE_DATA = load_and_prepare_data(DATA_PATH)
        REFERENCE_X, _, _, _, FEATURE_NAMES = preprocess_data(
            REFERENCE_DATA,
            fit=False,
            scaler=SCALER,
            encoder=ENCODER,
        )

        REFERENCE_STATS = {
            "MonthlyCharges": float(REFERENCE_DATA["MonthlyCharges"].mean()),
            "tenure": float(REFERENCE_DATA["tenure"].median()),
            "TotalCharges": float(REFERENCE_DATA["TotalCharges"].mean()),
        }

        # Use a deterministic, small background for explanations.
        # Keeping this in memory avoids CSV/scaler/encoder work per request.
        size = min(SHAP_BACKGROUND_SIZE, len(REFERENCE_X))
        SHAP_BACKGROUND_X = np.asarray(REFERENCE_X[:size], dtype=np.float32)

        # Warm TensorFlow once so the first customer prediction is not
        # paying the full graph/model initialization cost.
        warm = np.zeros((1, REFERENCE_X.shape[1]), dtype=np.float32)
        _predict_probability(MODEL, warm)

        logger.info(
            "All inference artifacts loaded. features=%d background=%d",
            len(FEATURE_NAMES),
            len(SHAP_BACKGROUND_X),
        )

    except Exception:
        MODEL = None
        SCALER = None
        ENCODER = None
        REFERENCE_DATA = None
        REFERENCE_X = None
        FEATURE_NAMES = None
        SHAP_BACKGROUND_X = None
        logger.exception("Failed to load inference artifacts.")


_load_artifacts()


# ============================================================
# VALIDATION / PREPROCESSING
# ============================================================
REQUIRED_FIELDS = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "InternetService",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
}


def _validate_customer(data):
    if not isinstance(data, dict) or not data:
        raise ValueError("Request body must be a non-empty JSON object")

    missing = sorted(REQUIRED_FIELDS - data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    for name in ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"):
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric") from None
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    # Keep model inputs inside sensible ranges.
    if float(data["SeniorCitizen"]) not in (0.0, 1.0):
        raise ValueError("SeniorCitizen must be 0 or 1")
    if float(data["tenure"]) < 0 or float(data["tenure"]) > 1000:
        raise ValueError("tenure must be between 0 and 1000 months")
    if float(data["MonthlyCharges"]) < 0 or float(data["TotalCharges"]) < 0:
        raise ValueError("Charges cannot be negative")

    for name, categories in zip(
        ("gender", "Partner", "Dependents", "InternetService", "Contract"),
        ENCODER.categories_,
    ):
        if data[name] not in categories:
            raise ValueError(f"Unknown value for {name}: {data[name]}")


def _load_customer_input(data):
    _validate_customer(data)
    X, _, _, _, _ = preprocess_data(
        pd.DataFrame([data]),
        fit=False,
        scaler=SCALER,
        encoder=ENCODER,
    )
    return np.asarray(X, dtype=np.float32)


def _model_input(model, X):
    if (
        len(getattr(model, "input_shape", ())) == 3
        and X.ndim == 2
    ):
        return X.reshape(X.shape[0], X.shape[1], 1)
    return X


def _predict_probability(model, X):
    prediction = model.predict(_model_input(model, X), verbose=0)
    value = float(np.asarray(prediction).reshape(-1)[0])
    if not np.isfinite(value):
        raise RuntimeError("Model returned an invalid probability")
    return float(np.clip(value, 0.0, 1.0))


def _error_response(message, status):
    return jsonify({"error": message}), status


# ============================================================
# ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Customer Churn Backend",
        "status": "running",
        "model_loaded": MODEL is not None,
    })


@app.route("/health", methods=["GET"])
def health():
    ready = MODEL is not None and SCALER is not None and ENCODER is not None
    return jsonify({
        "status": "healthy" if ready else "unhealthy",
        "model_loaded": MODEL is not None,
        "preprocessing_loaded": SCALER is not None and ENCODER is not None,
        "features": len(FEATURE_NAMES) if FEATURE_NAMES is not None else 0,
    }), 200 if ready else 503


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    try:
        if MODEL is None or SCALER is None or ENCODER is None:
            return _error_response("Backend model is not ready. Please try again shortly.", 503)

        data = request.get_json(silent=True)
        X = _load_customer_input(data)
        prob = _predict_probability(MODEL, X)

        if prob < 0.4:
            risk = "Low"
            time_to_churn = "90+ days"
        elif prob < 0.75:
            risk = "Medium"
            time_to_churn = "30-90 days"
        else:
            risk = "High"
            time_to_churn = "15-30 days"

        action = get_recommendation(risk)
        clv = calculate_clv(data["tenure"], data["MonthlyCharges"])

        return jsonify({
            "churn_probability": round(prob, 4),
            "risk_level": risk,
            "time_to_churn": time_to_churn,
            "customer_value": clv,
            "recommendation": action,
            "recommended_action": action,
            "top_reasons": [],
            "prediction_explanation": None,
        })

    except ValueError as error:
        return _error_response(str(error), 400)
    except Exception as error:
        logger.exception("Prediction endpoint failed")
        return _error_response("Prediction failed. Please try again.", 500)


@app.route("/explain", methods=["POST", "OPTIONS"])
def explain():
    if request.method == "OPTIONS":
        return "", 204

    try:
        if MODEL is None or SCALER is None or ENCODER is None:
            return _error_response("Backend model is not ready. Please try again shortly.", 503)

        data = request.get_json(silent=True)
        X = _load_customer_input(data)

        if SHAP_BACKGROUND_X is None:
            return _error_response("Explanation background is not ready.", 503)

        logger.info("Generating explanation for one customer")

        # Serialize explanation calculations on small instances.
        with SHAP_LOCK:
            explanation = build_prediction_explanation(
                MODEL,
                X,
                FEATURE_NAMES,
                input_data=data,
                reference_stats=REFERENCE_STATS,
                background_X=SHAP_BACKGROUND_X,
            )

        return jsonify({
            "message": "Explanation generated",
            "prediction_explanation": explanation,
            **explanation,
        })

    except ValueError as error:
        return _error_response(str(error), 400)
    except ShapExplanationError as error:
        logger.exception("SHAP explanation failed")
        return jsonify({
            "error": "Unable to generate explanation",
            "details": str(error),
        }), 500
    except Exception:
        logger.exception("Unexpected explanation error")
        return _error_response("Explanation failed. Please try again.", 500)


# ============================================================
# OPTIONAL TRAINING
# ============================================================
@app.route("/train", methods=["GET", "POST"])
def train():
    # Never allow an accidental public training request in production.
    if os.getenv("ENABLE_TRAINING", "false").lower() != "true":
        return _error_response(
            "Training endpoint is disabled in production. Set ENABLE_TRAINING=true only when needed.",
            403,
        )

    global MODEL, SCALER, ENCODER, REFERENCE_X, FEATURE_NAMES, SHAP_BACKGROUND_X

    try:
        df = load_and_prepare_data(DATA_PATH)
        X, y, scaler, encoder, feature_names = preprocess_data(df, fit=True)

        lstm_model = build_lstm_model(X.shape[1])
        train_lstm_model(lstm_model, X, y)
        lstm_metrics = eval_lstm(lstm_model, X, y)

        gru_model = build_gru_model(X.shape[1])
        train_gru_model(gru_model, X, y)
        gru_metrics = eval_gru(gru_model, X, y)

        if lstm_metrics["f1"] >= gru_metrics["f1"]:
            best_model, best_metrics = lstm_model, lstm_metrics
        else:
            best_model, best_metrics = gru_model, gru_metrics

        os.makedirs(MODEL_DIR, exist_ok=True)
        best_model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoder, ENCODER_PATH)

        MODEL = load_model(MODEL_PATH, compile=False)
        SCALER = scaler
        ENCODER = encoder
        REFERENCE_X, _, _, _, FEATURE_NAMES = preprocess_data(
            REFERENCE_DATA, fit=False, scaler=SCALER, encoder=ENCODER
        )
        SHAP_BACKGROUND_X = np.asarray(
            REFERENCE_X[:min(SHAP_BACKGROUND_SIZE, len(REFERENCE_X))],
            dtype=np.float32,
        )

        return jsonify({"status": "Model trained successfully", "metrics": best_metrics})

    except Exception:
        logger.exception("Training failed")
        return _error_response("Training failed. Check server logs.", 500)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
