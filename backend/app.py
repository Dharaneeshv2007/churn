import logging
import os

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

from model.lstm_model import (
    build_lstm_model,
    train_lstm_model,
    evaluate_model as eval_lstm,
)
from model.gru_model import (
    build_gru_model,
    train_gru_model,
    evaluate_model as eval_gru,
)


# ============================================================
# APP INITIALIZATION
# ============================================================

app = Flask(__name__)
logger = logging.getLogger(__name__)


# ============================================================
# CORS CONFIGURATION
# ============================================================
#
# Current Vercel frontend:
# https://churn-b2kwdxa1u-dharaneesh-s-projects4.vercel.app
#
# Old Vercel frontend:
# https://churn-56cwmaynb-dharaneesh-s-projects4.vercel.app
#
# FRONTEND_ORIGINS can also be configured in Render environment
# variables as a comma-separated list.
# ============================================================

default_frontend_origins = {
    "https://churn-b2kwdxa1u-dharaneesh-s-projects4.vercel.app",
    "https://churn-56cwmaynb-dharaneesh-s-projects4.vercel.app",
}

configured_frontend_origins = {
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
}

allowed_origins = sorted(
    default_frontend_origins | configured_frontend_origins
)

CORS(
    app,
    resources={
        r"/*": {
            "origins": allowed_origins
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# Optional logging so Render logs clearly show the configured origins
logger.info("Allowed CORS origins: %s", allowed_origins)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "churnprediction.csv",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.h5",
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl",
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "encoder.pkl",
)


# ============================================================
# REFERENCE DATA
# ============================================================

REFERENCE_DATA = load_and_prepare_data(DATA_PATH)

REFERENCE_STATS = {
    "MonthlyCharges": float(
        REFERENCE_DATA["MonthlyCharges"].mean()
    ),
    "tenure": float(
        REFERENCE_DATA["tenure"].median()
    ),
    "TotalCharges": float(
        REFERENCE_DATA["TotalCharges"].mean()
    ),
}


# ============================================================
# VALIDATE CUSTOMER DATA
# ============================================================

def _validate_customer(data, encoder):

    if not isinstance(data, dict) or not data:
        raise ValueError(
            "Request body must be a non-empty JSON object"
        )

    required = {
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

    missing = sorted(required - data.keys())

    if missing:
        raise ValueError(
            f"Missing required fields: {', '.join(missing)}"
        )

    # Numeric validation
    for name in (
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ):

        try:
            value = float(data[name])
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be numeric"
            ) from None

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

    # Categorical validation
    categorical_names = (
        "gender",
        "Partner",
        "Dependents",
        "InternetService",
        "Contract",
    )

    for index, name in enumerate(categorical_names):

        if data[name] not in encoder.categories_[index]:

            raise ValueError(
                f"Unknown value for {name}: {data[name]}"
            )


# ============================================================
# LOAD CUSTOMER CONTEXT
# ============================================================

def _load_customer_context(data):

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    _validate_customer(data, encoder)

    X, _, _, _, feature_names = preprocess_data(
        pd.DataFrame([data]),
        fit=False,
        scaler=scaler,
        encoder=encoder,
    )

    reference_X, _, _, _, _ = preprocess_data(
        REFERENCE_DATA,
        fit=False,
        scaler=scaler,
        encoder=encoder,
    )

    return (
        X,
        reference_X[:SHAP_BACKGROUND_SIZE],
        feature_names,
    )


# ============================================================
# PREDICT PROBABILITY
# ============================================================

def _predict_probability(model, X):

    model_input = X

    # LSTM/GRU models expect 3D input
    if (
        len(getattr(model, "input_shape", ())) == 3
        and X.ndim == 2
    ):
        model_input = X.reshape(
            X.shape[0],
            X.shape[1],
            1,
        )

    prediction = model.predict(
        model_input,
        verbose=0,
    )

    return float(
        np.asarray(prediction).reshape(-1)[0]
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return (
        "🚀 Customer Churn Backend is Running Successfully"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "cors_origins": allowed_origins,
        "model_exists": os.path.exists(MODEL_PATH),
    })


# ============================================================
# TRAIN
# ============================================================

@app.route("/train", methods=["GET"])
def train():

    df = load_and_prepare_data(DATA_PATH)

    X, y, scaler, encoder, _feature_names = preprocess_data(
        df,
        fit=True,
    )

    # --------------------------------------------------------
    # Train LSTM
    # --------------------------------------------------------

    lstm_model = build_lstm_model(
        X.shape[1]
    )

    train_lstm_model(
        lstm_model,
        X,
        y,
    )

    lstm_metrics = eval_lstm(
        lstm_model,
        X,
        y,
    )

    # --------------------------------------------------------
    # Train GRU
    # --------------------------------------------------------

    gru_model = build_gru_model(
        X.shape[1]
    )

    train_gru_model(
        gru_model,
        X,
        y,
    )

    gru_metrics = eval_gru(
        gru_model,
        X,
        y,
    )

    # --------------------------------------------------------
    # Select Best Model
    # --------------------------------------------------------

    if lstm_metrics["f1"] >= gru_metrics["f1"]:

        best_model = lstm_model
        best_metrics = lstm_metrics

    else:

        best_model = gru_model
        best_metrics = gru_metrics

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    best_model.save(
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    joblib.dump(
        encoder,
        ENCODER_PATH,
    )

    return jsonify({
        "status": "Model trained successfully",
        "metrics": best_metrics,
    })


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST", "OPTIONS"],
)
def predict():

    # --------------------------------------------------------
    # Explicit OPTIONS response
    # --------------------------------------------------------
    #
    # Flask-CORS normally handles this automatically, but
    # explicitly supporting OPTIONS makes the endpoint robust
    # against browser preflight requests.
    # --------------------------------------------------------

    if request.method == "OPTIONS":

        return "", 200

    try:

        data = request.get_json(
            silent=True
        )

        if not os.path.exists(MODEL_PATH):

            return jsonify({
                "error": "Model not trained. Call /train first"
            }), 400

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        (
            X,
            background_X,
            feature_names,
        ) = _load_customer_context(data)

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = load_model(
            MODEL_PATH
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prob = _predict_probability(
            model,
            X,
        )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if prob < 0.4:

            risk = "Low"

        elif prob < 0.75:

            risk = "Medium"

        else:

            risk = "High"

        # ----------------------------------------------------
        # Estimated time to churn
        # ----------------------------------------------------

        if risk == "High":

            time_to_churn = "15-30 days"

        elif risk == "Medium":

            time_to_churn = "30-90 days"

        else:

            time_to_churn = "90+ days"

        # ----------------------------------------------------
        # Customer Lifetime Value
        # ----------------------------------------------------

        clv = calculate_clv(
            data.get("tenure", 0),
            data.get("MonthlyCharges", 0),
        )

        # ----------------------------------------------------
        # SHAP Explanation
        # ----------------------------------------------------

        try:

            prediction_explanation = (
                build_prediction_explanation(
                    model,
                    X,
                    feature_names,
                    input_data=data,
                    reference_stats=REFERENCE_STATS,
                    background_X=background_X,
                )
            )

            top_reasons = (
                prediction_explanation["top_reasons"]
            )

        except ShapExplanationError as error:

            logger.exception(
                "SHAP explanation failed during prediction"
            )

            prediction_explanation = {
                "error": str(error)
            }

            top_reasons = []

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        action = get_recommendation(
            risk
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "churn_probability": round(
                prob,
                2,
            ),

            "risk_level": risk,

            "time_to_churn": time_to_churn,

            "customer_value": clv,

            "recommendation": action,

            "top_reasons": top_reasons,

            "recommended_action": action,

            "prediction_explanation":
                prediction_explanation,
        })

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except (
        TypeError,
        RuntimeError,
        OSError,
    ) as error:

        logger.exception(
            "Prediction endpoint failed"
        )

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# EXPLAIN
# ============================================================

@app.route(
    "/explain",
    methods=["POST", "OPTIONS"],
)
def explain():

    # Explicit CORS preflight support
    if request.method == "OPTIONS":

        return "", 200

    try:

        data = request.get_json(
            silent=True
        )

        if not os.path.exists(MODEL_PATH):

            return jsonify({
                "error": "Model not trained"
            }), 400

        # ----------------------------------------------------
        # Prepare customer context
        # ----------------------------------------------------

        (
            X,
            background_X,
            feature_names,
        ) = _load_customer_context(data)

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = load_model(
            MODEL_PATH
        )

        # ----------------------------------------------------
        # Generate explanation
        # ----------------------------------------------------

        prediction_explanation = (
            build_prediction_explanation(
                model,
                X,
                feature_names,
                input_data=data,
                reference_stats=REFERENCE_STATS,
                background_X=background_X,
            )
        )

        return jsonify({

            "message": "Explanation generated",

            "prediction_explanation":
                prediction_explanation,

            **prediction_explanation,
        })

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except ShapExplanationError as error:

        logger.exception(
            "SHAP ERROR"
        )

        return jsonify({
            "error": "Unable to generate SHAP explanation",
            "details": str(error),
        }), 500

    except (
        OSError,
        TypeError,
        RuntimeError,
    ) as error:

        logger.exception(
            "Explain endpoint failed"
        )

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )