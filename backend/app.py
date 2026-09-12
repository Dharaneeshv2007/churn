
import logging
import os
from threading import Lock

# ============================================================
# TensorFlow CPU settings
# ============================================================
# Render normally runs without a GPU. Limit CPU thread usage so
# TensorFlow does not consume the whole small Render instance.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

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

# Training is available only when explicitly enabled.
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
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)


# ============================================================
# CORS
# ============================================================
default_origins = [
    "https://churn-chi.vercel.app",
    "https://churn-git-main-dharaneesh-s-projects4.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

env_origins = [
    x.strip().rstrip("/")
    for x in os.getenv("FRONTEND_ORIGIN", "").split(",")
    if x.strip()
]

ALLOWED_ORIGINS = list(
    dict.fromkeys(default_origins + env_origins)
)

CORS(
    app,
    resources={
        r"/*": {
            "origins": ALLOWED_ORIGINS
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=False,
)


# ============================================================
# PATHS
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
# CACHED INFERENCE ARTIFACTS
# ============================================================
MODEL = None
SCALER = None
ENCODER = None

REFERENCE_DATA = None
REFERENCE_X = None
REFERENCE_STATS = {}

FEATURE_NAMES = None
SHAP_BACKGROUND_X = None

# Prevent multiple expensive SHAP calculations at once.
SHAP_LOCK = Lock()


# ============================================================
# REQUIRED CUSTOMER FIELDS
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


# ============================================================
# MODEL INPUT HELPERS
# ============================================================
def _model_input(model, X):
    """
    Convert preprocessed customer data to the input shape
    expected by the trained LSTM/GRU model.

    Most preprocessing returns a 2D array:
        (samples, features)

    LSTM/GRU models generally expect:
        (samples, timesteps, features)

    If the loaded model has a 3D input shape, reshape safely.
    """

    X = np.asarray(X, dtype=np.float32)

    input_shape = getattr(model, "input_shape", None)

    if input_shape is None:
        return X

    # Example:
    # model.input_shape = (None, 19, 1)
    if len(input_shape) == 3 and X.ndim == 2:

        expected_features = input_shape[-2]
        expected_channels = input_shape[-1]

        # Normal case for this project:
        # (samples, features) -> (samples, features, 1)
        if (
            expected_channels == 1
            and (
                expected_features is None
                or X.shape[1] == expected_features
            )
        ):
            return X.reshape(
                X.shape[0],
                X.shape[1],
                1,
            )

        # Generic fallback when the number of dimensions is right.
        return X.reshape(
            X.shape[0],
            X.shape[1],
            1,
        )

    return X


def _predict_probability(model, X):
    """
    Perform a single model prediction and return a safe
    probability between 0 and 1.

    IMPORTANT:
    This function is defined BEFORE _load_artifacts()
    because startup warm-up uses it.
    """

    if model is None:
        raise RuntimeError("Prediction model is not loaded")

    model_X = _model_input(model, X)

    prediction = model.predict(
        model_X,
        verbose=0,
    )

    values = np.asarray(
        prediction,
        dtype=np.float32,
    ).reshape(-1)

    if values.size == 0:
        raise RuntimeError(
            "Model returned an empty prediction"
        )

    value = float(values[0])

    if not np.isfinite(value):
        raise RuntimeError(
            "Model returned an invalid probability"
        )

    return float(
        np.clip(value, 0.0, 1.0)
    )


# ============================================================
# MODEL WARM-UP
# ============================================================
def _warm_up_model(model, reference_x):
    """
    Run one prediction after loading the model.

    This moves TensorFlow's first-call initialization cost
    to server startup instead of the first customer request.
    """

    if model is None:
        raise RuntimeError(
            "Cannot warm up: model is not loaded"
        )

    if reference_x is None:
        raise RuntimeError(
            "Cannot warm up: reference data is not loaded"
        )

    reference_x = np.asarray(
        reference_x,
        dtype=np.float32,
    )

    if reference_x.ndim != 2 or reference_x.shape[0] == 0:
        raise RuntimeError(
            "Invalid reference data for model warm-up"
        )

    # Use a real preprocessed sample rather than an arbitrary
    # zero vector. This guarantees the feature shape matches
    # the preprocessing pipeline.
    warm = reference_x[:1]

    probability = _predict_probability(
        model,
        warm,
    )

    logger.info(
        "Model warm-up successful. probability=%.6f",
        probability,
    )


# ============================================================
# LOAD ARTIFACTS
# ============================================================
def _load_artifacts():
    """
    Load model, scaler, encoder and reference data once
    when the Gunicorn worker starts.

    Nothing expensive should be repeated for every /predict
    request.
    """

    global MODEL
    global SCALER
    global ENCODER
    global REFERENCE_DATA
    global REFERENCE_X
    global REFERENCE_STATS
    global FEATURE_NAMES
    global SHAP_BACKGROUND_X

    try:
        logger.info("Loading model...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        MODEL = load_model(
            MODEL_PATH,
            compile=False,
        )

        logger.info(
            "Model loaded successfully. input_shape=%s",
            getattr(MODEL, "input_shape", None),
        )

        logger.info(
            "Loading preprocessing artifacts..."
        )

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler file not found: {SCALER_PATH}"
            )

        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(
                f"Encoder file not found: {ENCODER_PATH}"
            )

        SCALER = joblib.load(
            SCALER_PATH
        )

        ENCODER = joblib.load(
            ENCODER_PATH
        )

        logger.info(
            "Scaler and encoder loaded successfully."
        )

        logger.info(
            "Loading reference dataset..."
        )

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        REFERENCE_DATA = load_and_prepare_data(
            DATA_PATH
        )

        (
            REFERENCE_X,
            _,
            _,
            _,
            FEATURE_NAMES,
        ) = preprocess_data(
            REFERENCE_DATA,
            fit=False,
            scaler=SCALER,
            encoder=ENCODER,
        )

        REFERENCE_X = np.asarray(
            REFERENCE_X,
            dtype=np.float32,
        )

        logger.info(
            "Reference data prepared. shape=%s",
            REFERENCE_X.shape,
        )

        # ----------------------------------------------------
        # Reference statistics
        # ----------------------------------------------------
        REFERENCE_STATS = {}

        if "MonthlyCharges" in REFERENCE_DATA.columns:
            REFERENCE_STATS["MonthlyCharges"] = float(
                REFERENCE_DATA["MonthlyCharges"].mean()
            )

        if "tenure" in REFERENCE_DATA.columns:
            REFERENCE_STATS["tenure"] = float(
                REFERENCE_DATA["tenure"].median()
            )

        if "TotalCharges" in REFERENCE_DATA.columns:
            REFERENCE_STATS["TotalCharges"] = float(
                REFERENCE_DATA["TotalCharges"].mean()
            )

        # ----------------------------------------------------
        # SHAP background
        # ----------------------------------------------------
        size = min(
            SHAP_BACKGROUND_SIZE,
            len(REFERENCE_X),
        )

        if size <= 0:
            raise RuntimeError(
                "Reference dataset is empty"
            )

        SHAP_BACKGROUND_X = np.asarray(
            REFERENCE_X[:size],
            dtype=np.float32,
        )

        logger.info(
            "SHAP background prepared. size=%d",
            len(SHAP_BACKGROUND_X),
        )

        # ----------------------------------------------------
        # TensorFlow warm-up
        # ----------------------------------------------------
        _warm_up_model(
            MODEL,
            REFERENCE_X,
        )

        logger.info(
            "=================================================="
        )
        logger.info(
            "ALL INFERENCE ARTIFACTS LOADED SUCCESSFULLY"
        )
        logger.info(
            "Features: %d",
            len(FEATURE_NAMES),
        )
        logger.info(
            "SHAP background: %d",
            len(SHAP_BACKGROUND_X),
        )
        logger.info(
            "=================================================="
        )

    except Exception:
        MODEL = None
        SCALER = None
        ENCODER = None
        REFERENCE_DATA = None
        REFERENCE_X = None
        REFERENCE_STATS = {}
        FEATURE_NAMES = None
        SHAP_BACKGROUND_X = None

        logger.exception(
            "Failed to load inference artifacts."
        )


# ============================================================
# LOAD EVERYTHING BEFORE SERVING REQUESTS
# ============================================================
_load_artifacts()


# ============================================================
# ERROR RESPONSE
# ============================================================
def _error_response(message, status):
    return jsonify({
        "error": message
    }), status


# ============================================================
# CUSTOMER VALIDATION
# ============================================================
def _validate_customer(data):
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "Request body must be a non-empty JSON object"
        )

    missing = sorted(
        REQUIRED_FIELDS - data.keys()
    )

    if missing:
        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing)
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
        except (
            TypeError,
            ValueError,
        ):
            raise ValueError(
                f"{name} must be numeric"
            ) from None

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

    # SeniorCitizen must be binary.
    if float(data["SeniorCitizen"]) not in (
        0.0,
        1.0,
    ):
        raise ValueError(
            "SeniorCitizen must be 0 or 1"
        )

    # Reasonable limits.
    if (
        float(data["tenure"]) < 0
        or float(data["tenure"]) > 1000
    ):
        raise ValueError(
            "tenure must be between 0 and 1000 months"
        )

    if (
        float(data["MonthlyCharges"]) < 0
        or float(data["TotalCharges"]) < 0
    ):
        raise ValueError(
            "Charges cannot be negative"
        )

    # Validate categorical values against the
    # categories learned by the saved encoder.
    categorical_fields = (
        "gender",
        "Partner",
        "Dependents",
        "InternetService",
        "Contract",
    )

    try:
        categories = ENCODER.categories_
    except AttributeError:
        raise RuntimeError(
            "Encoder is not initialized correctly"
        )

    for name, allowed_values in zip(
        categorical_fields,
        categories,
    ):
        if data[name] not in allowed_values:
            raise ValueError(
                f"Unknown value for {name}: {data[name]}"
            )


# ============================================================
# CUSTOMER PREPROCESSING
# ============================================================
def _load_customer_input(data):
    _validate_customer(data)

    X, _, _, _, _ = preprocess_data(
        pd.DataFrame([data]),
        fit=False,
        scaler=SCALER,
        encoder=ENCODER,
    )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 2 or X.shape[0] != 1:
        raise ValueError(
            "Invalid preprocessed customer input shape"
        )

    return X


# ============================================================
# ROUTE: HOME
# ============================================================
@app.route("/", methods=["GET"])
def home():
    ready = (
        MODEL is not None
        and SCALER is not None
        and ENCODER is not None
    )

    return jsonify({
        "service": "Customer Churn Backend",
        "status": "running",
        "model_loaded": MODEL is not None,
        "preprocessing_loaded": (
            SCALER is not None
            and ENCODER is not None
        ),
        "ready": ready,
    })


# ============================================================
# ROUTE: HEALTH
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    ready = (
        MODEL is not None
        and SCALER is not None
        and ENCODER is not None
        and REFERENCE_X is not None
    )

    response = {
        "status": (
            "healthy"
            if ready
            else "unhealthy"
        ),
        "model_loaded": MODEL is not None,
        "preprocessing_loaded": (
            SCALER is not None
            and ENCODER is not None
        ),
        "reference_data_loaded": (
            REFERENCE_X is not None
        ),
        "features": (
            len(FEATURE_NAMES)
            if FEATURE_NAMES is not None
            else 0
        ),
    }

    return jsonify(response), (
        200 if ready else 503
    )


# ============================================================
# ROUTE: PREDICT
# ============================================================
@app.route(
    "/predict",
    methods=["POST", "OPTIONS"],
)
def predict():

    # Browser CORS preflight
    if request.method == "OPTIONS":
        logger.info("[PREDICT] OPTIONS preflight received")
        return "", 204

    logger.info("[PREDICT] POST request received")

    try:
        # ----------------------------------------------------
        # 1. Check backend readiness
        # ----------------------------------------------------
        logger.info("[PREDICT] Checking backend readiness")

        if (
            MODEL is None
            or SCALER is None
            or ENCODER is None
        ):
            logger.error(
                "[PREDICT] Backend artifacts are not ready"
            )

            return _error_response(
                "Backend model is not ready. Please try again shortly.",
                503,
            )

        logger.info(
            "[PREDICT] Backend artifacts are ready"
        )

        # ----------------------------------------------------
        # 2. Read JSON
        # ----------------------------------------------------
        logger.info(
            "[PREDICT] Reading request JSON"
        )

        data = request.get_json(
            silent=True
        )

        if data is None:
            logger.error(
                "[PREDICT] Request JSON is empty or invalid"
            )

            return _error_response(
                "Invalid or empty JSON request body.",
                400,
            )

        logger.info(
            "[PREDICT] JSON received. Fields=%s",
            list(data.keys()),
        )

        # ----------------------------------------------------
        # 3. Preprocess customer input
        # ----------------------------------------------------
        logger.info(
            "[PREDICT] Starting customer preprocessing"
        )

        X = _load_customer_input(data)

        logger.info(
            "[PREDICT] Customer preprocessing completed. shape=%s dtype=%s",
            X.shape,
            X.dtype,
        )

        # ----------------------------------------------------
        # 4. Prepare model input
        # ----------------------------------------------------
        logger.info(
            "[PREDICT] Preparing model input"
        )

        model_X = _model_input(
            MODEL,
            X,
        )

        logger.info(
            "[PREDICT] Model input ready. shape=%s dtype=%s",
            model_X.shape,
            model_X.dtype,
        )

        # ----------------------------------------------------
        # 5. Run TensorFlow prediction
        # ----------------------------------------------------
        logger.info(
            "[PREDICT] Starting model.predict()"
        )

        prediction = MODEL.predict(
            model_X,
            verbose=0,
        )

        logger.info(
            "[PREDICT] model.predict() completed"
        )

        # ----------------------------------------------------
        # 6. Extract probability
        # ----------------------------------------------------
        values = np.asarray(
            prediction,
            dtype=np.float32,
        ).reshape(-1)

        if values.size == 0:
            raise RuntimeError(
                "Model returned an empty prediction"
            )

        prob = float(values[0])

        if not np.isfinite(prob):
            raise RuntimeError(
                "Model returned an invalid probability"
            )

        prob = float(
            np.clip(
                prob,
                0.0,
                1.0,
            )
        )

        logger.info(
            "[PREDICT] Probability calculated: %.6f",
            prob,
        )

        # ----------------------------------------------------
        # 7. Risk classification
        # ----------------------------------------------------
        if prob < 0.4:
            risk = "Low"
            time_to_churn = "90+ days"

        elif prob < 0.75:
            risk = "Medium"
            time_to_churn = "30-90 days"

        else:
            risk = "High"
            time_to_churn = "15-30 days"

        logger.info(
            "[PREDICT] Risk calculated: %s",
            risk,
        )

        # ----------------------------------------------------
        # 8. Recommendation
        # ----------------------------------------------------
        logger.info(
            "[PREDICT] Calculating recommendation"
        )

        action = get_recommendation(
            risk
        )

        logger.info(
            "[PREDICT] Recommendation completed"
        )

        # ----------------------------------------------------
        # 9. Customer lifetime value
        # ----------------------------------------------------
        logger.info(
            "[PREDICT] Calculating customer lifetime value"
        )

        clv = calculate_clv(
            data["tenure"],
            data["MonthlyCharges"],
        )

        logger.info(
            "[PREDICT] CLV completed: %s",
            clv,
        )

        # ----------------------------------------------------
        # 10. Build response
        # ----------------------------------------------------
        response = {
            "churn_probability": round(
                prob,
                4,
            ),
            "risk_level": risk,
            "time_to_churn": time_to_churn,
            "customer_value": clv,
            "recommendation": action,
            "recommended_action": action,

            # SHAP is intentionally NOT executed here.
            "top_reasons": [],
            "prediction_explanation": None,
        }

        logger.info(
            "[PREDICT] Response created successfully"
        )

        # ----------------------------------------------------
        # 11. Return response
        # ----------------------------------------------------
        logger.info(
            "[PREDICT] Returning HTTP 200 response"
        )

        return jsonify(
            response
        ), 200

    except ValueError as error:
        logger.warning(
            "[PREDICT] Validation error: %s",
            error,
        )

        return _error_response(
            str(error),
            400,
        )

    except Exception as error:
        logger.exception(
            "[PREDICT] Prediction endpoint failed: %s",
            error,
        )

        return _error_response(
            "Prediction failed. Please try again.",
            500,
        )

# ============================================================
# ROUTE: EXPLAIN
# ============================================================
@app.route(
    "/explain",
    methods=["POST", "OPTIONS"],
)
def explain():

    # Browser CORS preflight
    if request.method == "OPTIONS":
        return "", 204

    try:
        if (
            MODEL is None
            or SCALER is None
            or ENCODER is None
        ):
            return _error_response(
                "Backend model is not ready. Please try again shortly.",
                503,
            )

        data = request.get_json(
            silent=True
        )

        X = _load_customer_input(data)

        if SHAP_BACKGROUND_X is None:
            return _error_response(
                "Explanation background is not ready.",
                503,
            )

        logger.info(
            "Generating explanation for one customer"
        )

        # Only one SHAP calculation at a time.
        # This prevents multiple users from exhausting
        # the CPU/memory on a small Render instance.
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
        }), 200

    except ValueError as error:
        return _error_response(
            str(error),
            400,
        )

    except ShapExplanationError as error:
        logger.exception(
            "SHAP explanation failed"
        )

        return jsonify({
            "error": "Unable to generate explanation",
            "details": str(error),
        }), 500

    except Exception:
        logger.exception(
            "Unexpected explanation error"
        )

        return _error_response(
            "Explanation failed. Please try again.",
            500,
        )


# ============================================================
# OPTIONAL TRAINING
# ============================================================
@app.route(
    "/train",
    methods=["GET", "POST"],
)
def train():

    # Never allow accidental public training.
    if (
        os.getenv(
            "ENABLE_TRAINING",
            "false",
        ).lower()
        != "true"
    ):
        return _error_response(
            "Training endpoint is disabled in production. "
            "Set ENABLE_TRAINING=true only when needed.",
            403,
        )

    global MODEL
    global SCALER
    global ENCODER
    global REFERENCE_DATA
    global REFERENCE_X
    global REFERENCE_STATS
    global FEATURE_NAMES
    global SHAP_BACKGROUND_X

    try:
        logger.info(
            "Starting model training..."
        )

        df = load_and_prepare_data(
            DATA_PATH
        )

        X, y, scaler, encoder, feature_names = (
            preprocess_data(
                df,
                fit=True,
            )
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------
        logger.info(
            "Training LSTM model..."
        )

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

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------
        logger.info(
            "Training GRU model..."
        )

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

        # ----------------------------------------------------
        # Select best model
        # ----------------------------------------------------
        if (
            lstm_metrics["f1"]
            >= gru_metrics["f1"]
        ):
            best_model = lstm_model
            best_metrics = lstm_metrics
            best_name = "LSTM"

        else:
            best_model = gru_model
            best_metrics = gru_metrics
            best_name = "GRU"

        logger.info(
            "Best model selected: %s",
            best_name,
        )

        # ----------------------------------------------------
        # Save artifacts
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Reload production artifacts
        # ----------------------------------------------------
        MODEL = load_model(
            MODEL_PATH,
            compile=False,
        )

        SCALER = scaler
        ENCODER = encoder

        REFERENCE_DATA = df

        (
            REFERENCE_X,
            _,
            _,
            _,
            FEATURE_NAMES,
        ) = preprocess_data(
            REFERENCE_DATA,
            fit=False,
            scaler=SCALER,
            encoder=ENCODER,
        )

        REFERENCE_X = np.asarray(
            REFERENCE_X,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Refresh statistics
        # ----------------------------------------------------
        REFERENCE_STATS = {}

        if "MonthlyCharges" in REFERENCE_DATA.columns:
            REFERENCE_STATS["MonthlyCharges"] = float(
                REFERENCE_DATA[
                    "MonthlyCharges"
                ].mean()
            )

        if "tenure" in REFERENCE_DATA.columns:
            REFERENCE_STATS["tenure"] = float(
                REFERENCE_DATA[
                    "tenure"
                ].median()
            )

        if "TotalCharges" in REFERENCE_DATA.columns:
            REFERENCE_STATS["TotalCharges"] = float(
                REFERENCE_DATA[
                    "TotalCharges"
                ].mean()
            )

        # ----------------------------------------------------
        # Refresh SHAP background
        # ----------------------------------------------------
        SHAP_BACKGROUND_X = np.asarray(
            REFERENCE_X[
                :min(
                    SHAP_BACKGROUND_SIZE,
                    len(REFERENCE_X),
                )
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Warm newly loaded model
        # ----------------------------------------------------
        _warm_up_model(
            MODEL,
            REFERENCE_X,
        )

        logger.info(
            "Training and production artifact refresh completed."
        )

        return jsonify({
            "status": "Model trained successfully",
            "model": best_name,
            "metrics": best_metrics,
        }), 200

    except Exception:
        logger.exception(
            "Training failed"
        )

        return _error_response(
            "Training failed. Check server logs.",
            500,
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================
if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            "5000",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )

