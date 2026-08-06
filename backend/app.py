import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

from utils.preprocessing import preprocess_data, load_and_prepare_data
from utils.shap_explainer import get_shap_values, build_prediction_explanation
from utils.recommendation import get_recommendation
from utils.clv import calculate_clv

from model.lstm_model import build_lstm_model, train_lstm_model, evaluate_model as eval_lstm
from model.gru_model import build_gru_model, train_gru_model, evaluate_model as eval_gru

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'churnprediction.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')

REFERENCE_DATA = load_and_prepare_data(DATA_PATH)
REFERENCE_STATS = {
    "MonthlyCharges": float(REFERENCE_DATA["MonthlyCharges"].mean()),
    "tenure": float(REFERENCE_DATA["tenure"].median()),
    "TotalCharges": float(REFERENCE_DATA["TotalCharges"].mean()),
}


@app.route("/")
def home():
    return "🚀 Customer Churn Backend is Running Successfully"


# =========================
# TRAIN
# =========================
@app.route('/train', methods=['GET'])
def train():
    df = load_and_prepare_data(DATA_PATH)

    X, y, scaler, encoder, _feature_names = preprocess_data(df, fit=True)

    # Train LSTM
    lstm_model = build_lstm_model(X.shape[1])
    train_lstm_model(lstm_model, X, y)
    lstm_metrics = eval_lstm(lstm_model, X, y)

    # Train GRU
    gru_model = build_gru_model(X.shape[1])
    train_gru_model(gru_model, X, y)
    gru_metrics = eval_gru(gru_model, X, y)

    # Select best
    if lstm_metrics['f1'] >= gru_metrics['f1']:
        best_model = lstm_model
        best_metrics = lstm_metrics
    else:
        best_model = gru_model
        best_metrics = gru_metrics

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    return jsonify({
        "status": "Model trained successfully",
        "metrics": best_metrics
    })


# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🔥 Predict API called")

        data = request.json
        print("Input:", data)

        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Model not trained. Call /train first"}), 400

        df = pd.DataFrame([data])

        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)

        X, _, _, _, feature_names = preprocess_data(
            df, fit=False, scaler=scaler, encoder=encoder
        )

        model = load_model(MODEL_PATH)

        prob = float(model.predict(X)[0][0])
        print("Prediction:", prob)

        # Risk level
        if prob < 0.4:
            risk = "Low"
        elif prob < 0.75:
            risk = "Medium"
        else:
            risk = "High"

        # Time to churn
        if risk == "High":
            time_to_churn = "15-30 days"
        elif risk == "Medium":
            time_to_churn = "30-90 days"
        else:
            time_to_churn = "90+ days"

        # CLV
        clv = calculate_clv(
            data.get('tenure', 0),
            data.get('MonthlyCharges', 0)
        )

        # SHAP explanation
        try:
            prediction_explanation = build_prediction_explanation(
                model,
                X,
                feature_names,
                input_data=data,
                reference_stats=REFERENCE_STATS,
            )
            top_reasons = prediction_explanation.get("top_reasons", get_shap_values(model, X, feature_names))
        except (TypeError, ValueError, RuntimeError):
            prediction_explanation = {
                "summary": "Fallback explanation generated because SHAP could not be computed.",
                "positive_factors": [],
                "negative_factors": [],
                "final_reason": "The model produced a valid prediction, but the explanation could not be generated.",
                "top_reasons": ["High MonthlyCharges", "Low tenure"],
            }
            top_reasons = ["High MonthlyCharges", "Low tenure"]

        action = get_recommendation(risk)

        return jsonify({
            "churn_probability": round(prob, 2),
            "risk_level": risk,
            "time_to_churn": time_to_churn,
            "customer_value": clv,
            "recommendation": action,
            "top_reasons": top_reasons,
            "recommended_action": action,
            "prediction_explanation": prediction_explanation
        })

    except (TypeError, ValueError, RuntimeError, OSError) as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# =========================
# EXPLAIN (🔥 NEW FIX)
# =========================
@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        print("🧠 Explain API called")

        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Model not trained"}), 400

        df = pd.DataFrame([data])

        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)

        X, _, _, _, feature_names = preprocess_data(
            df, fit=False, scaler=scaler, encoder=encoder
        )

        model = load_model(MODEL_PATH)

        prediction_explanation = build_prediction_explanation(
            model,
            X,
            feature_names,
            input_data=data,
            reference_stats=REFERENCE_STATS,
        )

        shap_values = get_shap_values(model, X, feature_names, return_full=True)

        return jsonify({
            "message": "Explanation generated",
            "shap_values": shap_values,
            "prediction_explanation": prediction_explanation
        })

    except (TypeError, ValueError, RuntimeError, OSError) as e:
        print("❌ EXPLAIN ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)