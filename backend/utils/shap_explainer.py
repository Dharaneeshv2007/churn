import logging

import numpy as np
import shap

from utils.preprocessing import CAT_COLS, NUM_COLS


logger = logging.getLogger(__name__)
SHAP_BACKGROUND_SIZE = 100
SHAP_NSAMPLES = 200
NEUTRAL_THRESHOLD = 0.005

LABELS = {
    "SeniorCitizen": "Senior Citizen",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
    "InternetService": "Internet Service",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "Contract": "Contract",
    "tenure": "Tenure",
    "gender": "Gender",
}


class ShapExplanationError(RuntimeError):
    """Raised when a real SHAP explanation cannot be produced."""


def _model_input(data, model):
    values = np.asarray(data, dtype=np.float32)
    input_shape = getattr(model, "input_shape", None)
    if input_shape and len(input_shape) == 3 and values.ndim == 2:
        return values.reshape(values.shape[0], values.shape[1], 1)
    return values


def _prediction_function(model):
    def predict_fn(data):
        predictions = np.asarray(model.predict(_model_input(data, model), verbose=0))
        return predictions.reshape(-1)

    return predict_fn


def _one_row(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.asarray([float(values)])
    if values.ndim == 1:
        return values
    return values[0].reshape(-1)


def _raw_feature_groups(feature_names):
    groups = {name: [] for name in NUM_COLS + CAT_COLS}
    for index, encoded_name in enumerate(feature_names):
        if encoded_name in NUM_COLS:
            groups[encoded_name].append(index)
            continue
        for category_name in CAT_COLS:
            if encoded_name.startswith(f"{category_name}_"):
                groups[category_name].append(index)
                break
        else:
            raise ShapExplanationError(
                f"Unable to map transformed feature '{encoded_name}' to an input feature"
            )
    return groups


def _normalize_value(value):
    if value is None:
        return "Unknown"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "Unknown"
        return int(value) if float(value).is_integer() else round(float(value), 2)
    return value


def _format_value(feature_name, value):
    value = _normalize_value(value)
    if feature_name == "SeniorCitizen":
        return "Yes" if str(value).strip().lower() in {"1", "yes", "true"} else "No"
    if feature_name == "tenure":
        return f"{value} months"
    if feature_name in {"MonthlyCharges", "TotalCharges"}:
        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return "Unknown"
    return value


def _impact(shap_value):
    if abs(shap_value) <= NEUTRAL_THRESHOLD:
        return "Minimal influence"
    return "Increased churn risk" if shap_value > 0 else "Reduced churn risk"


def _feature_reason(feature_name, value, impact):
    direction = {
        "Increased churn risk": "higher churn risk",
        "Reduced churn risk": "lower churn risk",
        "Minimal influence": "minimal influence on churn risk",
    }[impact]
    return (
        f"The customer's {LABELS[feature_name]} value ({value}) is pushing "
        f"the prediction toward {direction}."
    )


def _as_float(value):
    return float(np.asarray(value).reshape(-1)[0])


def build_prediction_explanation(
    model,
    X,
    feature_names,
    input_data=None,
    reference_stats=None,
    background_X=None,
):
    del reference_stats
    try:
        values = np.asarray(X, dtype=np.float32)
        background = np.asarray(background_X, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != 1:
            raise ShapExplanationError("SHAP expects exactly one transformed customer row")
        if background.ndim != 2 or background.shape[1] != values.shape[1]:
            raise ShapExplanationError("SHAP background does not match model feature dimensions")

        groups = _raw_feature_groups(feature_names)
        predict_fn = _prediction_function(model)
        explainer = shap.KernelExplainer(predict_fn, background)
        raw_shap_values = explainer.shap_values(values, nsamples=SHAP_NSAMPLES)
        shap_row = _one_row(raw_shap_values)
        if shap_row.size != len(feature_names):
            raise ShapExplanationError("SHAP returned an unexpected feature dimension")

        base_value = _as_float(explainer.expected_value)
        prediction_probability = _as_float(predict_fn(values))
        grouped_values = {
            feature: float(np.sum(shap_row[indexes]))
            for feature, indexes in groups.items()
        }
        reconstructed = base_value + sum(grouped_values.values())
        additivity_error = abs(reconstructed - prediction_probability)
        if additivity_error > 0.05:
            logger.warning(
                "SHAP additivity check exceeded tolerance: error=%f prediction=%f reconstructed=%f",
                additivity_error,
                prediction_probability,
                reconstructed,
            )

        features = []
        for feature_name, shap_value in grouped_values.items():
            formatted_value = _format_value(
                feature_name, (input_data or {}).get(feature_name, "Unknown")
            )
            impact = _impact(shap_value)
            features.append({
                "feature": LABELS[feature_name],
                "value": formatted_value,
                "shap_value": round(shap_value, 6),
                "contribution_percentage_points": round(shap_value * 100, 4),
                "importance": round(abs(shap_value), 6),
                "impact": impact,
                "reason": _feature_reason(feature_name, formatted_value, impact),
            })
        features.sort(key=lambda item: item["importance"], reverse=True)

        risk_drivers = [item for item in features if item["shap_value"] > NEUTRAL_THRESHOLD]
        protective_factors = [item for item in features if item["shap_value"] < -NEUTRAL_THRESHOLD]
        neutral_factors = [item for item in features if abs(item["shap_value"]) <= NEUTRAL_THRESHOLD]
        top_names = [item["feature"] for item in features[:3]]
        summary = "The prediction is mainly driven by " + ", ".join(top_names)
        if len(top_names) > 1:
            summary = "The prediction is mainly driven by " + ", ".join(top_names[:-1]) + f", and {top_names[-1]}"
        summary += "."
        return {
            "base_value": round(base_value, 6),
            "prediction_probability": round(prediction_probability, 6),
            "reconstructed_probability": round(reconstructed, 6),
            "additivity_error": round(additivity_error, 6),
            "features": features,
            "risk_drivers": risk_drivers,
            "protective_factors": protective_factors,
            "neutral_factors": neutral_factors,
            "summary": summary,
            "positive_factors": risk_drivers,
            "negative_factors": protective_factors,
            "final_reason": summary,
            "top_reasons": [
                f"{item['feature']}: {item['value']} ({item['impact']})"
                for item in features[:3]
            ],
        }
    except ShapExplanationError:
        raise
    except Exception as error:
        logger.exception("SHAP explanation failed")
        raise ShapExplanationError("Unable to generate SHAP explanation") from error


def get_shap_values(model, X, feature_names, return_full=False, background_X=None):
    explanation = build_prediction_explanation(
        model, X, feature_names, background_X=background_X
    )
    if return_full:
        return {item["feature"]: item["importance"] for item in explanation["features"]}
    return [item["feature"] for item in explanation["features"][:3]]