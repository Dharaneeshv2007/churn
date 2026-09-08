import logging
from threading import Lock

import numpy as np
import shap

from utils.preprocessing import CAT_COLS, NUM_COLS

logger = logging.getLogger(__name__)

# Small deterministic background keeps explanations practical on Render.
SHAP_BACKGROUND_SIZE = 12
SHAP_NSAMPLES = 32
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
    pass


_EXPLAINER = None
_EXPLAINER_MODEL_ID = None
_EXPLAINER_BACKGROUND = None
_EXPLAINER_LOCK = Lock()


def _model_input(data, model):
    values = np.asarray(data, dtype=np.float32)
    input_shape = getattr(model, "input_shape", None)
    if input_shape and len(input_shape) == 3 and values.ndim == 2:
        return values.reshape(values.shape[0], values.shape[1], 1)
    return values


def _prediction_function(model):
    def predict_fn(data):
        predictions = np.asarray(
            model.predict(_model_input(data, model), verbose=0)
        )
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
        matched = False
        for category_name in CAT_COLS:
            if encoded_name.startswith(f"{category_name}_"):
                groups[category_name].append(index)
                matched = True
                break
        if not matched:
            raise ShapExplanationError(
                f"Unable to map transformed feature '{encoded_name}'"
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
        if float(value).is_integer():
            return int(value)
        return round(float(value), 2)
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
        f"The customer's {LABELS[feature_name]} value ({value}) "
        f"is pushing the prediction toward {direction}."
    )


def _as_float(value):
    return float(np.asarray(value).reshape(-1)[0])


def _get_gradient_explainer(model, background):
    global _EXPLAINER, _EXPLAINER_MODEL_ID, _EXPLAINER_BACKGROUND

    model_id = id(model)
    background = np.asarray(background, dtype=np.float32)

    with _EXPLAINER_LOCK:
        if (
            _EXPLAINER is None
            or _EXPLAINER_MODEL_ID != model_id
            or _EXPLAINER_BACKGROUND is None
            or _EXPLAINER_BACKGROUND.shape != background.shape
        ):
            try:
                # GradientExplainer is substantially cheaper than KernelExplainer
                # for this differentiable Keras LSTM/GRU model.
                _EXPLAINER = shap.GradientExplainer(
                    model,
                    _model_input(background, model),
                )
                _EXPLAINER_MODEL_ID = model_id
                _EXPLAINER_BACKGROUND = background.copy()
                logger.info("Created cached SHAP GradientExplainer")
            except Exception as error:
                _EXPLAINER = None
                raise ShapExplanationError(
                    f"Could not initialize SHAP explainer: {error}"
                ) from error
        return _EXPLAINER


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
        explainer = _get_gradient_explainer(model, background)

        # Gradient SHAP works directly with the 3-D model input.
        raw = explainer.shap_values(
            _model_input(values, model),
            nsamples=SHAP_NSAMPLES,
        )
        shap_row = _one_row(raw)

        # Some SHAP versions return an extra output dimension.
        if shap_row.size != len(feature_names):
            arr = np.asarray(raw, dtype=float)
            while arr.ndim > 3 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            if arr.ndim >= 2:
                shap_row = arr.reshape(arr.shape[0], -1)[0]
            else:
                shap_row = arr.reshape(-1)

        if shap_row.size != len(feature_names):
            raise ShapExplanationError(
                f"SHAP returned {shap_row.size} values for {len(feature_names)} features"
            )

        prediction_probability = _as_float(
            _prediction_function(model)(values)
        )

        # GradientExplainer does not expose a scalar expected_value consistently
        # across SHAP versions. We therefore report the actual model probability
        # and feature contributions, which are the useful parts for the UI.
        grouped_values = {
            feature: float(np.sum(shap_row[indexes]))
            for feature, indexes in groups.items()
        }

        features = []
        for feature_name, shap_value in grouped_values.items():
            formatted_value = _format_value(
                feature_name,
                (input_data or {}).get(feature_name, "Unknown"),
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

        risk_drivers = [x for x in features if x["shap_value"] > NEUTRAL_THRESHOLD]
        protective_factors = [x for x in features if x["shap_value"] < -NEUTRAL_THRESHOLD]
        neutral_factors = [x for x in features if abs(x["shap_value"]) <= NEUTRAL_THRESHOLD]

        top_names = [x["feature"] for x in features[:3]]
        if not top_names:
            summary = "No major prediction factors were identified."
        elif len(top_names) == 1:
            summary = f"The prediction is mainly driven by {top_names[0]}."
        elif len(top_names) == 2:
            summary = f"The prediction is mainly driven by {top_names[0]} and {top_names[1]}."
        else:
            summary = (
                f"The prediction is mainly driven by {top_names[0]}, "
                f"{top_names[1]}, and {top_names[2]}."
            )

        return {
            "method": "SHAP GradientExplainer",
            "prediction_probability": round(prediction_probability, 6),
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
        raise ShapExplanationError(
            "Unable to generate SHAP explanation"
        ) from error


def get_shap_values(model, X, feature_names, return_full=False, background_X=None):
    explanation = build_prediction_explanation(
        model, X, feature_names, background_X=background_X
    )
    if return_full:
        return {x["feature"]: x["importance"] for x in explanation["features"]}
    return [x["feature"] for x in explanation["features"][:3]]
