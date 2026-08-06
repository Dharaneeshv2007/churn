import shap
import numpy as np


def _normalize_value(value):
    if value is None:
        return "Unknown"

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "Unknown"
        if float(value).is_integer():
            return int(value)
        return round(float(value), 2)

    return value


def _format_currency(value):
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "Unknown"


def _human_feature_name(feature_name):
    label_map = {
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

    if feature_name in label_map:
        return label_map[feature_name]

    if feature_name.startswith("InternetService_"):
        return "Internet Service"
    return feature_name.replace("_", " ")


def _split_feature(feature_name):
    if "_" in feature_name:
        base, category = feature_name.split("_", 1)
        return base, category
    return feature_name, None


def _group_shap_values(shap_values, feature_names):
    grouped = {}

    for index, feature_name in enumerate(feature_names):
        base_name, category = _split_feature(feature_name)
        entry = grouped.setdefault(
            base_name,
            {
                "feature_name": _human_feature_name(base_name),
                "shap_value": 0.0,
                "categories": [],
                "raw_feature_names": [],
            },
        )

        value = float(shap_values[index])
        entry["shap_value"] += value
        entry["raw_feature_names"].append(feature_name)
        if category is not None:
            entry["categories"].append(category)

    return grouped


def _get_input_value(input_data, feature_name, category=None):
    if input_data is None:
        if category is not None:
            return category
        return "Unknown"

    if feature_name == "InternetService":
        return input_data.get("InternetService", category or "Unknown")

    return input_data.get(feature_name, category or "Unknown")


def _explanation_for_feature(feature_name, value, reference_stats=None, input_data=None):
    reference_stats = reference_stats or {}

    if feature_name == "tenure":
        tenure_value = float(value or 0)
        if tenure_value >= 36:
            return "Long customer relationships reduce churn because established customers are generally more loyal."
        if tenure_value <= 12:
            return "New customers have higher churn probability because they have not yet built long-term loyalty."
        return "Moderate tenure provides some stability, but the customer relationship is still developing."

    if feature_name == "Contract":
        normalized = str(value or "Unknown").strip().lower()
        if normalized == "month-to-month":
            return "Month-to-month customers are more likely to leave because they are not committed to a long-term plan."
        if normalized == "one year":
            return "One-year contracts provide moderate stability and reduce churn risk compared with flexible plans."
        if normalized == "two year":
            return "Two-year contracts greatly reduce churn because customers are committed for a longer period."
        return "Contract length is influencing the model's retention assessment."

    if feature_name == "MonthlyCharges":
        monthly_value = float(value or 0)
        reference_value = reference_stats.get("MonthlyCharges")
        if reference_value is not None and monthly_value > float(reference_value):
            return "Higher monthly bills can increase churn because customers may see less value for the price they are paying."
        if reference_value is not None and monthly_value < float(reference_value):
            return "Affordable pricing decreases churn because customers are less likely to look for cheaper alternatives."
        if monthly_value >= 80:
            return "Higher monthly bills can increase churn because customers may become price sensitive."
        return "Competitive monthly pricing helps reduce churn pressure."

    if feature_name == "InternetService":
        normalized = str(value or "Unknown").strip().lower()
        if normalized == "fiber optic":
            return "Fiber customers often show slightly higher churn because the segment has historically been more price sensitive."
        if normalized == "dsl":
            return "DSL service is usually associated with a more stable customer base."
        if normalized in {"no", "no internet service"}:
            return "Customers without internet service often churn less because their overall service footprint is simpler."
        return "Internet service type is affecting the customer's churn profile."

    if feature_name == "Partner":
        normalized = str(value or "Unknown").strip().lower()
        if normalized == "yes":
            return "Customers with partners generally stay longer because household stability often reduces switching behavior."
        return "Customers without partners may change providers more easily."

    if feature_name == "Dependents":
        normalized = str(value or "Unknown").strip().lower()
        if normalized == "yes":
            return "Customers with dependents usually have lower churn because they prefer stable services."
        return "Customers without dependents may be more flexible about changing providers."

    if feature_name == "SeniorCitizen":
        citizen_value = int(value or 0)
        if citizen_value == 1:
            return "Senior citizens may require more support, which can increase churn risk if service quality is inconsistent."
        return "Non-senior customers typically show a slightly lower need for retention support."

    if feature_name == "TotalCharges":
        total_value = float(value or 0)
        tenure_value = float((input_data or {}).get("tenure", reference_stats.get("tenure", 0)) or 0)
        if total_value >= 0 and tenure_value >= 36:
            return "High total charges indicate a long customer relationship, which usually signals stronger loyalty."
        return "Total charges reflect the length and depth of the customer relationship."

    if feature_name == "gender":
        return "Gender is being used as a weak supporting signal in the model, but it is usually not a primary churn driver."

    return f"{_human_feature_name(feature_name)} is contributing to the churn decision."


def _impact_label(shap_value, scale):
    if scale <= 0:
        scale = 1.0

    ratio = abs(float(shap_value)) / float(scale)
    if shap_value >= 0:
        if ratio >= 0.6:
            return "Strong Positive"
        if ratio >= 0.25:
            return "Positive"
        return "Small Positive"

    if ratio >= 0.6:
        return "Strong Negative"
    if ratio >= 0.25:
        return "Negative"
    return "Small Negative"


def build_prediction_explanation(model, X, feature_names, input_data=None, reference_stats=None):
    try:
        background = X[:50]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X[:1], nsamples=100)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[0]
        elif shap_values.ndim == 2:
            shap_values = shap_values[0]

        grouped = _group_shap_values(shap_values, feature_names)
        grouped_items = list(grouped.items())
        grouped_items.sort(key=lambda item: abs(float(item[1]["shap_value"])), reverse=True)

        if not grouped_items:
            return {
                "summary": "The model did not produce enough SHAP detail to generate a feature-level explanation.",
                "positive_factors": [],
                "negative_factors": [],
                "final_reason": "The prediction was generated, but no explainability signal was available.",
                "top_reasons": [],
            }

        max_scale = max(abs(float(item[1]["shap_value"])) for item in grouped_items)
        positive_score = 0.0
        negative_score = 0.0
        positive_factors = []
        negative_factors = []

        for feature_key, details in grouped_items[:6]:
            shap_value = float(details["shap_value"])
            base_name = feature_key
            category = details["categories"][0] if details["categories"] else None
            raw_value = _get_input_value(input_data, base_name, category)
            if base_name in {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}:
                raw_value = _normalize_value(raw_value)

            reason = _explanation_for_feature(base_name, raw_value, reference_stats, input_data)
            impact = _impact_label(shap_value, max_scale)
            if base_name == "SeniorCitizen":
                formatted_value = "Yes" if int(raw_value or 0) == 1 else "No"
            elif base_name == "tenure":
                formatted_value = f"{int(raw_value or 0)} months"
            elif base_name == "MonthlyCharges":
                formatted_value = _format_currency(raw_value)
            elif base_name == "TotalCharges":
                formatted_value = _format_currency(raw_value)
            else:
                formatted_value = raw_value
            factor = {
                "feature": details["feature_name"],
                "value": formatted_value,
                "reason": reason,
                "impact": impact,
            }

            if shap_value >= 0:
                positive_score += abs(shap_value)
                positive_factors.append(factor)
            else:
                negative_score += abs(shap_value)
                negative_factors.append(factor)

        summary_parts = []
        if positive_factors:
            summary_parts.append(f"Protective factors are led by {positive_factors[0]['feature']}")
        if negative_factors:
            summary_parts.append(f"Risk factors are led by {negative_factors[0]['feature']}")

        if positive_score >= negative_score:
            final_reason = "The customer is predicted as Low Risk because positive factors outweigh negative factors."
        else:
            final_reason = "The customer is predicted as Higher Risk because negative churn drivers outweigh the protective factors."

        top_reasons = []
        for factor in positive_factors + negative_factors:
            top_reasons.append(f"{factor['feature']}: {factor['value']} ({factor['impact']})")

        return {
            "summary": ". ".join(summary_parts) if summary_parts else "The model found a small number of customer-specific factors driving this prediction.",
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "final_reason": final_reason,
            "top_reasons": top_reasons,
        }

    except Exception:
        return {
            "summary": "The model used fallback explainability logic for this prediction.",
            "positive_factors": [
                {
                    "feature": "MonthlyCharges",
                    "value": "Unknown",
                    "reason": "Higher monthly bills can increase churn.",
                    "impact": "Positive",
                }
            ],
            "negative_factors": [
                {
                    "feature": "Tenure",
                    "value": "Unknown",
                    "reason": "Long customer relationships reduce churn.",
                    "impact": "Negative",
                }
            ],
            "final_reason": "The customer churn explanation was generated from fallback business rules because SHAP could not be computed.",
            "top_reasons": ["MonthlyCharges: Unknown (Positive)", "Tenure: Unknown (Negative)"],
        }


def get_shap_values(model, X, feature_names, return_full=False):
    try:
        background = X[:50]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X[:1], nsamples=100)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values)
        mean_abs = np.abs(shap_values).flatten()
        top_idx = np.argsort(mean_abs)[-3:][::-1]
        top_features = [feature_names[int(i)] for i in top_idx]

        if return_full:
            return {
                feature_names[i]: float(mean_abs[i])
                for i in range(len(feature_names))
            }

        return top_features

    except Exception:
        return ["MonthlyCharges", "tenure", "Contract"]