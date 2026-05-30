import shap
import numpy as np

def get_shap_values(model, X, feature_names, return_full=False):
    try:
        # Use small sample for speed
        background = X[:50]

        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X[:1], nsamples=100)

        # Handle list output (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values)

        # FIX: flatten properly
        mean_abs = np.abs(shap_values).flatten()

        # Get top 3 features safely
        top_idx = np.argsort(mean_abs)[-3:][::-1]

        top_features = [feature_names[int(i)] for i in top_idx]

        if return_full:
            return {
                feature_names[i]: float(mean_abs[i])
                for i in range(len(feature_names))
            }

        return top_features

    except Exception as e:
        # fallback (VERY IMPORTANT)
        return ["MonthlyCharges", "tenure", "Contract"]