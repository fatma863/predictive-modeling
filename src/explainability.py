import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List

def compute_shap_explanations(model: Any, X_sample: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Computes SHAP (SHapley Additive exPlanations) values using TreeExplainer
    to uncover global feature attribution and individual prediction breakdowns.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # In binary classification, some versions return list of arrays, or single array
    if isinstance(shap_values, list):
        vals = shap_values[1] # positive class
    else:
        vals = shap_values

    # Mean absolute SHAP value per feature
    global_importance = np.mean(np.abs(vals), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.round(global_importance, 4)
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Local explanation for the highest-risk individual
    highest_risk_idx = int(np.argmax(vals.sum(axis=1)))
    local_breakdown = {
        "sample_index": highest_risk_idx,
        "base_value": float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value),
        "feature_attributions": {
            feat: round(float(val), 4) for feat, val in zip(feature_names, vals[highest_risk_idx])
        }
    }

    return {
        "global_importance_df": importance_df,
        "local_highest_risk_sample": local_breakdown
    }

