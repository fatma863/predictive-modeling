import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix
)
from typing import Dict, Any

def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluates probabilistic model performance with both statistical
    and domain-specific credit cost metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    
    # Financial Cost Matrix Assumption:
    # False Negative (missed default): Loan lost = -$10,000
    # False Positive (rejecting good borrower): Opportunity loss = -$500
    tn, fp, fn, tp = cm.ravel()
    estimated_cost = (fn * 10000) + (fp * 500)

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "financial_loss_estimate_usd": int(estimated_cost)
    }

