from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np

def get_model_zoo() -> Dict[str, Any]:
    """Returns a dictionary of candidate algorithms with baseline production parameters."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1, class_weight="balanced"),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3.5,
            random_state=42,
            eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=3.5,
            random_state=42,
            verbose=-1
        )
    }

def cross_validate_models(X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5) -> Dict[str, Dict[str, float]]:
    """Runs 5-fold Stratified Cross-Validation on all candidate models."""
    models = get_model_zoo()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["roc_auc", "average_precision", "f1", "precision", "recall"]
    results = {}

    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        results[name] = {
            "mean_roc_auc": round(float(np.mean(scores["test_roc_auc"])), 4),
            "std_roc_auc": round(float(np.std(scores["test_roc_auc"])), 4),
            "mean_pr_auc": round(float(np.mean(scores["test_average_precision"])), 4),
            "mean_f1": round(float(np.mean(scores["test_f1"])), 4),
            "mean_recall": round(float(np.mean(scores["test_recall"])), 4)
        }

    return results

