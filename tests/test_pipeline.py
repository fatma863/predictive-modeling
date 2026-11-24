import pytest
import numpy as np
from src.data_loader import generate_credit_dataset, get_train_test_split
from src.preprocessing import build_preprocessor, get_feature_names
from src.models import get_model_zoo
from src.evaluate import evaluate_predictions

def test_data_generation():
    df = generate_credit_dataset(n_samples=200, random_state=42)
    assert len(df) == 200
    assert "default" in df.columns
    assert set(df["default"].unique()).issubset({0, 1})

def test_preprocessing_shapes():
    df = generate_credit_dataset(n_samples=100, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    assert X_train_proc.shape[0] == 80
    assert X_test_proc.shape[0] == 20
    assert X_train_proc.shape[1] == len(feature_names)

def test_model_training_and_evaluation():
    df = generate_credit_dataset(n_samples=300, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    model = get_model_zoo()["LogisticRegression"]
    model.fit(X_train_proc, y_train)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    metrics = evaluate_predictions(y_test.values, y_proba)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["financial_loss_estimate_usd"] >= 0

