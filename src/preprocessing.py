import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from typing import Tuple

NUMERICAL_FEATURES = [
    "age",
    "annual_income",
    "credit_utilization",
    "dti_ratio",
    "delinquencies_2yrs",
    "loan_amount",
    "employment_years"
]

CATEGORICAL_FEATURES = [
    "home_ownership"
]

def build_preprocessor() -> ColumnTransformer:
    """Creates a production preprocessing pipeline for tabular credit risk data."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extracts transformed feature names after fitting."""
    feature_names = list(NUMERICAL_FEATURES)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_cols)
    return feature_names

