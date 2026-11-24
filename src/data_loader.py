import numpy as np
import pandas as pd
from typing import Tuple

def generate_credit_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic credit default risk tabular dataset
    with non-linear risk interactions, skewed income distributions, and class imbalance.
    """
    np.random.seed(random_state)
    
    # 1. Age: 21 to 75
    age = np.random.randint(21, 75, size=n_samples)
    
    # 2. Annual Income: Log-normal distribution (skewed)
    annual_income = np.round(np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples), 2)
    annual_income = np.clip(annual_income, 15000, 300000)
    
    # 3. Credit Utilization Ratio (0.0 to 1.2)
    credit_utilization = np.round(np.random.beta(a=2, b=5, size=n_samples) * 1.2, 4)
    
    # 4. Debt-to-Income (DTI) Ratio (0.05 to 0.65)
    dti_ratio = np.round(np.random.beta(a=2, b=4, size=n_samples) * 0.65, 4)
    
    # 5. Number of Delinquencies in Last 2 Years (0 to 6)
    delinquencies = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.72, 0.15, 0.07, 0.03, 0.02, 0.01], size=n_samples)
    
    # 6. Loan Amount requested
    loan_amount = np.round(np.random.uniform(2000, 45000, size=n_samples), 2)
    
    # 7. Employment Duration in Years (0 to 30)
    employment_years = np.clip(np.random.exponential(scale=5.0, size=n_samples), 0, 35).astype(int)
    
    # 8. Home Ownership Type (Categorical)
    home_ownership = np.random.choice(["RENT", "MORTGAGE", "OWN"], p=[0.45, 0.45, 0.10], size=n_samples)
    
    # Calculate Latent Default Log-Odds
    # Base risk + high utilization + high DTI + delinquencies - age - income
    log_odds = (
        -3.2
        + 2.8 * credit_utilization
        + 2.5 * dti_ratio
        + 0.7 * delinquencies
        - 0.02 * (age - 21)
        - 0.000015 * (annual_income - 30000)
        - 0.05 * employment_years
        + (0.35 if (home_ownership == "RENT").all() else 0.0)
    )
    
    default_prob = 1.0 / (1.0 + np.exp(-log_odds))
    default_label = (np.random.rand(n_samples) < default_prob).astype(int)
    
    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "credit_utilization": credit_utilization,
        "dti_ratio": dti_ratio,
        "delinquencies_2yrs": delinquencies,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "default": default_label
    })
    
    return df

def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["default"])
    y = df["default"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

