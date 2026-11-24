import json
from src.data_loader import generate_credit_dataset, get_train_test_split
from src.preprocessing import build_preprocessor, get_feature_names
from src.models import cross_validate_models, get_model_zoo
from src.evaluate import evaluate_predictions
from src.explainability import compute_shap_explanations

def main():
    print("=" * 60)
    print("1. Generating Synthetic Tabular Credit Risk Dataset...")
    df = generate_credit_dataset(n_samples=5000, random_state=42)
    print(f"Dataset shape: {df.shape} | Default rate: {df['default'].mean():.2%}")

    print("\n2. Splitting and Preprocessing Features...")
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"Engineered Feature count: {len(feature_names)}")

    print("\n3. Running 5-Fold Stratified Cross-Validation on Model Zoo...")
    cv_results = cross_validate_models(X_train_proc, y_train.values)
    for model_name, metrics in cv_results.items():
        print(f"  [{model_name}] ROC-AUC: {metrics['mean_roc_auc']} (+/- {metrics['std_roc_auc']}) | PR-AUC: {metrics['mean_pr_auc']}")

    print("\n4. Training Final Champion Model (XGBoost)...")
    champion = get_model_zoo()["XGBoost"]
    champion.fit(X_train_proc, y_train)

    y_proba = champion.predict_proba(X_test_proc)[:, 1]
    eval_metrics = evaluate_predictions(y_test.values, y_proba)
    print("Test Set Performance:")
    print(json.dumps(eval_metrics, indent=2))

    print("\n5. Computing SHAP Explainability (XAI)...")
    shap_results = compute_shap_explanations(champion, X_test_proc[:200], feature_names)
    print("Top 5 Drivers of Default Risk (Mean Absolute SHAP):")
    print(shap_results["global_importance_df"].head(5).to_string(index=False))

    print("\n[SUCCESS] Machine Learning & Explainable AI Pipeline completed!")

if __name__ == "__main__":
    main()

