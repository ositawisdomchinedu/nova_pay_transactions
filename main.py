from pathlib import Path

from src.nova.data_loader import load_data
from src.nova.eda import run_eda
from src.nova.evaluate import evaluate_model, get_feature_importance
from src.nova.feature_engineering import engineer_features
from src.nova.processing import preprocess_data
from src.nova.train import prepare_model_data, train_model
from src.nova.utils import save_object


PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "nova_pay_transaction.csv"


def main():
    df = load_data(DATA_PATH)
    print("Loaded:", df.shape)

    df = preprocess_data(df)
    print("Preprocessed:", df.shape)

    run_eda(df)

    df = engineer_features(df)
    print("Feature engineered:", df.shape)

    X_train, X_test, y_train, y_test = prepare_model_data(df)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    pipeline = train_model(X_train, y_train)

    results = evaluate_model(pipeline, X_test, y_test)
    importance_df = get_feature_importance(pipeline, top_n=20)

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nClassification Report:")
    print(results["classification_report"])

    print(f"ROC-AUC: {results['roc_auc']:.3f}")
    print(f"PR-AUC: {results['pr_auc']:.3f}")

    print("\nTop 20 Important Features:")
    print(importance_df)

    save_object(pipeline, PROJECT_ROOT / "models" / "fraud_pipeline.pkl")
    save_object(
        X_train.sample(min(200, len(X_train)), random_state=42),
        PROJECT_ROOT / "models" / "shap_background.pkl",
    )

    importance_df.to_csv(PROJECT_ROOT / "reports" / "feature_importance.csv", index=False)

    print("\nSaved pipeline and reports successfully.")


if __name__ == "__main__":
    main()