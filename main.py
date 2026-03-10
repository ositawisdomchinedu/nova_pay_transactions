from src.nova.data_loader import load_data
from src.nova.processing import preprocess_data
from src.nova.eda import run_eda
from src.nova.feature_engineering import engineer_features
from src.nova.train import prepare_model_data, scale_data, train_model
from src.nova.evaluate import evaluate_model, get_feature_importance
from src.nova.utils import save_object


DATA_PATH = "data/raw/nova_pay_transactions.csv"


def main():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    df = preprocess_data(df)

     # Run EDA
    run_eda(df)

    # Feature engineering
    df = engineer_features(df)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_model_data(df)

    # Align train/test columns in case dummy columns differ
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Scale
    X_train, X_test, scaler = scale_data(X_train, X_test)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    importance_df = get_feature_importance(model, X_train)

    print("Confusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["classification_report"])
    print(f"ROC-AUC: {results['roc_auc']:.3f}")
    print(f"PR-AUC: {results['pr_auc']:.3f}")

    print("\nTop 10 Important Features:")
    print(importance_df.head(10))

    # Save model artifacts
    save_object(model, "models/fraud_detection_model_01.pkl")
    save_object(scaler, "models/scaler_01.pkl")
    save_object(list(X_train.columns), "models/model_columns.pkl")


if __name__ == "__main__":
    main()