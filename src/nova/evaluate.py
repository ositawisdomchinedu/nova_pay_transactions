import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    results = {
        "confusion_matrix": confusion_matrix(y_test, pred),
        "classification_report": classification_report(y_test, pred, digits=3),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    return results


def get_feature_importance(pipeline, top_n: int = 10) -> pd.DataFrame:
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    return coef_df.head(top_n)