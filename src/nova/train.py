import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

FEATURE_COLS = [
    "home_country", "source_currency", "dest_currency", "channel",
    "new_device", "location_mismatch", "ip_risk_score", "kyc_tier",
    "account_age_days", "device_trust_score", "chargeback_history_count",
    "risk_score_internal", "txn_velocity_1h", "txn_velocity_24h",
    "corridor_risk", "amount_src", "amount_usd", "fee",
    "exchange_rate_src_to_dest", "timestamp_missing", "hour",
    "dayofweek", "is_weekend", "month", "days_since_start",
    "customer_prev_txn_count", "customer_prev_avg_amount_usd",
    "time_since_prev_txn_hours", "amount_to_prev_avg_ratio",
    "fee_to_amount_ratio", "is_cross_border", "currency_mismatch"
]


def prepare_model_data(df: pd.DataFrame):
    model_df = df[FEATURE_COLS + ["is_fraud"]].copy()

    X = model_df.drop(columns="is_fraud")
    y = model_df["is_fraud"]

    split_idx = int(len(model_df) * 0.8)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def build_pipeline(X_train: pd.DataFrame):
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    return pipeline


def train_model(X_train, y_train):
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    return pipeline