import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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

    for col in model_df.select_dtypes(include="number").columns:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    for col in model_df.select_dtypes(include="object").columns:
        model_df[col] = model_df[col].fillna("missing")

    X = pd.get_dummies(model_df.drop(columns="is_fraud"), drop_first=False)
    y = model_df["is_fraud"]

    split_idx = int(len(model_df) * 0.8)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    num_cols = X_train.select_dtypes(include="number").columns

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, scaler


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model