import numpy as np
import pandas as pd


FEATURE_COLS = [
    "home_country",
    "source_currency",
    "dest_currency",
    "channel",
    "new_device",
    "location_mismatch",
    "ip_risk_score",
    "kyc_tier",
    "account_age_days",
    "device_trust_score",
    "chargeback_history_count",
    "risk_score_internal",
    "txn_velocity_1h",
    "txn_velocity_24h",
    "corridor_risk",
    "amount_src",
    "amount_usd",
    "fee",
    "exchange_rate_src_to_dest",
    "timestamp_missing",
    "hour",
    "dayofweek",
    "is_weekend",
    "month",
    "days_since_start",
    "customer_prev_txn_count",
    "customer_prev_avg_amount_usd",
    "time_since_prev_txn_hours",
    "amount_to_prev_avg_ratio",
    "fee_to_amount_ratio",
    "is_cross_border",
    "currency_mismatch",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "timestamp" not in work.columns:
        raise ValueError("Column 'timestamp' is required for feature engineering.")

    valid_ts = work["timestamp"].dropna().sort_values()
    if len(valid_ts) == 0:
        raise ValueError("Column 'timestamp' contains only missing values.")

    median_ts = valid_ts.iloc[len(valid_ts) // 2]

    work["timestamp_missing"] = work["timestamp"].isna().astype(int)
    work["timestamp_filled"] = work["timestamp"].fillna(median_ts)

    work = work.sort_values(
        ["customer_id", "timestamp_filled", "transaction_id"]
    ).reset_index(drop=True)

    ts = work["timestamp_filled"]
    work["hour"] = ts.dt.hour
    work["dayofweek"] = ts.dt.dayofweek
    work["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    work["month"] = ts.dt.month
    work["days_since_start"] = (ts - ts.min()).dt.total_seconds() / 86400

    return work


def add_customer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    grp = work.groupby("customer_id", sort=False)
    work["customer_prev_txn_count"] = grp.cumcount()

    work["customer_prev_avg_amount_usd"] = grp["amount_usd"].transform(
        lambda s: s.shift().expanding().mean()
    )

    work["time_since_prev_txn_hours"] = (
        grp["timestamp_filled"].diff().dt.total_seconds() / 3600
    )

    prev_avg_safe = work["customer_prev_avg_amount_usd"].replace(0, np.nan)
    amount_safe = work["amount_usd"].replace(0, np.nan)

    work["amount_to_prev_avg_ratio"] = work["amount_usd"] / prev_avg_safe
    work["fee_to_amount_ratio"] = work["fee"] / amount_safe
    work["is_cross_border"] = (work["home_country"] != work["ip_country"]).astype(int)
    work["currency_mismatch"] = (
        work["source_currency"] != work["dest_currency"]
    ).astype(int)

    return work


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = add_time_features(df)
    work = add_customer_behavior_features(work)
    work = work.sort_values("timestamp_filled").reset_index(drop=True)
    return work