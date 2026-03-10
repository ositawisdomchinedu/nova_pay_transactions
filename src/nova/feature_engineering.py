import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    median_ts = work["timestamp"].dropna().sort_values().iloc[
        len(work["timestamp"].dropna()) // 2
    ]

    work["timestamp_missing"] = work["timestamp"].isna().astype(int)
    work["timestamp_filled"] = work["timestamp"].fillna(median_ts)

    work = work.sort_values(["customer_id", "timestamp_filled", "transaction_id"]).reset_index(drop=True)

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
    work["time_since_prev_txn_hours"] = grp["timestamp_filled"].diff().dt.total_seconds() / 3600

    work["amount_to_prev_avg_ratio"] = work["amount_usd"] / work["customer_prev_avg_amount_usd"]
    work["fee_to_amount_ratio"] = work["fee"] / work["amount_usd"]
    work["is_cross_border"] = (work["home_country"] != work["ip_country"]).astype(int)
    work["currency_mismatch"] = (work["source_currency"] != work["dest_currency"]).astype(int)

    return work


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = add_time_features(df)
    work = add_customer_behavior_features(work)
    work = work.sort_values("timestamp_filled").reset_index(drop=True)
    return work