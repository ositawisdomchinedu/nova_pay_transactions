import numpy as np
import pandas as pd


def clean_value(val, field):
    if pd.isna(val):
        return np.nan

    s = str(val).strip().lower()
    if s in {"", "nan"}:
        return np.nan

    if field == "channel":
        mapping = {
            "web": "web",
            "weeb": "web",
            "mobile": "mobile",
            "mobille": "mobile",
            "atm": "atm",
            "unknown": "unknown",
        }
        return mapping.get(s, s)

    if field == "kyc_tier":
        mapping = {
            "standard": "standard",
            "standrd": "standard",
            "enhanced": "enhanced",
            "enhancd": "enhanced",
            "low": "low",
            "unknown": "unknown",
        }
        return mapping.get(s, s)

    return s.upper()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset and fix datatypes."""
    work = df.copy()

    country_cols = ["home_country", "source_currency", "dest_currency", "ip_country"]
    for col in country_cols:
        if col in work.columns:
            work[col] = work[col].apply(lambda x: clean_value(x, col))

    cat_cols = ["channel", "kyc_tier"]
    for col in cat_cols:
        if col in work.columns:
            work[col] = work[col].apply(lambda x: clean_value(x, col))

    numeric_cols = [
        "amount_src",
        "amount_usd",
        "fee",
        "exchange_rate_src_to_dest",
        "ip_risk_score",
        "account_age_days",
        "device_trust_score",
        "chargeback_history_count",
        "risk_score_internal",
        "txn_velocity_1h",
        "txn_velocity_24h",
        "corridor_risk",
        "new_device",
        "location_mismatch",
        "is_fraud",
    ]

    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)

    return work