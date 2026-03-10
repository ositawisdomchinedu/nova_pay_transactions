import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# Page config

st.set_page_config(page_title="Nova Pay Fraud Detector", page_icon="💳", layout="wide")

st.title("💳 Nova Pay Fraud Detection App")
st.write("Enter transaction details to predict whether a transaction is likely fraudulent.")


# Load saved artifacts

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/fraud_detection_model_01.pkl")
    scaler = joblib.load("models/scaler_01.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

# -----------------------------
# Helper functions
# -----------------------------
def build_features(input_data: dict) -> pd.DataFrame:
    """
    Convert user input into a feature dataframe consistent with model training.
    """
    df = pd.DataFrame([input_data])

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Timestamp handling
    df["timestamp_missing"] = df["timestamp"].isna().astype(int)
    fallback_ts = pd.Timestamp("2025-01-01", tz="UTC")
    df["timestamp_filled"] = df["timestamp"].fillna(fallback_ts)

    ts = df["timestamp_filled"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    df["month"] = ts.dt.month
    df["days_since_start"] = 0  # placeholder for single prediction

    # Customer-level / behavioral placeholders
   
    df["customer_prev_txn_count"] = df["customer_prev_txn_count"].fillna(0)
    df["customer_prev_avg_amount_usd"] = df["customer_prev_avg_amount_usd"].fillna(df["amount_usd"])
    df["time_since_prev_txn_hours"] = df["time_since_prev_txn_hours"].fillna(999)

    # Derived features
    prev_avg = df["customer_prev_avg_amount_usd"].replace(0, 1)
    amount_usd_safe = df["amount_usd"].replace(0, 1)

    df["amount_to_prev_avg_ratio"] = df["amount_usd"] / prev_avg
    df["fee_to_amount_ratio"] = df["fee"] / amount_usd_safe
    df["is_cross_border"] = (df["home_country"] != df["ip_country"]).astype(int)
    df["currency_mismatch"] = (df["source_currency"] != df["dest_currency"]).astype(int)

    feature_cols = [
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

    model_df = df[feature_cols].copy()

    # Fill missing numeric/categorical
    for col in model_df.select_dtypes(include="number").columns:
        model_df[col] = model_df[col].fillna(0)

    for col in model_df.select_dtypes(include="object").columns:
        model_df[col] = model_df[col].fillna("missing")

    # One-hot encode
    X = pd.get_dummies(model_df, drop_first=False)

    # Align to training columns
    X = X.reindex(columns=model_columns, fill_value=0)

    return X


def scale_features(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply scaler only on numeric columns.
    """
    X_scaled = X.copy()
    
    scaler_cols = list(scaler.feature_names_in_)
    X_scaled[scaler_cols] = scaler.transform(X_scaled[scaler_cols])

    return X_scaled

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Transaction Input")

timestamp = st.sidebar.text_input("Timestamp", value="2025-02-15 14:30:00")
home_country = st.sidebar.text_input("Home Country", value="US")
ip_country = st.sidebar.text_input("IP Country", value="US")
source_currency = st.sidebar.text_input("Source Currency", value="USD")
dest_currency = st.sidebar.text_input("Destination Currency", value="USD")

channel = st.sidebar.selectbox("Channel", ["web", "mobile", "atm", "unknown"])
kyc_tier = st.sidebar.selectbox("KYC Tier", ["low", "standard", "enhanced", "unknown"])

amount_src = st.sidebar.number_input("Amount (Source Currency)", min_value=0.0, value=50000.0, step=1000.0)
amount_usd = st.sidebar.number_input("Amount (USD)", min_value=0.0, value=35.0, step=1.0)
fee = st.sidebar.number_input("Fee", min_value=0.0, value=1.0, step=0.1)
exchange_rate_src_to_dest = st.sidebar.number_input("Exchange Rate", min_value=0.0, value=1.0, step=0.01)

new_device = st.sidebar.selectbox("New Device", [0, 1])
location_mismatch = st.sidebar.selectbox("Location Mismatch", [0, 1])

ip_risk_score = st.sidebar.slider("IP Risk Score", 0.0, 100.0, 20.0)
account_age_days = st.sidebar.number_input("Account Age (days)", min_value=0, value=180, step=1)
device_trust_score = st.sidebar.slider("Device Trust Score", 0.0, 100.0, 70.0)
chargeback_history_count = st.sidebar.number_input("Chargeback History Count", min_value=0, value=0, step=1)
risk_score_internal = st.sidebar.slider("Internal Risk Score", 0.0, 100.0, 25.0)
txn_velocity_1h = st.sidebar.number_input("Transaction Velocity (1h)", min_value=0, value=1, step=1)
txn_velocity_24h = st.sidebar.number_input("Transaction Velocity (24h)", min_value=0, value=3, step=1)
corridor_risk = st.sidebar.slider("Corridor Risk", 0.0, 100.0, 20.0)

# Historical placeholders
st.sidebar.markdown("---")
st.sidebar.subheader("Customer History")
customer_prev_txn_count = st.sidebar.number_input("Previous Transaction Count", min_value=0, value=2, step=1)
customer_prev_avg_amount_usd = st.sidebar.number_input("Previous Avg Amount (USD)", min_value=0.0, value=30.0, step=1.0)
time_since_prev_txn_hours = st.sidebar.number_input("Hours Since Previous Transaction", min_value=0.0, value=12.0, step=1.0)

# -----------------------------
# Build input dictionary
# -----------------------------
input_data = {
    "timestamp": timestamp,
    "home_country": home_country.upper().strip(),
    "ip_country": ip_country.upper().strip(),
    "source_currency": source_currency.upper().strip(),
    "dest_currency": dest_currency.upper().strip(),
    "channel": channel.strip().lower(),
    "kyc_tier": kyc_tier.strip().lower(),
    "amount_src": amount_src,
    "amount_usd": amount_usd,
    "fee": fee,
    "exchange_rate_src_to_dest": exchange_rate_src_to_dest,
    "new_device": new_device,
    "location_mismatch": location_mismatch,
    "ip_risk_score": ip_risk_score,
    "account_age_days": account_age_days,
    "device_trust_score": device_trust_score,
    "chargeback_history_count": chargeback_history_count,
    "risk_score_internal": risk_score_internal,
    "txn_velocity_1h": txn_velocity_1h,
    "txn_velocity_24h": txn_velocity_24h,
    "corridor_risk": corridor_risk,
    "customer_prev_txn_count": customer_prev_txn_count,
    "customer_prev_avg_amount_usd": customer_prev_avg_amount_usd,
    "time_since_prev_txn_hours": time_since_prev_txn_hours,
}

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Fraud Risk"):
    try:
        X = build_features(input_data)
        X = X.reindex(columns=model_columns, fill_value=0)
        X_scaled = scale_features(X, scaler)

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected")
        else:
            st.success(f"✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{proba:.2%}")

        st.subheader("Input Summary")
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

    except Exception as e:
        st.exception(e)

# -----------------------------
# Footer note
# -----------------------------
st.markdown("---")
st.caption(
    "Note: customer-history features in this demo use manual inputs/placeholders. "
    "In production, these should be computed from real historical transaction records."
)