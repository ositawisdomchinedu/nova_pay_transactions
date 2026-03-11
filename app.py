from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st


st.set_page_config(
    page_title="Nova Pay Fraud Detector",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Nova Pay Fraud Detection App")
st.write("Enter transaction details to estimate fraud risk and see what influenced the prediction.")


# ---------------------------------------------------
# Load saved artifacts
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_artifacts():
    pipeline = joblib.load(MODELS_DIR / "fraud_pipeline.pkl")
    shap_background = joblib.load(MODELS_DIR / "shap_background.pkl")
    return pipeline, shap_background


pipeline, shap_background = load_artifacts()


# ---------------------------------------------------
# Feature engineering for single prediction
# ---------------------------------------------------
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


def build_features(input_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data]).copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    fallback_ts = pd.Timestamp("2025-01-01", tz="UTC")
    df["timestamp_missing"] = df["timestamp"].isna().astype(int)
    df["timestamp_filled"] = df["timestamp"].fillna(fallback_ts)

    ts = df["timestamp_filled"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    df["month"] = ts.dt.month
    df["days_since_start"] = 0.0

    df["customer_prev_txn_count"] = df["customer_prev_txn_count"].fillna(0)
    df["customer_prev_avg_amount_usd"] = df["customer_prev_avg_amount_usd"].fillna(df["amount_usd"])
    df["time_since_prev_txn_hours"] = df["time_since_prev_txn_hours"].fillna(999)

    prev_avg_safe = df["customer_prev_avg_amount_usd"].replace(0, 1)
    amount_safe = df["amount_usd"].replace(0, 1)

    df["amount_to_prev_avg_ratio"] = df["amount_usd"] / prev_avg_safe
    df["fee_to_amount_ratio"] = df["fee"] / amount_safe
    df["is_cross_border"] = (df["home_country"] != df["ip_country"]).astype(int)
    df["currency_mismatch"] = (df["source_currency"] != df["dest_currency"]).astype(int)

    return df[FEATURE_COLS].copy()


# ---------------------------------------------------
# SHAP explanation
# ---------------------------------------------------
def explain_prediction(model_pipeline, background_df: pd.DataFrame, input_df: pd.DataFrame):
    """
    Explain prediction using the transformed numeric feature space
    after preprocessing.
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    # Transform background and input using the fitted preprocessor
    X_bg = preprocessor.transform(background_df)
    X_input = preprocessor.transform(input_df)

    # Convert sparse matrices to dense if needed
    if hasattr(X_bg, "toarray"):
        X_bg = X_bg.toarray()
    if hasattr(X_input, "toarray"):
        X_input = X_input.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(classifier, X_bg, feature_names=feature_names)
    shap_values = explainer(X_input)

    return shap_values, feature_names, X_input


# ---------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------
st.sidebar.header("Transaction Input")

timestamp = st.sidebar.text_input("Timestamp", value="2025-02-15 14:30:00")

home_country = st.sidebar.text_input("Home Country", value="US")
ip_country = st.sidebar.text_input("IP Country", value="US")
source_currency = st.sidebar.text_input("Source Currency", value="USD")
dest_currency = st.sidebar.text_input("Destination Currency", value="USD")

channel = st.sidebar.selectbox("Channel", ["web", "mobile", "atm", "unknown"])
kyc_tier = st.sidebar.selectbox("KYC Tier", ["low", "standard", "enhanced", "unknown"])

amount_src = st.sidebar.number_input(
    "Amount (Source Currency)", min_value=0.0, value=5000.0, step=100.0
)
amount_usd = st.sidebar.number_input(
    "Amount (USD)", min_value=0.0, value=35.0, step=1.0
)
fee = st.sidebar.number_input(
    "Fee", min_value=0.0, value=1.0, step=0.1
)
exchange_rate_src_to_dest = st.sidebar.number_input(
    "Exchange Rate", min_value=0.0, value=1.0, step=0.01
)

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

st.sidebar.markdown("---")
st.sidebar.subheader("Customer History")
customer_prev_txn_count = st.sidebar.number_input("Previous Transaction Count", min_value=0, value=2, step=1)
customer_prev_avg_amount_usd = st.sidebar.number_input("Previous Avg Amount (USD)", min_value=0.0, value=30.0, step=1.0)
time_since_prev_txn_hours = st.sidebar.number_input("Hours Since Previous Transaction", min_value=0.0, value=12.0, step=1.0)

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5, 0.01)
show_debug = st.sidebar.checkbox("Show debug info", value=False)


# ---------------------------------------------------
# Build input data
# ---------------------------------------------------
input_data = {
    "timestamp": timestamp,
    "home_country": home_country.upper().strip(),
    "ip_country": ip_country.upper().strip(),
    "source_currency": source_currency.upper().strip(),
    "dest_currency": dest_currency.upper().strip(),
    "channel": channel.strip().lower(),
    "kyc_tier": kyc_tier.strip().lower(),
    "amount_src": float(amount_src),
    "amount_usd": float(amount_usd),
    "fee": float(fee),
    "exchange_rate_src_to_dest": float(exchange_rate_src_to_dest),
    "new_device": int(new_device),
    "location_mismatch": int(location_mismatch),
    "ip_risk_score": float(ip_risk_score),
    "account_age_days": float(account_age_days),
    "device_trust_score": float(device_trust_score),
    "chargeback_history_count": float(chargeback_history_count),
    "risk_score_internal": float(risk_score_internal),
    "txn_velocity_1h": float(txn_velocity_1h),
    "txn_velocity_24h": float(txn_velocity_24h),
    "corridor_risk": float(corridor_risk),
    "customer_prev_txn_count": float(customer_prev_txn_count),
    "customer_prev_avg_amount_usd": float(customer_prev_avg_amount_usd),
    "time_since_prev_txn_hours": float(time_since_prev_txn_hours),
}


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict Fraud Risk"):
    try:
        X = build_features(input_data)

        proba = float(pipeline.predict_proba(X)[0][1])
        pred = int(proba >= threshold)

        left, right = st.columns(2)

        with left:
            st.subheader("Prediction Result")
            if pred == 1:
                st.error("⚠️ Fraudulent Transaction Detected")
            else:
                st.success("✅ Legitimate Transaction")

            st.metric("Fraud Probability", f"{proba:.2f}")
            st.metric("Decision Threshold", f"{threshold:.2f}")

        with right:
            st.subheader("Input Summary")
            st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

        #st.subheader("Engineered Features")
        #st.dataframe(X, use_container_width=True)

        if show_debug:
            st.subheader("Debug Info")
            active_features = list(X.columns[(X != 0).any(axis=0)])
            st.write("Active (non-zero) features:", active_features)
            st.write("Number of non-zero features:", int((X != 0).sum().sum()))
            st.write("Raw fraud probability:", proba)

        st.subheader("Why the Model Made This Prediction")

        shap_values, shap_feature_names, X_input_transformed = explain_prediction(
                                                    pipeline, shap_background, X
                    )

        try:
            plt.figure(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], max_display=12, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception:
            st.info("Could not render SHAP waterfall plot here. Showing contribution table instead.")

        contrib_df = pd.DataFrame(
            {
                 "feature": shap_feature_names,
                 "value": X_input_transformed[0],
                 "shap_value": shap_values.values[0],
         }
        )
        
        contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
        contrib_df = contrib_df.sort_values("abs_shap", ascending=False)

        st.subheader("Top Contributing Features")
        st.dataframe(
            contrib_df[["feature", "value", "shap_value"]].head(15),
            use_container_width=True,
        )

    except Exception as e:
        st.exception(e)


st.markdown("---")
st.caption(
    "This demo uses manually entered customer-history features. "
    "In production, these should come from real historical transaction records."
)