import pandas as pd
import matplotlib.pyplot as plt
import os

REPORT_PATH = "reports/figures"


def save_plot(filename):
    os.makedirs(REPORT_PATH, exist_ok=True)
    plt.savefig(f"{REPORT_PATH}/{filename}", bbox_inches="tight")
    plt.close()


def dataset_summary(df: pd.DataFrame):
    """Basic dataset overview"""
    
    print("Dataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isna().sum().sort_values(ascending=False))

    fraud_rate = df["is_fraud"].mean()
    print(f"\nFraud Rate: {fraud_rate:.2%}")


def fraud_distribution(df: pd.DataFrame):
    """Class distribution"""
    
    plt.figure(figsize=(6,4))
    df["is_fraud"].value_counts().sort_index().plot(kind="bar")
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xlabel("is_fraud")
    plt.ylabel("Count")
    plt.tight_layout()
    
    save_plot("fraud_distribution.png")


def amount_distribution(df: pd.DataFrame):
    """Transaction amount distribution"""

    plt.figure(figsize=(7,4))

    df.loc[df["is_fraud"] == 0, "amount_usd"].clip(upper=1000).hist(
        bins=40, alpha=0.7, label="Non Fraud"
    )

    df.loc[df["is_fraud"] == 1, "amount_usd"].clip(upper=1000).hist(
        bins=40, alpha=0.7, label="Fraud"
    )

    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount USD")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    
    save_plot("transaction_amount_distribution.png")


def fraud_by_channel(df: pd.DataFrame):
    """Fraud rate by transaction channel"""

    channel_rate = df.groupby("channel")["is_fraud"].mean().sort_values(ascending=False)

    print("\nFraud Rate by Channel")
    print(channel_rate)

    plt.figure(figsize=(6,4))
    channel_rate.plot(kind="bar")
    plt.title("Fraud Rate by Channel")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    
    save_plot("fraud_by_channel.png")


def fraud_by_kyc(df: pd.DataFrame):
    """Fraud rate by KYC level"""

    kyc_rate = df.groupby("kyc_tier")["is_fraud"].mean().sort_values(ascending=False)

    print("\nFraud Rate by KYC Tier")
    print(kyc_rate)

    plt.figure(figsize=(6,4))
    kyc_rate.plot(kind="bar")
    plt.title("Fraud Rate by KYC Tier")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    
    save_plot("fraud_by_kyc.png")


def velocity_analysis(df):
    """Analyze fraud rate by transaction velocity"""

    velocity = df.groupby("txn_velocity_1h")["is_fraud"].mean()

    plt.figure(figsize=(7,4))
    velocity.plot()
    plt.title("Fraud Rate vs Transaction Velocity (1h)")
    plt.xlabel("Transactions in Last Hour")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    
    save_plot("fraud_vs_velocity.png")


def account_age_analysis(df):
    """Fraud vs account age"""

    plt.figure(figsize=(7,4))

    df[df["is_fraud"] == 0]["account_age_days"].clip(upper=200).hist(
        bins=40, alpha=0.7, label="Non Fraud"
    )

    df[df["is_fraud"] == 1]["account_age_days"].clip(upper=200).hist(
        bins=40, alpha=0.7, label="Fraud"
    )

    plt.title("Account Age Distribution")
    plt.xlabel("Account Age (days)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    
    save_plot("fraud_vs_account_age.png")


def run_eda(df: pd.DataFrame):
    """Run full exploratory analysis"""

    dataset_summary(df)

    fraud_distribution(df)

    amount_distribution(df)

    fraud_by_channel(df)

    fraud_by_kyc(df)

    velocity_analysis(df)

    account_age_analysis(df)