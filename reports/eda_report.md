# Exploratory Data Analysis (EDA)

## Dataset Overview

The dataset contains **10,200 financial transactions** with a binary target variable `is_fraud` indicating whether a transaction was fraudulent.

Fraudulent transactions represent **a small minority of the dataset**, which is typical in fraud detection problems where legitimate activity dominates.

This class imbalance highlights the importance of using evaluation metrics such as **precision, recall, ROC-AUC, and PR-AUC**, rather than relying solely on accuracy.

---

# Key Distribution Patterns

## Fraud vs Non-Fraud Transactions

The distribution of the target variable shows a **strong imbalance**, with the vast majority of transactions being legitimate.

**Key Insight**

Fraud cases are rare events. A model that simply predicts “non-fraud” for all transactions could achieve high accuracy but would fail to detect fraud effectively.

**Implication**

The model must prioritize **fraud detection sensitivity** while maintaining acceptable false positive rates.

---

## Transaction Amount Distribution

Transaction amounts exhibit a **right-skewed distribution**, where most transactions occur at lower values and only a small number reach high values.

**Key Observations**

- Most transactions fall between **$50 and $200**.
- A long tail of larger transactions exists, reaching values near **$1000**.
- Fraudulent transactions appear across the range but become relatively more visible in higher-value ranges.

**Implication**

Transaction amount alone is not sufficient to detect fraud, but it may provide useful signals when combined with behavioral features such as transaction velocity or account history.

---

# Behavioral Risk Indicators

## Fraud Rate vs Transaction Velocity

Transaction velocity measures the number of transactions performed by a customer within the last hour.

**Key Observations**

- Fraud rates increase as transaction velocity rises.
- Customers performing **multiple transactions within a short time window** are more likely to generate fraudulent activity.

**Interpretation**

High transaction velocity may indicate:

- Automated fraud attempts
- Account takeover behavior
- Bot-driven payment activity

**Implication**

Transaction velocity features are strong candidates for **behavioral fraud detection models**.

---

# Customer Risk Characteristics

## Account Age Distribution

Account age represents the number of days since the customer account was created.

**Key Observations**

- Most transactions originate from accounts that are approximately **200 days old**, suggesting a mature customer base.
- Fraudulent activity is relatively more common among **newer accounts**.

**Interpretation**

Fraudsters often create new accounts to perform short-lived fraudulent activity before detection mechanisms trigger.

**Implication**

Account age is an important **risk indicator** when combined with other behavioral signals.

---

# Identity Verification and Fraud Risk

## Fraud Rate by KYC Tier

KYC (Know Your Customer) tier reflects the level of identity verification applied to a user.

**Key Observations**

- **Low KYC tier accounts exhibit the highest fraud rate.**
- **Enhanced KYC accounts show the lowest fraud rate.**
- Accounts with **unknown verification status also show elevated fraud rates.**

**Interpretation**

Stronger identity verification appears to reduce fraud risk, suggesting that stricter onboarding processes help mitigate fraudulent activity.

**Implication**

KYC level is a valuable **predictive feature for fraud detection models**.

---

# Transaction Channel Risk

## Fraud Rate by Channel

Transactions occur across multiple channels:

- Web
- Mobile
- ATM
- Unknown

**Key Observations**

- The **unknown and web channels exhibit the highest fraud rates**.
- Mobile and ATM channels appear comparatively safer.

**Interpretation**

Fraudsters may prefer channels where authentication or monitoring mechanisms are weaker or where automated activity is easier to execute.

**Implication**

Transaction channel should be incorporated as a **categorical risk feature** in the model.

---

# Model Feature Importance Insights

Feature importance from the logistic regression model highlights several strong predictors of fraud risk.

Top contributing features include:

### KYC Tier
Enhanced KYC significantly decreases fraud probability, while lower verification tiers increase fraud risk.

### Currency and Transaction Corridor
Certain currency corridors show stronger associations with fraud, likely reflecting higher-risk payment routes.

### Account Age
Older accounts tend to exhibit lower fraud risk.

### Behavioral Signals
Behavioral features such as:

- transaction velocity
- chargeback history
- corridor risk

play an important role in distinguishing fraudulent transactions.

---

# Summary of Key Insights

The exploratory analysis reveals several meaningful fraud risk patterns:

1. **Fraud events are rare**, creating a strong class imbalance.
2. **High transaction velocity increases fraud risk.**
3. **Lower KYC verification levels correlate with higher fraud activity.**
4. **New accounts are more likely to be associated with fraud.**
5. **Certain transaction channels and currency corridors show elevated fraud rates.**
6. **Behavioral features provide valuable signals beyond raw transaction values.**

These insights support the design of a fraud detection model that leverages **transaction behavior, account characteristics, and risk indicators** rather than relying on single variables.