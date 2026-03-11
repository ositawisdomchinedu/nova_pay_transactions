# Nova Pay Fraud Detection Prototype

## Overview

This project implements a small **end-to-end fraud detection prototype** using the Nova Pay transactions dataset.

The goal is to demonstrate a complete **machine learning pipeline** including:

- Data preparation and cleaning
- Exploratory data analysis (EDA)
- Feature engineering
- Fraud detection model training
- Model evaluation
- Model interpretability
- An interactive prediction interface using **Streamlit**

The system predicts whether a transaction is **fraudulent or legitimate** and provides explanations for predictions.

---

# Project Structure

nova_pay_transactions/
│
├── app.py
├── main.py
├── requirements.txt
│
├── data/
│ └── nova_pay_transaction.csv
│
├── models/
│ ├── fraud_pipeline.pkl
│ └── shap_background.pkl
│
├── reports/
│ ├── eda_report.md
│ ├── feature_importance.csv
│ └── figures/
│
└── src/
└── nova/
├── data_loader.py
├── preprocessing.py
├── feature_engineering.py
├── eda.py
├── training.py
├── evaluate.py
└── utils.py


---

# Key Components

## Data Loader

Loads the raw dataset from the `data` directory.

File:

````markdown
src/nova/data_loader.py


# Data Preprocessing

Handles:

- Missing values

- Cleaning categorical fields

- Numeric conversion

- Timestamp parsing

## File: src/nova/preprocessing.py


# Feature Engineering

## Adds useful behavioral and time-based features such as:

- Hour of transaction

- Day of week

- Weekend indicator

- Customer transaction history

- Transaction velocity

- Cross-border indicator

- Currency mismatch

- Amount behavior ratios

## File:

src/nova/feature_engineering.py


# Exploratory Data Analysis (EDA)

## Generates insights and visualizations including:

- Fraud vs non-fraud distribution

- Transaction amount distribution

- Fraud rate by channel

- Fraud rate by KYC tier

- Transaction velocity patterns

- Account age distribution

## All plots are saved to: src/nova/eda.py

# Model Training

## The fraud model uses a Scikit-Learn Pipeline consisting of:

- Missing value imputation

- Feature scaling

- One-hot encoding for categorical features

- Logistic Regression classifier

- Class imbalance is handled using: class_weight="balanced"



# Model Evaluation

- Evaluation metrics include:

- Confusion Matrix

- Precision / Recall / F1 Score

- ROC-AUC

- Precision-Recall AUC

# Feature Importance

Feature importance is computed using the **logistic regression coefficients**, which provide a **global interpretation** of the model.

## Saved to: reports/feature_importance.csv


# Interactive Fraud Prediction App

The Streamlit application allows users to:

- Enter transaction details

- Predict fraud probability

- View prediction results

- See model explanations using SHAP


# Installation
## 1. Clone the Repository

git clone <https://github.com/ositawisdomchinedu/nova_pay_transactions>
cd nova_pay_transactions

## 2. Create a Virtual Environment

python -m venv env
env\Scripts\activate

## 3. Install Dependencies

pip install -r requirements.txt

# Required Python Packages

- pandas
- numpy
- matplotlib
- scikit-learn
- joblib
- streamlit
- shap

# Running the Pipeline

## Train the model and generate reports: python main.py

This will:

- Load the dataset

- Clean and preprocess the data

- Perform exploratory data analysis

- Engineer new features

- Train the fraud detection model

- Evaluate model performance

- Save the trained model pipeline

Generated outputs:

- models/fraud_pipeline.pkl
- models/shap_background.pkl
- reports/feature_importance.csv
- reports/eda_report.md
- reports/figures/*.png

# Running the Fraud Detection App

Launch the Streamlit interface:

streamlit run app.py

# Using the App

Enter transaction information in the sidebar.

- Click Predict Fraud Risk.

The app will display:

- Fraud prediction result

- Fraud probability score

- Transaction summary

- Model explanation for the prediction

# Model Interpretability

Two levels of interpretability are included.

## Global Explanation

Feature importance derived from logistic regression coefficients explains which variables influence fraud predictions across the entire dataset.

## Local Explanation

SHAP explanations show which features influenced a specific transaction prediction.

These complementary explanations help improve transparency and trust in the model.

## Author

**Wisdom Chinedu Osita**
Data Scientist / AI Engineer