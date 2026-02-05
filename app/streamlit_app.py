import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📉 Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader(
    "Upload customer data (CSV)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Features & target
    X = df.drop(columns=["churn"])
    y = df["churn"]

    numeric_features = ["age", "monthly_spend", "tenure_months", "support_tickets"]
    categorical_features = ["contract_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression())
        ]
    )

    model.fit(X, y)

    churn_probabilities = model.predict_proba(X)[:, 1]
    df["churn_probability"] = churn_probabilities


    st.subheader("🚨 High-Risk Customers")

    high_risk_threshold = st.slider(
        "Churn probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05
    )

    high_risk_customers = df[df["churn_probability"] >= high_risk_threshold]

    st.metric(
        "High-Risk Customers",
        len(high_risk_customers)
    )

    st.dataframe(
        high_risk_customers
        .sort_values("churn_probability", ascending=False)
    )
