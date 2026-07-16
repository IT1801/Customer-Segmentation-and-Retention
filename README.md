# Customer Segmentation & Retention Analysis

An end-to-end data science solution designed to understand customer behaviour, identify high-value segments, and predict churn risk. This project analyses transactional data to provide actionable intelligence for targeted marketing and retention strategies.

---

## 🎯 Executive Summary

Understanding customer behaviour is critical for sustainable growth. This project implements a robust machine learning pipeline that transforms raw transactional data into strategic insights. 

By clustering customers into distinct segments and predicting their likelihood of churning, the system enables businesses to:
- Identify and reward "Champion" customers.
- Proactively engage "At Risk" segments before they churn.
- Tailor marketing campaigns based on historical purchasing patterns.
- Monitor high-level KPIs across the entire customer base.

---

## 🏗️ System Architecture

The project is built on a modern, scalable data stack designed for reliability and performance:

- **Data Pipeline (ETL)**: A robust Python pipeline that extracts raw data, performs cleaning and imputation, and loads it into a PostgreSQL database.
- **Feature Engineering**: Calculates Recency, Frequency, Monetary (RFM) metrics alongside complex behavioural features for each customer.
- **Machine Learning Models**:
  - **Segmentation**: K-Means clustering algorithm used to group customers into 4 distinct segments (Champions, Loyal Customers, At Risk, Lost/Inactive).
  - **Churn Prediction**: XGBoost classifier trained to predict churn probability, enhanced with SHAP (SHapley Additive exPlanations) for model interpretability.
- **REST API**: A FastAPI application that serves real-time model predictions and customer profiles for downstream consumption.
- **Interactive Dashboard**: A Streamlit application providing business users with an intuitive interface to explore data, view KPI dashboards, and lookup individual customers.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Data Processing** | `pandas`, `numpy`, `pyarrow` |
| **Database** | PostgreSQL 16, `SQLAlchemy` |
| **Machine Learning** | `scikit-learn`, `XGBoost`, `shap` |
| **Model Serving (API)** | `FastAPI`, `uvicorn`, `pydantic` |
| **Frontend Dashboard** | `Streamlit`, `Plotly` |
| **Experiment Tracking**| `MLflow` |
| **Infrastructure** | Docker, `docker-compose` |

---

## 📊 Business Intelligence Dashboard

The analytical dashboard provides stakeholders with immediate access to insights across four key views:

1. **🏠 Overview**: Real-time KPI cards displaying total revenue, active customers, and overall churn rates.
2. **🎯 Segments**: Visualisation of customer clusters through PCA scatter plots and RFM distributions, allowing marketing teams to understand segment characteristics.
3. **⚠️ Churn Analysis**: Risk distribution charts and a ranked list of the top 100 customers at the highest risk of churning, prioritised by historic revenue.
4. **🔍 Customer Lookup**: Deep-dive profiles for individual customers, showcasing their RFM stats, segment label, and real-time churn probability.

---

## 📈 Key Performance Metrics

The models have been thoroughly evaluated and demonstrate strong predictive capabilities:

- **Customers Analysed**: ~5,900 unique profiles mapped from ~1 million historical transactions.
- **Segmentation Quality**: Silhouette score of ~0.42, indicating well-defined and logically separated clusters.
- **Churn Prediction Performance**: ROC-AUC score of ~0.91, proving highly effective at distinguishing between loyal and churning customers.

---

## 🗄️ Data Schema

The PostgreSQL database acts as the central source of truth, housing the following curated tables for analytics and serving:

- `retail.cleaned_transactions`: Granular transaction history post-ETL processing.
- `retail.customer_features`: Engineered feature matrix for machine learning ingestion.
- `retail.segment_results`: Customer IDs mapped to their respective K-Means segment labels.
- `retail.churn_predictions`: Computed churn probabilities and risk tiers per customer.

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
make install-dev

# 2. Start PostgreSQL database
cp .env.example .env
make db-up

# 3. Run the full pipeline (ETL → Features → Train Models)
make all

# 4. Launch the services
make serve       # FastAPI API (http://localhost:8000/docs)
make dashboard   # Streamlit UI (http://localhost:8501)
make mlflow      # MLflow tracking (http://localhost:5000)
```
