# Customer Segmentation & Retention Analysis

End-to-end data science pipeline for customer segmentation and retention analysis on the [Online Retail II UCI dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) (~1M transactions, 2009–2011).

---

## What this project does

| Component | What it builds |
|---|---|
| **ETL pipeline** | Extracts raw `.xlsx` → cleans → loads into PostgreSQL |
| **Feature engineering** | RFM, behavioural, and cohort features per customer |
| **Segmentation** | K-Means clustering → 4 customer segments |
| **Churn prediction** | XGBoost classifier with SHAP explainability |
| **CLV modelling** | BG/NBD + Gamma-Gamma → 12-month CLV per customer |
| **Market basket** | FP-Growth association rules for product recommendations |
| **FastAPI** | REST API serving all model predictions |
| **Streamlit** | Interactive dashboard with 6 pages |
| **MLflow** | Experiment tracking across all model runs |

---

## Project structure

```
customer-segmentation/
├── config/
│   └── config.yaml               ← central config (DB, paths, model params)
├── src/csr/
│   ├── constants/__init__.py     ← all hardcoded values
│   ├── config/configuration.py  ← typed config manager
│   ├── etl/
│   │   ├── extract.py
│   │   ├── transform.py
│   │   ├── load.py
│   │   └── pipeline.py
│   ├── features/
│   │   ├── rfm.py
│   │   ├── behavioural.py
│   │   ├── cohort.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── segmentation.py
│   │   ├── churn.py
│   │   ├── clv.py
│   │   └── market_basket.py
│   └── api/
│       ├── main.py
│       ├── router.py
│       └── schemas.py
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 01_segments.py
│       ├── 02_churn.py
│       ├── 03_clv.py
│       ├── 04_cohort.py
│       ├── 05_market_basket.py
│       └── 06_customer_lookup.py
├── research/                     ← exploratory notebooks
├── models/artifacts/             ← saved .joblib model files
├── data/
│   ├── raw/                      ← original .xlsx (gitignored)
│   ├── interim/                  ← cleaned parquet (research only)
│   └── processed/                ← feature parquet (research only)
├── tests/
├── scripts/
│   └── init.sql                  ← Postgres schema initialisation
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── .env.example
└── README.md
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Data | pandas, numpy, pyarrow, openpyxl |
| Database | PostgreSQL 16, SQLAlchemy, psycopg2-binary |
| ML | scikit-learn, XGBoost, lifetimes, mlxtend, shap, lifelines |
| API | FastAPI, uvicorn, pydantic |
| Dashboard | Streamlit, Plotly |
| Orchestration | Prefect |
| Experiment tracking | MLflow |
| Infra | Docker, docker-compose |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
cp .env.example .env        # fill in DB_PASSWORD
make install-dev            # installs all dependencies
```

### 2. Download the dataset

Download **Online Retail** from [UCI](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) or [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) and place it at:

```
data/raw/online_retail_II.xlsx
```

### 3. Start PostgreSQL

```bash
make db-up
```

### 4. Run the full pipeline

```bash
make all
```

This runs ETL → features → all 4 models in sequence.

Or run each step individually:

```bash
make etl                   # ~3 min
make features              # ~1 min
make train-segmentation    # ~30s
make train-churn           # ~1 min
make train-clv             # ~2 min
make train-basket          # ~2 min
```

### 5. Start the services

```bash
make serve       # FastAPI  → http://localhost:8000/docs
make dashboard   # Streamlit → http://localhost:8501
make mlflow      # MLflow   → http://localhost:5000
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness check |
| `POST` | `/api/v1/predict/segment` | RFM → segment label |
| `POST` | `/api/v1/predict/churn` | Features → churn probability |
| `POST` | `/api/v1/predict/clv` | RFM-T → 12M CLV |
| `POST` | `/api/v1/predict/basket` | Basket → product recommendations |
| `GET` | `/api/v1/customer/{id}` | Full customer profile |

Interactive docs: **http://localhost:8000/docs**

---

## Dashboard pages

| Page | What you see |
|---|---|
| 🏠 Overview | KPI cards — revenue, churn rate, avg CLV, P(alive) |
| 🎯 Segments | PCA scatter, RFM box plots, segment profile table |
| ⚠️ Churn | Risk distribution, churn vs CLV scatter, high-risk customer list |
| 💰 CLV | CLV tiers, P(alive) vs expected purchases, top customers |
| 📅 Cohort | Retention heatmap, monthly active customers, acquisition trend |
| 🛒 Market Basket | Association rules explorer, product recommendation search |
| 🔍 Customer Lookup | Full profile + transaction history + recommendations per customer |

---

## Run with Docker (all services)

```bash
cp .env.example .env    # fill in DB_PASSWORD
docker-compose up -d    # starts Postgres + API + Dashboard + MLflow
```

Services:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000

---

## Environment variables

Copy `.env.example` to `.env` and set:

```env
DB_HOST=db                        # Docker service name (not localhost)
DB_PORT=6543
DB_NAME=customer_segmentation
DB_USER=postgres
DB_PASSWORD=your_password_here
```

> **Note:** When running pipeline scripts locally (not in Docker), set `DB_HOST=localhost`.

---

## Postgres tables (retail schema)

| Table | Written by | Contents |
|---|---|---|
| `retail.cleaned_transactions` | `etl/pipeline.py` | ~824K cleaned rows |
| `retail.customer_features` | `build_features.py` | ~5.9K customers × 19 features |
| `retail.segment_results` | `segmentation.py` | Features + segment label |
| `retail.churn_predictions` | `churn.py` | ChurnProb, ChurnRisk per customer |
| `retail.clv_predictions` | `clv.py` | CLV, CLVTier, ProbAlive per customer |
| `retail.association_rules` | `market_basket.py` | FP-Growth rules |

---

## Key results (typical run)

| Metric | Value |
|---|---|
| Customers segmented | ~5,900 |
| Churn model ROC-AUC | ~0.91 |
| Silhouette score | ~0.42 |
| Median 12M CLV | ~£320 |
| Association rules | ~180 |

---

## Makefile reference

```bash
make help               # list all commands
make setup              # first-time setup
make all                # full pipeline
make db-up / db-down    # start / stop Postgres
make db-shell           # psql shell in container
make etl                # ETL pipeline
make features           # feature engineering
make train              # all 4 models
make serve              # FastAPI
make dashboard          # Streamlit
make mlflow             # MLflow UI
make test               # pytest
make lint               # ruff
make format             # black
make clean              # remove logs / caches
```