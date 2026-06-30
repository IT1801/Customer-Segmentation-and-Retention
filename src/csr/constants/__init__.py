import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Project root ────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[3]   # project root
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"

# Load local configuration before constants backed by environment variables.
load_dotenv(ROOT_DIR / ".env")

# ─── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR          = ROOT_DIR / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"
INTERIM_DATA_DIR  = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "online_retail.xlsx"

# ─── Model artifacts ─────────────────────────────────────────────────────────
ARTIFACTS_DIR         = ROOT_DIR / "models" / "artifacts"
KMEANS_ARTIFACT       = ARTIFACTS_DIR / "kmeans.joblib"
SCALER_RFM_ARTIFACT   = ARTIFACTS_DIR / "scaler_rfm.joblib"
CHURN_MODEL_ARTIFACT  = ARTIFACTS_DIR / "churn_xgb.joblib"
BGF_ARTIFACT          = ARTIFACTS_DIR / "bgf.joblib"
GGF_ARTIFACT          = ARTIFACTS_DIR / "ggf.joblib"

# ─── Database schema & table names ───────────────────────────────────────────
DB_SCHEMA                  = "retail"
TABLE_CLEANED_TRANSACTIONS = "cleaned_transactions"
TABLE_CUSTOMER_FEATURES    = "customer_features"
TABLE_CUSTOMER_RFM         = "customer_rfm"
TABLE_CUSTOMER_BEHAVIOURAL = "customer_behavioural"
TABLE_CUSTOMER_COHORT      = "customer_cohort"
TABLE_SEGMENT_RESULTS      = "segment_results"
TABLE_CHURN_PREDICTIONS    = "churn_predictions"
TABLE_CLV_PREDICTIONS      = "clv_predictions"

# ─── Raw data column names ───────────────────────────────────────────────────
COL_INVOICE     = "InvoiceNo"
COL_STOCK_CODE  = "StockCode"
COL_DESCRIPTION = "Description"
COL_QUANTITY    = "Quantity"
COL_INVOICE_DATE = "InvoiceDate"
COL_PRICE       = "UnitPrice"
COL_CUSTOMER_ID = "CustomerID"
COL_COUNTRY     = "Country"
COL_REVENUE     = "Revenue"

# ─── ETL constants ───────────────────────────────────────────────────────────
EXCEL_SHEET_NAMES      = ["Online Retail"]
CANCELLATION_PREFIX    = "C"
OUTLIER_QUANTILE       = 0.99
DB_INSERT_CHUNKSIZE    = 10_000

# ─── Feature engineering constants ───────────────────────────────────────────
#CHURN_THRESHOLD_DAYS   = 90
CLV_HORIZON_MONTHS     = 12
CLV_DISCOUNT_RATE      = 0.01
CLV_PENALIZER_COEF     = 0.1

# ─── Segmentation constants ──────────────────────────────────────────────────
N_CLUSTERS             = 4
RANDOM_STATE           = 42
N_INIT                 = 10

RFM_FEATURE_COLUMNS = [
    "Recency",
    "Log_Frequency",
    "Log_Monetary",
]

SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At Risk",
    3: "Lost / Inactive",
}

# ─── Churn model constants ───────────────────────────────────────────────────
CHURN_TEST_SIZE        = 0.2
CHURN_N_ESTIMATORS     = 300
CHURN_MAX_DEPTH        = 5
CHURN_LEARNING_RATE    = 0.05
CHURN_SUBSAMPLE        = 0.8
CHURN_COLSAMPLE_BYTREE = 0.8
CHURN_THRESHOLD_DAYS   = 90

CHURN_FEATURE_COLUMNS = [
    "Frequency",
    "Monetary",
    "AOV",
    "SpendStd",
    "UniqueSKUs",
    "RepeatSKURatio",
    "TotalItems",
    "AvgGap",
    "StdGap",
    "WeekendRatio",
    "PreferredDayOfWeek",
    "ActiveMonths",
    "DaysSinceFirstPurchase",
    "ReturnRate",
]

CHURN_RISK_BINS   = [0.0, 0.3, 0.6, 1.0]
CHURN_RISK_LABELS = ["Low", "Medium", "High"]

# ─── MLflow constants ─────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "customer-segmentation-retention",
)

# ─── API constants ────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
