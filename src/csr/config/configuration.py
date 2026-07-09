import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

from csr.constants import CONFIG_FILE_PATH

# Load .env once at import time
load_dotenv()


# ─── Dataclasses (typed config objects) ──────────────────────────────────────

@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str
    schema: str

    @property
    def url(self) -> str:
        db_url = os.getenv("DB_URL")
        return db_url


@dataclass
class PathsConfig:
    raw_data: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path


@dataclass
class ETLConfig:
    sheet_names: List[str]
    date_column: str
    customer_id_column: str
    invoice_column: str
    quantity_column: str
    price_column: str
    stock_code_column: str
    description_column: str
    country_column: str
    cancellation_prefix: str
    outlier_quantile: float
    interim_table: str
    chunksize: int


@dataclass
class SegmentationConfig:
    n_clusters: int
    random_state: int
    n_init: int
    rfm_columns: List[str]
    artifact_name: str
    scaler_artifact: str


@dataclass
class ChurnConfig:
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    artifact_name: str
    feature_columns: List[str]


@dataclass
class CLVConfig:
    penalizer_coef: float
    bgf_artifact: str
    ggf_artifact: str


@dataclass
class FeaturesConfig:
    rfm_table: str
    behavioural_table: str
    cohort_table: str
    features_table: str
    churn_threshold_days: int
    clv_horizon_months: int
    clv_discount_rate: float


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str


@dataclass
class APIConfig:
    host: str
    port: int
    reload: bool


# ─── Raw yaml loader ─────────────────────────────────────────────────────────

def _load_yaml(path: Path = CONFIG_FILE_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_env(value: str) -> str:
    """Replace ${VAR} placeholders with environment variable values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        resolved = os.getenv(var_name)
        if resolved is None:
            raise EnvironmentError(
                f"Environment variable '{var_name}' is not set. "
                f"Check your .env file."
            )
        return resolved
    return value


# ─── Config manager ──────────────────────────────────────────────────────────

class ConfigurationManager:
    """
    Single entry point for all configuration.

    Usage:
        from src.csr.config.configuration import ConfigurationManager
        cfg = ConfigurationManager()
        db  = cfg.get_database_config()
        etl = cfg.get_etl_config()
    """

    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        self._raw = _load_yaml(config_path)

    def get_database_config(self) -> DatabaseConfig:
        db = self._raw["database"]
        return DatabaseConfig(
            host     = _resolve_env(db["host"]),
            port     = int(_resolve_env(str(db["port"]))),
            name     = _resolve_env(db["name"]),
            user     = _resolve_env(db["user"]),
            password = _resolve_env(db["password"]),
            schema   = db["schema"],
        )

    def get_paths_config(self) -> PathsConfig:
        p = self._raw["paths"]
        return PathsConfig(
            raw_data      = Path(p["raw_data"]),
            interim_dir   = Path(p["interim_dir"]),
            processed_dir = Path(p["processed_dir"]),
            artifacts_dir = Path(p["artifacts_dir"]),
        )

    def get_etl_config(self) -> ETLConfig:
        e = self._raw["etl"]
        return ETLConfig(
            sheet_names          = e["sheet_names"],
            date_column          = e["date_column"],
            customer_id_column   = e["customer_id_column"],
            invoice_column       = e["invoice_column"],
            quantity_column      = e["quantity_column"],
            price_column         = e["price_column"],
            stock_code_column    = e["stock_code_column"],
            description_column   = e["description_column"],
            country_column       = e["country_column"],
            cancellation_prefix  = e["cancellation_prefix"],
            outlier_quantile     = float(e["outlier_quantile"]),
            interim_table        = e["interim_table"],
            chunksize            = int(e["chunksize"]),
        )

    def get_features_config(self) -> FeaturesConfig:
        f = self._raw["features"]
        return FeaturesConfig(
            rfm_table            = f["rfm_table"],
            behavioural_table    = f["behavioural_table"],
            cohort_table         = f["cohort_table"],
            features_table       = f["features_table"],
            churn_threshold_days = int(f["churn_threshold_days"]),
            clv_horizon_months   = int(f["clv_horizon_months"]),
            clv_discount_rate    = float(f["clv_discount_rate"]),
        )

    def get_segmentation_config(self) -> SegmentationConfig:
        s = self._raw["models"]["segmentation"]
        return SegmentationConfig(
            n_clusters      = int(s["n_clusters"]),
            random_state    = int(s["random_state"]),
            n_init          = int(s["n_init"]),
            rfm_columns     = s["rfm_columns"],
            artifact_name   = s["artifact_name"],
            scaler_artifact = s["scaler_artifact"],
        )

    def get_churn_config(self) -> ChurnConfig:
        c = self._raw["models"]["churn"]
        return ChurnConfig(
            test_size        = float(c["test_size"]),
            random_state     = int(c["random_state"]),
            n_estimators     = int(c["n_estimators"]),
            max_depth        = int(c["max_depth"]),
            learning_rate    = float(c["learning_rate"]),
            subsample        = float(c["subsample"]),
            colsample_bytree = float(c["colsample_bytree"]),
            artifact_name    = c["artifact_name"],
            feature_columns  = c["feature_columns"],
        )

    def get_clv_config(self) -> CLVConfig:
        c = self._raw["models"]["clv"]
        return CLVConfig(
            penalizer_coef = float(c["penalizer_coef"]),
            bgf_artifact   = c["bgf_artifact"],
            ggf_artifact   = c["ggf_artifact"],
        )

    def get_mlflow_config(self) -> MLflowConfig:
        m = self._raw["mlflow"]
        return MLflowConfig(
            tracking_uri    = _resolve_env(m["tracking_uri"]),
            experiment_name = _resolve_env(m["experiment_name"]),
        )

    def get_api_config(self) -> APIConfig:
        a = self._raw["api"]
        return APIConfig(
            host   = a["host"],
            port   = int(a["port"]),
            reload = bool(a["reload"]),
        )