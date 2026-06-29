"""
K-Means customer segmentation on scaled RFM features.

Steps:
    1. Load customer features from PostgreSQL
    2. Scale RFM columns (StandardScaler)
    3. Fit KMeans
    4. Assign segment labels
    5. Log experiment to MLflow
    6. Save model artifacts (KMeans + Scaler)
    7. Write segment results back to PostgreSQL

Run directly:
    python -m src.csr.models.segmentation
"""

import sys
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from csr.exception.exception import CSRException
from csr.logging.logger import logging
from csr.config.configuration import ConfigurationManager
from csr.constants import (
    ARTIFACTS_DIR,
    COL_CUSTOMER_ID,
    DB_INSERT_CHUNKSIZE,
    DB_SCHEMA,
    KMEANS_ARTIFACT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_CLUSTERS,
    N_INIT,
    RANDOM_STATE,
    RFM_FEATURE_COLUMNS,
    SCALER_RFM_ARTIFACT,
    SEGMENT_LABELS,
    TABLE_CUSTOMER_FEATURES,
    TABLE_SEGMENT_RESULTS,
)
from csr.etl.load import get_engine


def run_segmentation() -> None:
    """
    Run the full segmentation pipeline end to end.
    """
    try:
        start = time.time()
        logging.info("=" * 60)
        logging.info("SEGMENTATION PIPELINE STARTED")
        logging.info("=" * 60)

        # ── Step 0: Config & engine ───────────────────────────────────────────
        cfg    = ConfigurationManager()
        db_cfg = cfg.get_database_config()
        seg_cfg = cfg.get_segmentation_config()
        engine = get_engine(db_config=db_cfg)

        # ── MLflow setup ──────────────────────────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="kmeans_segmentation"):

            # ── Step 1: Load features ─────────────────────────────────────────
            logging.info("Step 1/5 — Loading customer features from Postgres")
            df = _load_features(engine)

            # ── Step 2: Scale RFM ─────────────────────────────────────────────
            logging.info("Step 2/5 — Scaling RFM features")
            X_scaled, scaler = _scale_features(df, seg_cfg.rfm_columns)

            # ── Step 3: Fit KMeans ────────────────────────────────────────────
            logging.info(f"Step 3/5 — Fitting KMeans (K={seg_cfg.n_clusters})")
            km, labels, sil_score = _fit_kmeans(
                X_scaled,
                n_clusters   = seg_cfg.n_clusters,
                random_state = seg_cfg.random_state,
                n_init       = seg_cfg.n_init,
            )

            # ── Step 4: Build results DataFrame ──────────────────────────────
            logging.info("Step 4/5 — Building segment results")
            results = _build_results(df, labels)

            # ── Step 5: Log to MLflow + save artifacts ────────────────────────
            logging.info("Step 5/5 — Logging to MLflow and saving artifacts")
            _log_mlflow(seg_cfg, sil_score, results)
            _save_artifacts(km, scaler, seg_cfg)
            _save_results(results, engine)

        elapsed = time.time() - start
        logging.info("=" * 60)
        logging.info("SEGMENTATION PIPELINE COMPLETE")
        logging.info(f"  Customers    : {len(results):,}")
        logging.info(f"  Segments     : {seg_cfg.n_clusters}")
        logging.info(f"  Silhouette   : {sil_score:.4f}")
        logging.info(f"  Total time   : {elapsed:.1f}s")
        logging.info("=" * 60)

    except Exception as e:
        logging.error("SEGMENTATION PIPELINE FAILED")
        raise CSRException(e, sys)


# ─── Step 1: Load features ────────────────────────────────────────────────────

def _load_features(engine) -> pd.DataFrame:
    try:
        query = text(
            f"SELECT * FROM {DB_SCHEMA}.{TABLE_CUSTOMER_FEATURES}"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        logging.info(
            f"Features loaded — "
            f"{len(df):,} customers × {df.shape[1]} features"
        )
        return df

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 2: Scale features ───────────────────────────────────────────────────

def _scale_features(
    df: pd.DataFrame,
    rfm_columns: list,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Log1p-transform then StandardScale the RFM columns.
    Returns scaled numpy array and fitted scaler.
    """
    try:
        # Log_Frequency and Log_Monetary already log-transformed in rfm.py
        # We just need Recency as-is since it's already roughly normal
        X_raw = df[rfm_columns].values

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        logging.info(
            f"Features scaled — "
            f"columns: {rfm_columns} | "
            f"shape: {X_scaled.shape}"
        )
        return X_scaled, scaler

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 3: Fit KMeans ───────────────────────────────────────────────────────

def _fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple:
    """
    Fit KMeans and compute silhouette score.
    Returns fitted model, labels array, and silhouette score.
    """
    try:
        km = KMeans(
            n_clusters   = n_clusters,
            random_state = random_state,
            n_init       = n_init,
        )
        labels    = km.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)

        logging.info(
            f"KMeans fitted — "
            f"inertia: {km.inertia_:.2f} | "
            f"silhouette: {sil_score:.4f}"
        )

        # Log segment distribution
        unique, counts = np.unique(labels, return_counts=True)
        for seg, count in zip(unique, counts):
            logging.info(
                f"  Segment {seg} ({SEGMENT_LABELS.get(seg, 'Unknown')}): "
                f"{count:,} customers ({count/len(labels)*100:.1f}%)"
            )

        return km, labels, sil_score

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 4: Build results DataFrame ─────────────────────────────────────────

def _build_results(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Attach segment labels to the customer feature DataFrame.
    """
    try:
        results = df.copy()
        results["Segment"]      = labels
        results["SegmentLabel"] = results["Segment"].map(SEGMENT_LABELS)

        # Segment profile summary
        profile = (
            results.groupby("SegmentLabel")[["Recency", "Frequency", "Monetary", "AOV"]]
            .mean()
            .round(2)
        )
        profile["Count"]        = results.groupby("SegmentLabel").size()
        profile["RevenueShare"] = (
            results.groupby("SegmentLabel")["Monetary"].sum()
            / results["Monetary"].sum() * 100
        ).round(1)

        logging.info(f"Segment profile:\n{profile.to_string()}")
        return results

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 5a: Log to MLflow ───────────────────────────────────────────────────

def _log_mlflow(
    seg_cfg,
    sil_score: float,
    results: pd.DataFrame,
) -> None:
    try:
        # Params
        mlflow.log_params({
            "n_clusters"   : seg_cfg.n_clusters,
            "random_state" : seg_cfg.random_state,
            "n_init"       : seg_cfg.n_init,
            "rfm_columns"  : str(seg_cfg.rfm_columns),
        })

        # Metrics
        mlflow.log_metrics({
            "silhouette_score" : round(sil_score, 4),
            "n_customers"      : len(results),
        })

        # Per-segment customer count as metrics
        for label, count in results["SegmentLabel"].value_counts().items():
            safe_label = label.replace(" ", "_").replace("/", "_")
            mlflow.log_metric(f"segment_{safe_label}_count", count)

        logging.info("MLflow logging complete ✓")

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 5b: Save artifacts ──────────────────────────────────────────────────

def _save_artifacts(km, scaler, seg_cfg) -> None:
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        km_path     = ARTIFACTS_DIR / seg_cfg.artifact_name
        scaler_path = ARTIFACTS_DIR / seg_cfg.scaler_artifact

        joblib.dump(km,     km_path)
        joblib.dump(scaler, scaler_path)

        # Also log to MLflow artifact store
        mlflow.sklearn.log_model(km,     "kmeans_model")
        mlflow.sklearn.log_model(scaler, "rfm_scaler")

        logging.info(f"KMeans artifact saved  → {km_path}")
        logging.info(f"Scaler artifact saved  → {scaler_path}")

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 5c: Save results to Postgres ───────────────────────────────────────

def _save_results(results: pd.DataFrame, engine) -> None:
    try:
        full_table = f"{DB_SCHEMA}.{TABLE_SEGMENT_RESULTS}"

        results.to_sql(
            name      = TABLE_SEGMENT_RESULTS,
            con       = engine,
            schema    = DB_SCHEMA,
            if_exists = "replace",
            index     = False,
            chunksize = DB_INSERT_CHUNKSIZE,
            method    = "multi",
        )

        with engine.connect() as conn:
            actual = conn.execute(
                text(f"SELECT COUNT(*) FROM {full_table}")
            ).scalar()

        if actual != len(results):
            raise ValueError(
                f"Row count mismatch: expected {len(results):,}, got {actual:,}"
            )

        logging.info(f"Segment results saved ✓ — {actual:,} rows → {full_table}")

    except Exception as e:
        raise CSRException(e, sys)


if __name__ == "__main__":
    run_segmentation()