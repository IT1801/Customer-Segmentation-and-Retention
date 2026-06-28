"""
Master runner for the feature engineering pipeline.

Steps:
    1. Load cleaned transactions from PostgreSQL
    2. Build RFM features
    3. Build behavioural features
    4. Build cohort features
    5. Merge all into a single customer-level feature matrix
    6. Save to PostgreSQL (retail.customer_features)

Run directly:
    python -m src.csr.features.build_features

Or via Makefile:
    make features
"""

import sys
import time

import pandas as pd
from sqlalchemy import text

from csr.logging.logger import logging
from csr.exception.exception import CSRException
from src.csr.config.configuration import ConfigurationManager
from src.csr.constants import (
    COL_CUSTOMER_ID,
    COL_INVOICE_DATE,
    COL_REVENUE,
    DB_SCHEMA,
    TABLE_CLEANED_TRANSACTIONS,
    TABLE_CUSTOMER_FEATURES,
    DB_INSERT_CHUNKSIZE,
)
from src.csr.etl.load import get_engine
from src.csr.features.rfm import build_rfm
from src.csr.features.behavioural import build_behavioural
from src.csr.features.cohort import build_cohort_features


def run_feature_pipeline() -> None:
    """
    Run the full feature engineering pipeline end to end.

    Steps
    -----
    1. Load config
    2. Load cleaned transactions from Postgres
    3. Build RFM features
    4. Build behavioural features
    5. Build cohort features
    6. Merge all feature groups
    7. Validate final feature matrix
    8. Save to Postgres
    """
    try:
        start = time.time()
        logging.info("=" * 60)
        logging.info("FEATURE PIPELINE STARTED")
        logging.info("=" * 60)

        # ── Step 0: Config & engine ───────────────────────────────────────────
        logging.info("Step 0/6 — Loading configuration")
        cfg        = ConfigurationManager()
        db_config  = cfg.get_database_config()
        feat_config = cfg.get_features_config()
        engine     = get_engine(db_config=db_config)
        logging.info("Configuration and DB engine ready ✓")

        # ── Step 1: Load cleaned transactions from Postgres ───────────────────
        logging.info("Step 1/6 — Loading cleaned transactions from Postgres")
        t1 = time.time()
        df = _load_cleaned_transactions(engine)
        logging.info(
            f"Loaded ✓ — {len(df):,} rows | "
            f"{time.time() - t1:.1f}s"
        )

        # ── Step 2: RFM features ──────────────────────────────────────────────
        logging.info("Step 2/6 — Building RFM features")
        t2  = time.time()
        rfm = build_rfm(df)
        logging.info(
            f"RFM complete ✓ — {len(rfm):,} customers | "
            f"{time.time() - t2:.1f}s"
        )

        # ── Step 3: Behavioural features ──────────────────────────────────────
        logging.info("Step 3/6 — Building behavioural features")
        t3           = time.time()
        behavioural  = build_behavioural(df, engine=engine)
        logging.info(
            f"Behavioural complete ✓ — {len(behavioural):,} customers | "
            f"{time.time() - t3:.1f}s"
        )

        # ── Step 4: Cohort features ───────────────────────────────────────────
        logging.info("Step 4/6 — Building cohort features")
        t4     = time.time()
        cohort = build_cohort_features(df)
        logging.info(
            f"Cohort complete ✓ — {len(cohort):,} customers | "
            f"{time.time() - t4:.1f}s"
        )

        # ── Step 5: Merge all feature groups ──────────────────────────────────
        logging.info("Step 5/6 — Merging all feature groups")
        t5       = time.time()
        features = _merge_features(rfm, behavioural, cohort)
        logging.info(
            f"Merge complete ✓ — "
            f"shape: {features.shape} | "
            f"{time.time() - t5:.1f}s"
        )

        # ── Step 6: Validate & save to Postgres ───────────────────────────────
        logging.info("Step 6/6 — Validating and saving to Postgres")
        t6 = time.time()
        _validate_features(features)
        _save_features(features, engine, feat_config.features_table)
        logging.info(
            f"Save complete ✓ — "
            f"retail.{feat_config.features_table} | "
            f"{time.time() - t6:.1f}s"
        )

        # ── Summary ───────────────────────────────────────────────────────────
        elapsed = time.time() - start
        logging.info("=" * 60)
        logging.info("FEATURE PIPELINE COMPLETE")
        logging.info(f"  Customers    : {len(features):,}")
        logging.info(f"  Features     : {features.shape[1]}")
        logging.info(f"  Total time   : {elapsed:.1f}s")
        logging.info("=" * 60)

    except Exception as e:
        logging.error("FEATURE PIPELINE FAILED")
        raise CSRException(e, sys)


# ─── Load cleaned transactions ────────────────────────────────────────────────

def _load_cleaned_transactions(engine) -> pd.DataFrame:
    """
    Load the cleaned transactions table from Postgres into a DataFrame.
    Parses InvoiceDate to datetime on load.
    """
    try:
        query = text(
            f'SELECT * FROM {DB_SCHEMA}.{TABLE_CLEANED_TRANSACTIONS}'
        )

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        # Ensure correct dtypes after loading from Postgres
        df[COL_INVOICE_DATE] = pd.to_datetime(df[COL_INVOICE_DATE])

        # Add Revenue if not already present
        # (should exist from transform, but guard against schema drift)
        if COL_REVENUE not in df.columns:
            from src.csr.constants import COL_QUANTITY, COL_PRICE
            df[COL_REVENUE] = (df[COL_QUANTITY] * df[COL_PRICE]).round(2)
            logging.warning(
                "Revenue column was missing from DB — recomputed on load"
            )

        logging.info(
            f"Transactions loaded — "
            f"shape: {df.shape} | "
            f"customers: {df[COL_CUSTOMER_ID].nunique():,} | "
            f"date range: {df[COL_INVOICE_DATE].min().date()} → "
            f"{df[COL_INVOICE_DATE].max().date()}"
        )

        return df

    except Exception as e:
        raise CSRException(e, sys)


# ─── Merge feature groups ─────────────────────────────────────────────────────

def _merge_features(
    rfm: pd.DataFrame,
    behavioural: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join all feature groups on CustomerID.
    RFM is the base — every customer in RFM will appear in the output.
    """
    try:
        features = (
            rfm
            .merge(behavioural, on=COL_CUSTOMER_ID, how="left")
            .merge(cohort,      on=COL_CUSTOMER_ID, how="left")
        )

        logging.info(
            f"Feature groups merged — "
            f"RFM: {len(rfm):,} | "
            f"Behavioural: {len(behavioural):,} | "
            f"Cohort: {len(cohort):,} | "
            f"Final: {len(features):,}"
        )

        return features

    except Exception as e:
        raise CSRException(e, sys)


# ─── Validate final feature matrix ───────────────────────────────────────────

def _validate_features(df: pd.DataFrame) -> None:
    """
    Final validation on the merged feature matrix before saving.
    Checks shape, nulls, duplicates, and expected columns.
    """
    try:
        errors = []

        # Must have rows
        if df.empty:
            errors.append("Feature matrix is empty")

        # No duplicate customers
        if df[COL_CUSTOMER_ID].duplicated().any():
            n_dupes = df[COL_CUSTOMER_ID].duplicated().sum()
            errors.append(f"{n_dupes:,} duplicate CustomerIDs in feature matrix")

        # Check for unexpected nulls
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            errors.append(
                f"Null values in feature matrix:\n"
                + "\n".join(
                    f"    {col}: {count:,}"
                    for col, count in cols_with_nulls.items()
                )
            )

        # Core RFM columns must exist
        required_cols = [
            "Recency", "Frequency", "Monetary",
            "Log_Frequency", "Log_Monetary",
            "AOV", "SpendStd",
            "UniqueSKUs", "TotalItems", "RepeatSKURatio",
            "AvgGap", "StdGap",
            "WeekendRatio", "PreferredDayOfWeek",
            "ReturnRate",
            "CohortMonth", "ActiveMonths", "DaysSinceFirstPurchase",
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing expected feature columns: {missing_cols}")

        if errors:
            raise ValueError(
                "Feature matrix validation failed:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

        logging.info(
            f"Feature matrix validation passed ✓ — "
            f"{len(df):,} customers × {df.shape[1]} features"
        )

    except Exception as e:
        raise CSRException(e, sys)


# ─── Save to Postgres ─────────────────────────────────────────────────────────

def _save_features(
    df: pd.DataFrame,
    engine,
    table: str = TABLE_CUSTOMER_FEATURES,
    schema: str = DB_SCHEMA,
) -> None:
    """
    Persist the feature matrix to PostgreSQL.
    Verifies row count after write.
    """
    try:
        full_table = f"{schema}.{table}"
        logging.info(f"Saving {len(df):,} rows → {full_table}")

        df.to_sql(
            name      = table,
            con       = engine,
            schema    = schema,
            if_exists = "replace",
            index     = False,
            chunksize = DB_INSERT_CHUNKSIZE,
            method    = "multi",
        )

        # Verify
        with engine.connect() as conn:
            actual = conn.execute(
                text(f"SELECT COUNT(*) FROM {full_table}")
            ).scalar()

        if actual != len(df):
            raise ValueError(
                f"Row count mismatch in {full_table}: "
                f"expected {len(df):,}, found {actual:,}"
            )

        logging.info(f"Saved and verified ✓ — {actual:,} rows in {full_table}")

    except Exception as e:
        raise CSRException(e, sys)


if __name__ == "__main__":
    run_feature_pipeline()