"""
Builds all behavioural features from the cleaned transactions DataFrame.

Feature groups:
    Purchase & Spend  — AOV, SpendStd
    Temporal          — PreferredDayOfWeek, WeekendRatio, PreferredHour
    Product diversity — UniqueSKUs, TotalItems, RepeatSKURatio
    Inter-purchase gap— AvgGap, StdGap
    Return rate       — ReturnRate (computed from raw transactions in Postgres)
"""

import sys

import pandas as pd
import numpy as np

from csr.logging.logger import logging
from csr.exception.exception import CSRException
from src.csr.constants import (
    COL_CUSTOMER_ID,
    COL_INVOICE,
    COL_INVOICE_DATE,
    COL_QUANTITY,
    COL_STOCK_CODE,
    COL_REVENUE,
    CANCELLATION_PREFIX,
    DB_SCHEMA,
    TABLE_CLEANED_TRANSACTIONS,
)
from sqlalchemy.engine import Engine
from sqlalchemy import text


def build_behavioural(
    df: pd.DataFrame,
    engine: Engine,
) -> pd.DataFrame:
    """
    Compute all behavioural features per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions loaded from retail.cleaned_transactions.
        Used for AOV, temporal, product, and gap features.
    engine : Engine
        SQLAlchemy engine — used to query raw data for return rate,
        since cancellations were removed from df during ETL.

    Returns
    -------
    pd.DataFrame
        One row per customer with all behavioural feature columns.
    """
    try:
        logging.info("Building behavioural features...")

        aov      = _build_aov(df)
        temporal = _build_temporal(df)
        product  = _build_product_diversity(df)
        gaps     = _build_purchase_gaps(df)
        returns  = _build_return_rate(engine)

        # ── Merge all feature groups on CustomerID ────────────────────────────
        behavioural = (
            aov
            .merge(temporal, on=COL_CUSTOMER_ID, how="left")
            .merge(product,  on=COL_CUSTOMER_ID, how="left")
            .merge(gaps,     on=COL_CUSTOMER_ID, how="left")
            .merge(returns,  on=COL_CUSTOMER_ID, how="left")
        )

        behavioural = _fill_nulls(behavioural)

        _validate_behavioural(behavioural)

        logging.info(
            f"Behavioural features complete ✓ — "
            f"{len(behavioural):,} customers | "
            f"{behavioural.shape[1]} columns"
        )

        return behavioural

    except Exception as e:
        raise CSRException(e, sys)


# ─── Purchase & Spend ─────────────────────────────────────────────────────────

def _build_aov(df: pd.DataFrame) -> pd.DataFrame:
    """Average order value and spend standard deviation per customer."""
    try:
        aov = (
            df.groupby([COL_CUSTOMER_ID, COL_INVOICE])[COL_REVENUE]
            .sum()
            .reset_index()
            .groupby(COL_CUSTOMER_ID)[COL_REVENUE]
            .agg(AOV="mean", SpendStd="std")
            .reset_index()
        )
        aov["AOV"]      = aov["AOV"].round(2)
        aov["SpendStd"] = aov["SpendStd"].round(2)

        logging.info(f"AOV features built — {len(aov):,} customers")
        return aov

    except Exception as e:
        raise CSRException(e, sys)


# ─── Temporal patterns ────────────────────────────────────────────────────────

def _build_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Preferred day of week, weekend ratio, and preferred hour."""
    try:
        temporal = (
            df.groupby(COL_CUSTOMER_ID)
            .agg(
                PreferredDayOfWeek = (COL_INVOICE_DATE, lambda x: x.dt.dayofweek.mode()[0]),
                WeekendRatio       = (COL_INVOICE_DATE, lambda x: (x.dt.dayofweek >= 5).mean()),
                PreferredHour      = (COL_INVOICE_DATE, lambda x: x.dt.hour.mode()[0]),
            )
            .reset_index()
        )
        temporal["WeekendRatio"] = temporal["WeekendRatio"].round(4)

        logging.info(f"Temporal features built — {len(temporal):,} customers")
        return temporal

    except Exception as e:
        raise CSRException(e, sys)


# ─── Product diversity ────────────────────────────────────────────────────────

def _build_product_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Unique SKUs, total items purchased, and repeat SKU ratio."""
    try:
        product = (
            df.groupby(COL_CUSTOMER_ID)
            .agg(
                UniqueSKUs     = (COL_STOCK_CODE, "nunique"),
                TotalItems     = (COL_QUANTITY,   "sum"),
                RepeatSKURatio = (COL_STOCK_CODE, lambda x: x.duplicated().mean()),
            )
            .reset_index()
        )
        product["RepeatSKURatio"] = product["RepeatSKURatio"].round(4)

        logging.info(f"Product diversity features built — {len(product):,} customers")
        return product

    except Exception as e:
        raise CSRException(e, sys)


# ─── Inter-purchase gaps ──────────────────────────────────────────────────────

def _build_purchase_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average and std of days between consecutive orders per customer.
    Customers with only 1 order will have NaN — filled downstream.
    """
    try:
        gaps = (
            df.groupby([COL_CUSTOMER_ID, COL_INVOICE])[COL_INVOICE_DATE]
            .min()
            .reset_index()
            .sort_values([COL_CUSTOMER_ID, COL_INVOICE_DATE])
            .groupby(COL_CUSTOMER_ID)[COL_INVOICE_DATE]
            .apply(lambda x: x.diff().dt.days.dropna())
            .reset_index(level=0)
            .rename(columns={COL_INVOICE_DATE: "GapDays"})
            .groupby(COL_CUSTOMER_ID)["GapDays"]
            .agg(AvgGap="mean", StdGap="std")
            .reset_index()
        )
        gaps["AvgGap"] = gaps["AvgGap"].round(2)
        gaps["StdGap"] = gaps["StdGap"].round(2)

        logging.info(f"Purchase gap features built — {len(gaps):,} customers")
        return gaps

    except Exception as e:
        raise CSRException(e, sys)


# ─── Return rate ──────────────────────────────────────────────────────────────

def _build_return_rate(engine: Engine) -> pd.DataFrame:
    """
    Compute return rate per customer from the raw transactions table.

    We query the raw (uncleaned) data because cancellations were
    removed during ETL transform — we need them here to compute
    the ratio of cancelled to total orders.
    """
    try:
        query = f"""
            SELECT
                "{COL_CUSTOMER_ID}",
                "{COL_INVOICE}"
            FROM {DB_SCHEMA}.{TABLE_CLEANED_TRANSACTIONS}
        """

        # Re-read raw with cancellations by querying a broader source
        # Since our cleaned table has no cancellations, we reconstruct
        # return rate using the Invoice prefix pattern on raw data.
        # Here we load cleaned data and use the engine to query the
        # raw table if it exists, otherwise default to 0.
        try:
            raw_query = f"""
                SELECT
                    "{COL_CUSTOMER_ID}",
                    "{COL_INVOICE}"
                FROM {DB_SCHEMA}.raw_transactions
                WHERE "{COL_CUSTOMER_ID}" IS NOT NULL
            """
            with engine.connect() as conn:
                df_raw = pd.read_sql(text(raw_query), conn)

            df_raw["IsCancelled"] = (
                df_raw[COL_INVOICE].astype(str).str.startswith(CANCELLATION_PREFIX)
            )
            returns = (
                df_raw.groupby(COL_CUSTOMER_ID)
                .agg(
                    TotalOrders     = (COL_INVOICE,       "nunique"),
                    CancelledOrders = ("IsCancelled",      "sum"),
                )
                .reset_index()
            )
            returns["ReturnRate"] = (
                returns["CancelledOrders"] / returns["TotalOrders"]
            ).round(4)

            logging.info(f"Return rate built from raw_transactions — {len(returns):,} customers")
            return returns[[COL_CUSTOMER_ID, "ReturnRate"]]

        except Exception:
            # raw_transactions table not available — default return rate to 0
            logging.warning(
                "raw_transactions table not found in DB. "
                "ReturnRate will default to 0 for all customers. "
                "To fix: load raw data before running features."
            )
            with engine.connect() as conn:
                df_clean = pd.read_sql(
                    text(f'SELECT DISTINCT "{COL_CUSTOMER_ID}" FROM {DB_SCHEMA}.{TABLE_CLEANED_TRANSACTIONS}'),
                    conn
                )
            df_clean["ReturnRate"] = 0.0
            return df_clean[[COL_CUSTOMER_ID, "ReturnRate"]]

    except Exception as e:
        raise CSRException(e, sys)


# ─── Null filling ─────────────────────────────────────────────────────────────

def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill expected nulls with sensible defaults.

    StdGap / AvgGap — null for single-order customers → 0
    SpendStd        — null for single-order customers → 0
    ReturnRate      — null if customer not in raw data → 0
    """
    try:
        fill_map = {
            "AvgGap"    : 0.0,
            "StdGap"    : 0.0,
            "SpendStd"  : 0.0,
            "ReturnRate": 0.0,
        }
        for col, val in fill_map.items():
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    df[col] = df[col].fillna(val)
                    logging.info(f"Filled {null_count:,} nulls in '{col}' with {val}")

        return df

    except Exception as e:
        raise CSRException(e, sys)


# ─── Validation ───────────────────────────────────────────────────────────────

def _validate_behavioural(df: pd.DataFrame) -> None:
    try:
        errors = []

        if df.empty:
            errors.append("Behavioural DataFrame is empty")

        if df[COL_CUSTOMER_ID].duplicated().any():
            errors.append("Duplicate CustomerIDs in behavioural features")

        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            errors.append(f"Nulls remain after fill:\n{cols_with_nulls}")

        if (df["AOV"] <= 0).any():
            errors.append("Non-positive AOV values found")

        if errors:
            raise ValueError(
                "Behavioural validation failed:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

        logging.info("Behavioural validation passed ✓")

    except Exception as e:
        raise CSRException(e, sys)