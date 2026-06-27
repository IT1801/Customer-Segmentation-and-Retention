"""
Responsible for all cleaning and transformation logic.
Takes the raw DataFrame from extract.py and returns a clean,
validated DataFrame ready to be loaded into PostgreSQL.

Cleaning steps (in order):
    1.  Drop internal tracking column (_source_sheet)
    2.  Parse InvoiceDate to datetime
    3.  Drop rows with null CustomerID
    4.  Cast CustomerID to clean string
    5.  Remove cancellations (Invoice starts with 'C')
    6.  Remove non-positive Quantity and Price
    7.  Remove duplicate rows
    8.  Cap outliers at configured quantile
    9.  Add Revenue column (Quantity × Price)
    10. Strip whitespace from string columns
    11. Final validation
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

from csr.logging.logger import logging
from csr.exception.exception import CSRException
from csr.constants import (
    COL_CUSTOMER_ID,
    COL_INVOICE,
    COL_INVOICE_DATE,
    COL_PRICE,
    COL_QUANTITY,
    COL_STOCK_CODE,
    COL_DESCRIPTION,
    COL_COUNTRY,
    COL_REVENUE,
    CANCELLATION_PREFIX,
    OUTLIER_QUANTILE,
)
from csr.config.configuration import ConfigurationManager, ETLConfig


def transform(
    df: pd.DataFrame,
    etl_config: ETLConfig = None,
) -> pd.DataFrame:
    """
    Apply all cleaning and transformation steps to the raw DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by extract().
    etl_config : ETLConfig, optional
        ETL config object. If None, loads from ConfigurationManager.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for loading into PostgreSQL.
    """
    try:
        if etl_config is None:
            etl_config = ConfigurationManager().get_etl_config()

        logging.info(f"Starting transform — input shape: {df.shape}")
        df = df.copy()  # never mutate the input

        df = _drop_internal_columns(df)
        df = _parse_dates(df, etl_config)
        df = _drop_null_customer_id(df, etl_config)
        df = _cast_customer_id(df, etl_config)
        df = _remove_cancellations(df, etl_config)
        df = _remove_non_positive(df, etl_config)
        df = _remove_duplicates(df)
        df = _cap_outliers(df, etl_config)
        df = _add_revenue(df, etl_config)
        df = _strip_whitespace(df)

        _validate_clean(df, etl_config)

        logging.info(f"Transform complete — output shape: {df.shape}")
        return df

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 1: Drop internal columns ───────────────────────────────────────────

def _drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        before_cols = df.columns.tolist()
        df = df.drop(columns=["_source_sheet"], errors="ignore")
        logging.info(f"Dropped internal columns — remaining: {df.columns.tolist()}")
        return df
    except Exception as e:
        raise CSRException(e, sys)

 
# ─── Step 2: Parse dates ──────────────────────────────────────────────────────

def _parse_dates(df: pd.DataFrame, etl_config: ETLConfig) -> pd.DataFrame:
    try:
        col = etl_config.date_column
        df[col] = pd.to_datetime(df[col], errors="coerce")

        null_dates = df[col].isnull().sum()
        if null_dates > 0:
            logging.warning(f"Dropping {null_dates:,} rows with unparseable dates")
            df = df.dropna(subset=[col])

        logging.info(
            f"Dates parsed — range: "
            f"{df[col].min().date()} → {df[col].max().date()}"
        )
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 3: Drop null CustomerID ────────────────────────────────────────────

def _drop_null_customer_id(df: pd.DataFrame, etl_config: ETLConfig) -> pd.DataFrame:
    try:
        col = etl_config.customer_id_column
        before = len(df)
        df = df.dropna(subset=[col])
        dropped = before - len(df)
        logging.info(
            f"Dropped {dropped:,} rows with null CustomerID "
            f"({dropped / before * 100:.1f}% of raw data)"
        )
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 4: Cast CustomerID ──────────────────────────────────────────────────

def _cast_customer_id(df: pd.DataFrame, etl_config: ETLConfig) -> pd.DataFrame:
    try:
        col = etl_config.customer_id_column
        # strip trailing .0 from float strings e.g. "12345.0" → "12345"
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        logging.info(
            f"CustomerID cast to string — "
            f"unique customers: {df[col].nunique():,}"
        )
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 5: Remove cancellations ────────────────────────────────────────────

def _remove_cancellations(df: pd.DataFrame, etl_config: ETLConfig) -> pd.DataFrame:
    try:
        col    = etl_config.invoice_column
        prefix = etl_config.cancellation_prefix
        before = len(df)

        is_cancellation = df[col].astype(str).str.startswith(prefix)
        df = df[~is_cancellation]

        dropped = before - len(df)
        logging.info(
            f"Removed {dropped:,} cancellation rows "
            f"(Invoice starting with '{prefix}')"
        )
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 6: Remove non-positive Quantity and Price ──────────────────────────

def _remove_non_positive(df: pd.DataFrame, etl_config: ETLConfig) -> pd.DataFrame:
    try:
        qty_col   = etl_config.quantity_column
        price_col = etl_config.price_column
        before    = len(df)

        df = df[(df[qty_col] > 0) & (df[price_col] > 0)]

        dropped = before - len(df)
        logging.info(
            f"Removed {dropped:,} rows with non-positive "
            f"Quantity or Price"
        )
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 7: Remove duplicates ───────────────────────────────────────────────

def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        logging.info(f"Removed {dropped:,} duplicate rows")
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 8: Cap outliers ────────────────────────────────────────────────────

def _cap_outliers(df: pd.DataFrame, etl_config: ETLConfig) -> pd.DataFrame:
    try:
        qty_col   = etl_config.quantity_column
        price_col = etl_config.price_column
        q         = etl_config.outlier_quantile
        before    = len(df)

        qty_cap   = df[qty_col].quantile(q)
        price_cap = df[price_col].quantile(q)

        df = df[
            (df[qty_col]   <= qty_cap) &
            (df[price_col] <= price_cap)
        ]

        dropped = before - len(df)
        logging.info(
            f"Outlier cap @ {q} quantile — "
            f"Qty ≤ {qty_cap:.0f}, Price ≤ £{price_cap:.2f} — "
            f"removed {dropped:,} rows"
        )
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 9: Add Revenue column ──────────────────────────────────────────────

def _add_revenue(df: pd.DataFrame, etl_config: ETLConfig) -> pd.DataFrame:
    try:
        qty_col   = etl_config.quantity_column
        price_col = etl_config.price_column
        df[COL_REVENUE] = (
            df[qty_col] * df[price_col]
        ).round(2)

        logging.info(
            f"Revenue column added — "
            f"total: £{df[COL_REVENUE].sum():,.2f}"
        )
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 10: Strip whitespace from string columns ───────────────────────────

def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    try:
        str_cols = df.select_dtypes(include="object").columns

        for col in str_cols:
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
        logging.info(f"Stripped whitespace from columns: {str_cols.tolist()}")
        return df
    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 11: Final validation ───────────────────────────────────────────────

def _validate_clean(df: pd.DataFrame, etl_config: ETLConfig) -> None:
    try:
        errors = []

        # No nulls allowed after cleaning
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            errors.append(f"Nulls found after cleaning:\n{cols_with_nulls}")

        # Quantity and Price must all be positive
        if not (df[etl_config.quantity_column] > 0).all():
            errors.append("Non-positive values found in Quantity after cleaning")

        if not (df[etl_config.price_column] > 0).all():
            errors.append("Non-positive values found in Price after cleaning")

        # Revenue column must exist
        if COL_REVENUE not in df.columns:
            errors.append("Revenue column missing after transform")

        # No cancellations should remain
        cancellations_remaining = (
            df[etl_config.invoice_column]
            .astype(str)
            .str.startswith(etl_config.cancellation_prefix)
            .sum()
        )
        if cancellations_remaining > 0:
            errors.append(
                f"{cancellations_remaining:,} cancellation rows still present"
            )

        # DataFrame must not be empty
        if df.empty:
            errors.append("Cleaned DataFrame is empty")

        if errors:
            raise ValueError(
                "Validation failed after transform:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

        logging.info("Clean data validation passed ✓")
        logging.info(
            f"Final summary — "
            f"rows: {len(df):,} | "
            f"customers: {df[etl_config.customer_id_column].nunique():,} | "
            f"orders: {df[etl_config.invoice_column].nunique():,} | "
            f"revenue: £{df[COL_REVENUE].sum():,.2f}"
        )

    except Exception as e:
        raise CSRException(e, sys)