"""
Builds RFM (Recency, Frequency, Monetary) features from the
cleaned transactions table in PostgreSQL.

Output columns per customer:
    Recency        — days since last purchase (lower = more recent)
    Frequency      — number of unique orders placed
    Monetary       — total revenue generated
    Log_Frequency  — log1p(Frequency) for skew reduction
    Log_Monetary   — log1p(Monetary)  for skew reduction
"""

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from csr.logging.logger import logging
from csr.exception.exception import CSRException
from src.csr.constants import (
    COL_CUSTOMER_ID,
    COL_INVOICE,
    COL_INVOICE_DATE,
    COL_REVENUE,
)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM features from the cleaned transactions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions — output of transform() loaded from
        retail.cleaned_transactions.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        CustomerID, Recency, Frequency, Monetary,
        Log_Frequency, Log_Monetary
    """
    try:
        logging.info("Building RFM features...")

        # Snapshot date — 1 day after the last transaction
        # so the most recent customer gets Recency = 1, not 0
        snapshot_date = df[COL_INVOICE_DATE].max() + timedelta(days=1)
        logging.info(f"Snapshot date: {snapshot_date.date()}")

        # ── Core RFM aggregation ──────────────────────────────────────────────
        rfm = (
            df.groupby(COL_CUSTOMER_ID)
            .agg(
                Recency   = (COL_INVOICE_DATE, lambda x: (snapshot_date - x.max()).days),
                Frequency = (COL_INVOICE,       "nunique"),
                Monetary  = (COL_REVENUE,        "sum"),
            )
            .reset_index()
        )

        # ── Log transform to reduce right skew ───────────────────────────────
        # log1p used instead of log to safely handle any zero values
        rfm["Log_Frequency"] = np.log1p(rfm["Frequency"])
        rfm["Log_Monetary"]  = np.log1p(rfm["Monetary"])

        # ── Round monetary values ─────────────────────────────────────────────
        rfm["Monetary"]     = rfm["Monetary"].round(2)
        rfm["Log_Monetary"] = rfm["Log_Monetary"].round(4)

        _validate_rfm(rfm)

        logging.info(
            f"RFM complete ✓ — "
            f"{len(rfm):,} customers | "
            f"Recency range: {rfm['Recency'].min()}–{rfm['Recency'].max()} days | "
            f"Monetary range: £{rfm['Monetary'].min():.2f}–£{rfm['Monetary'].max():.2f}"
        )

        return rfm

    except Exception as e:
        raise CSRException(e, sys)


def _validate_rfm(rfm: pd.DataFrame) -> None:
    """Sanity checks on the RFM output."""
    try:
        errors = []

        if rfm.empty:
            errors.append("RFM DataFrame is empty")

        if rfm[COL_CUSTOMER_ID].duplicated().any():
            errors.append("Duplicate CustomerIDs found in RFM output")

        if (rfm["Recency"] < 0).any():
            errors.append("Negative Recency values found")

        if (rfm["Frequency"] <= 0).any():
            errors.append("Zero or negative Frequency values found")

        if (rfm["Monetary"] <= 0).any():
            errors.append("Zero or negative Monetary values found")

        if rfm[["Recency","Frequency","Monetary",
                 "Log_Frequency","Log_Monetary"]].isnull().any().any():
            errors.append("Null values found in RFM columns")

        if errors:
            raise ValueError(
                "RFM validation failed:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

        logging.info("RFM validation passed ✓")

    except Exception as e:
        raise CSRException(e, sys)