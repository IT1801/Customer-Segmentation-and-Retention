"""
Builds cohort-level features from the cleaned transactions DataFrame.

Output columns per customer:
    CohortMonth            — month of first ever purchase (str: "YYYY-MM")
    ActiveMonths           — number of distinct months with at least one order
    DaysSinceFirstPurchase — (last purchase date - first purchase date) in days
"""

import sys

import pandas as pd

from csr.logging.logger import logging
from csr.exception.exception import CSRException
from src.csr.constants import (
    COL_CUSTOMER_ID,
    COL_INVOICE_DATE,
)


def build_cohort_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cohort features per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions loaded from retail.cleaned_transactions.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        CustomerID, CohortMonth, ActiveMonths, DaysSinceFirstPurchase
    """
    try:
        logging.info("Building cohort features...")

        df = df.copy()
        df["InvoiceMonth"] = df[COL_INVOICE_DATE].dt.to_period("M")

        # ── Cohort month — month of first purchase per customer ───────────────
        first_purchase = (
            df.groupby(COL_CUSTOMER_ID)["InvoiceMonth"]
            .min()
            .rename("CohortMonth")
            .reset_index()
        )

        df = df.merge(first_purchase, on=COL_CUSTOMER_ID, how="left")

        # ── Aggregate cohort features ─────────────────────────────────────────
        cohort_feats = (
            df.groupby(COL_CUSTOMER_ID)
            .agg(
                CohortMonth            = ("CohortMonth",      "first"),
                ActiveMonths           = ("InvoiceMonth",      "nunique"),
                DaysSinceFirstPurchase = (COL_INVOICE_DATE,
                                          lambda x: (x.max() - x.min()).days),
            )
            .reset_index()
        )

        # Convert Period to string so it serialises cleanly into Postgres
        cohort_feats["CohortMonth"] = cohort_feats["CohortMonth"].astype(str)

        _validate_cohort(cohort_feats)

        logging.info(
            f"Cohort features complete ✓ — "
            f"{len(cohort_feats):,} customers | "
            f"cohorts: {cohort_feats['CohortMonth'].nunique()} months"
        )

        return cohort_feats

    except Exception as e:
        raise CSRException(e, sys)


def _validate_cohort(df: pd.DataFrame) -> None:
    try:
        errors = []

        if df.empty:
            errors.append("Cohort DataFrame is empty")

        if df[COL_CUSTOMER_ID].duplicated().any():
            errors.append("Duplicate CustomerIDs in cohort features")

        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            errors.append(f"Nulls found in cohort columns: {null_cols}")

        if (df["ActiveMonths"] <= 0).any():
            errors.append("Zero or negative ActiveMonths found")

        if (df["DaysSinceFirstPurchase"] < 0).any():
            errors.append("Negative DaysSinceFirstPurchase found")

        if errors:
            raise ValueError(
                "Cohort validation failed:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

        logging.info("Cohort validation passed ✓")

    except Exception as e:
        raise CSRException(e, sys)