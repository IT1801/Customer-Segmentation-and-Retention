"""
Customer Lifetime Value modelling using BG/NBD + Gamma-Gamma.

BG/NBD  — models purchase frequency and predicts expected future
           number of transactions per customer.
Gamma-Gamma — models monetary value per transaction and predicts
              expected average order value conditional on being alive.

Together they produce a 12-month predicted CLV per customer.

Steps:
    1. Load cleaned transactions from PostgreSQL
    2. Build lifetimes summary data (frequency, recency, T, monetary)
    3. Fit BG/NBD model
    4. Fit Gamma-Gamma model
    5. Predict 12-month CLV
    6. Assign CLV tiers
    7. Log to MLflow
    8. Save model artifacts
    9. Write CLV predictions to PostgreSQL

Run directly:
    python -m src.csr.models.clv
"""

import sys
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sqlalchemy import text

from csr.exception.exception import CSRException
from csr.logging.logger import logging
from csr.config.configuration import ConfigurationManager
from csr.constants import (
    ARTIFACTS_DIR,
    BGF_ARTIFACT,
    CLV_DISCOUNT_RATE,
    CLV_HORIZON_MONTHS,
    CLV_PENALIZER_COEF,
    COL_CUSTOMER_ID,
    COL_INVOICE_DATE,
    COL_INVOICE,
    COL_REVENUE,
    DB_INSERT_CHUNKSIZE,
    DB_SCHEMA,
    GGF_ARTIFACT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TABLE_CLEANED_TRANSACTIONS,
    TABLE_CLV_PREDICTIONS,
)
from csr.etl.load import get_engine


def run_clv_pipeline() -> None:
    """
    Run the full CLV modelling pipeline end to end.
    """
    try:
        start = time.time()
        logging.info("=" * 60)
        logging.info("CLV PIPELINE STARTED")
        logging.info("=" * 60)

        # ── Step 0: Config & engine ───────────────────────────────────────────
        cfg      = ConfigurationManager()
        db_cfg   = cfg.get_database_config()
        clv_cfg  = cfg.get_clv_config()
        feat_cfg = cfg.get_features_config()
        engine   = get_engine(db_config=db_cfg)

        # ── MLflow setup ──────────────────────────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="bgnbd_gamma_gamma_clv"):

            # ── Step 1: Load cleaned transactions ─────────────────────────────
            logging.info("Step 1/6 — Loading cleaned transactions from Postgres")
            df = _load_transactions(engine)

            # ── Step 2: Build lifetimes summary ───────────────────────────────
            logging.info("Step 2/6 — Building lifetimes summary data")
            summary = _build_summary(df)

            # ── Step 3: Fit BG/NBD ────────────────────────────────────────────
            logging.info("Step 3/6 — Fitting BG/NBD model")
            bgf = _fit_bgnbd(summary, CLV_PENALIZER_COEF)

            # ── Step 4: Fit Gamma-Gamma ───────────────────────────────────────
            logging.info("Step 4/6 — Fitting Gamma-Gamma model")
            ggf = _fit_gamma_gamma(summary, CLV_PENALIZER_COEF)

            # ── Step 5: Predict CLV ───────────────────────────────────────────
            logging.info(
                f"Step 5/6 — Predicting {CLV_HORIZON_MONTHS}-month CLV"
            )
            clv_df = _predict_clv(
                summary,
                bgf,
                ggf,
                horizon_months = CLV_HORIZON_MONTHS,
                discount_rate  = CLV_DISCOUNT_RATE,
            )

            # ── Step 6: Log + save + persist ──────────────────────────────────
            logging.info("Step 6/6 — MLflow, artifacts, saving predictions")
            _log_mlflow(clv_cfg, feat_cfg, bgf, ggf, clv_df)
            _save_artifacts(bgf, ggf, clv_cfg)
            _save_predictions(clv_df, engine)

        elapsed = time.time() - start
        logging.info("=" * 60)
        logging.info("CLV PIPELINE COMPLETE")
        logging.info(f"  Customers modelled : {len(clv_df):,}")
        logging.info(f"  Median 12M CLV     : £{clv_df['PredictedCLV_12M'].median():.2f}")
        logging.info(f"  Total 12M CLV      : £{clv_df['PredictedCLV_12M'].sum():,.2f}")
        logging.info(f"  Total time         : {elapsed:.1f}s")
        logging.info("=" * 60)

    except Exception as e:
        logging.error("CLV PIPELINE FAILED")
        raise CSRException(e, sys)


# ─── Step 1: Load transactions ────────────────────────────────────────────────

def _load_transactions(engine) -> pd.DataFrame:
    try:
        query = text(
            f"SELECT "
            f'"{COL_CUSTOMER_ID}", '
            f'"{COL_INVOICE}", '
            f'"{COL_INVOICE_DATE}", '
            f'"{COL_REVENUE}" '
            f"FROM {DB_SCHEMA}.{TABLE_CLEANED_TRANSACTIONS}"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        df[COL_INVOICE_DATE] = pd.to_datetime(df[COL_INVOICE_DATE])

        logging.info(
            f"Transactions loaded — "
            f"{len(df):,} rows | "
            f"{df[COL_CUSTOMER_ID].nunique():,} customers"
        )
        return df

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 2: Build lifetimes summary ─────────────────────────────────────────

def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use lifetimes utility to build the RFM-T summary table:

        frequency      — number of repeat purchases (total - 1)
        recency        — time between first and last purchase
        T              — total observation period (age of customer)
        monetary_value — mean revenue per repeat transaction

    Customers with frequency == 0 (only one purchase) are excluded
    from Gamma-Gamma but kept for BG/NBD alive probability.
    """
    try:
        observation_period_end = df[COL_INVOICE_DATE].max()

        summary = summary_data_from_transaction_data(
            df,
            customer_id_col    = COL_CUSTOMER_ID,
            datetime_col       = COL_INVOICE_DATE,
            monetary_value_col = COL_REVENUE,
            observation_period_end = observation_period_end,
            freq               = "D",   # daily frequency
        )

        logging.info(
            f"Lifetimes summary built — "
            f"{len(summary):,} customers | "
            f"observation end: {observation_period_end.date()}"
        )
        logging.info(
            f"  frequency range : {summary['frequency'].min():.0f} – "
            f"{summary['frequency'].max():.0f}"
        )
        logging.info(
            f"  T (age) range   : {summary['T'].min():.0f} – "
            f"{summary['T'].max():.0f} days"
        )

        # Validate freq-monetary correlation (must be < 0.3 for Gamma-Gamma)
        repeat_buyers = summary[summary["frequency"] > 0]
        corr = repeat_buyers[["frequency", "monetary_value"]].corr().iloc[0, 1]
        logging.info(f"  freq-monetary corr: {corr:.3f} (must be < 0.3 for GG)")
        if abs(corr) >= 0.3:
            logging.warning(
                f"freq-monetary correlation {corr:.3f} exceeds 0.3 — "
                f"Gamma-Gamma assumptions may be violated"
            )

        return summary

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 3: Fit BG/NBD ───────────────────────────────────────────────────────

def _fit_bgnbd(
    summary: pd.DataFrame,
    penalizer_coef: float,
) -> BetaGeoFitter:
    """
    Fit the BG/NBD model on all customers (including one-time buyers).
    BG/NBD models the probability a customer is still alive and their
    expected number of future purchases.
    """
    try:
        
        bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
        bgf.fit(
            summary["frequency"],
            summary["recency"],
            summary["T"],
        )

        # Expected purchases in next 90 days
        summary["ExpectedPurchases_90D"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            90,
            summary["frequency"],
            summary["recency"],
            summary["T"],
        ).round(4)

        # Probability alive
        summary["ProbAlive"] = bgf.conditional_probability_alive(
            summary["frequency"],
            summary["recency"],
            summary["T"],
        ).round(4)

        logging.info(
            f"BG/NBD fitted ✓ — "
            f"params: {dict(zip(['r','alpha','a','b'], bgf.params_.values))}"
        )
        logging.info(
            f"  Avg P(alive)          : {summary['ProbAlive'].mean():.3f}"
        )
        logging.info(
            f"  Avg expected purchases: {summary['ExpectedPurchases_90D'].mean():.3f}"
        )

        return bgf

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 4: Fit Gamma-Gamma ──────────────────────────────────────────────────

def _fit_gamma_gamma(
    summary: pd.DataFrame,
    penalizer_coef: float,
) -> GammaGammaFitter:
    """
    Fit the Gamma-Gamma model on repeat buyers only (frequency > 0).
    Gamma-Gamma models expected monetary value per transaction.
    """
    try:
        # Gamma-Gamma requires customers with at least one repeat purchase
        repeat_buyers = summary[summary["frequency"] > 0].copy()

        logging.info(
            f"Fitting Gamma-Gamma on {len(repeat_buyers):,} repeat buyers "
            f"(excluded {len(summary) - len(repeat_buyers):,} one-time buyers)"
        )

        ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
        ggf.fit(
            repeat_buyers["frequency"],
            repeat_buyers["monetary_value"],
        )

        logging.info(
            f"Gamma-Gamma fitted ✓ — "
            f"params: {dict(zip(['p','q','v'], ggf.params_.values))}"
        )

        return ggf

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 5: Predict CLV ──────────────────────────────────────────────────────

def _predict_clv(
    summary: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    horizon_months: int,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Predict CLV for repeat buyers over the configured horizon.
    One-time buyers (frequency == 0) get CLV = 0 as a conservative estimate.
    """
    try:
        repeat_buyers = summary[summary["frequency"] > 0].copy()

        # Predicted CLV from Gamma-Gamma
        repeat_buyers["PredictedCLV_12M"] = ggf.customer_lifetime_value(
            bgf,
            repeat_buyers["frequency"],
            repeat_buyers["recency"],
            repeat_buyers["T"],
            repeat_buyers["monetary_value"],
            time          = horizon_months,
            discount_rate = discount_rate,
            freq          = "D",
        ).round(2)

        # Merge back with full summary — one-time buyers get CLV = 0
        clv_df = summary.merge(
            repeat_buyers[["PredictedCLV_12M"]],
            left_index  = True,
            right_index = True,
            how         = "left",
        )
        clv_df["PredictedCLV_12M"] = clv_df["PredictedCLV_12M"].fillna(0.0)

        # CLV tiers (quartile-based)
        tiers, bins = pd.qcut(
            clv_df["PredictedCLV_12M"],
            q=4,
            retbins=True,
            duplicates="drop",
        )

        labels = ["Bronze", "Silver", "Gold", "Platinum"][: len(bins) - 1]

        tiers = tiers.cat.rename_categories(labels)

        clv_df["CLVTier"] = tiers

        # Reset index so CustomerID becomes a column
        clv_df = clv_df.reset_index()

        logging.info(
            f"CLV predictions complete ✓ — "
            f"{len(clv_df):,} customers | "
            f"horizon: {horizon_months} months"
        )
        logging.info(f"  Min CLV    : £{clv_df['PredictedCLV_12M'].min():.2f}")
        logging.info(f"  Median CLV : £{clv_df['PredictedCLV_12M'].median():.2f}")
        logging.info(f"  Mean CLV   : £{clv_df['PredictedCLV_12M'].mean():.2f}")
        logging.info(f"  Max CLV    : £{clv_df['PredictedCLV_12M'].max():.2f}")

        # CLV tier breakdown
        for tier, count in clv_df["CLVTier"].value_counts().items():
            logging.info(
                f"  {tier}: {count:,} customers "
                f"({count/len(clv_df)*100:.1f}%)"
            )

        return clv_df

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 6a: Log to MLflow ───────────────────────────────────────────────────

def _log_mlflow(clv_cfg, feat_cfg, bgf, ggf, clv_df) -> None:
    try:
        mlflow.log_params({
            "penalizer_coef"   : clv_cfg.penalizer_coef,
            "clv_horizon_months": feat_cfg.clv_horizon_months,
            "clv_discount_rate" : feat_cfg.clv_discount_rate,
            "bgf_params"       : str(dict(zip(
                ["r","alpha","a","b"], bgf.params_.values
            ))),
            "ggf_params"       : str(dict(zip(
                ["p","q","v"], ggf.params_.values
            ))),
        })

        mlflow.log_metrics({
            "n_customers"        : len(clv_df),
            "median_clv_12m"     : round(float(clv_df["PredictedCLV_12M"].median()), 2),
            "mean_clv_12m"       : round(float(clv_df["PredictedCLV_12M"].mean()), 2),
            "total_clv_12m"      : round(float(clv_df["PredictedCLV_12M"].sum()), 2),
            "mean_prob_alive"     : round(float(clv_df["ProbAlive"].mean()), 4),
            "pct_one_time_buyers" : round(
                float((clv_df["frequency"] == 0).mean() * 100), 2
            ),
        })

        logging.info("MLflow logging complete ✓")

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 6b: Save artifacts ─────────────────────────────────────────────────

def _save_artifacts(bgf, ggf, clv_cfg) -> None:
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        bgf_path = ARTIFACTS_DIR / clv_cfg.bgf_artifact
        ggf_path = ARTIFACTS_DIR / clv_cfg.ggf_artifact

        joblib.dump(
            {
                "params": bgf.params_,
                "penalizer_coef": bgf.penalizer_coef,
            },
            bgf_path,
        )

        joblib.dump(
            {
                "params": ggf.params_,
                "penalizer_coef": ggf.penalizer_coef,
            },
            ggf_path,
        )

        mlflow.log_artifact(str(bgf_path), artifact_path="clv_models")
        mlflow.log_artifact(str(ggf_path), artifact_path="clv_models")

        logging.info(f"BG/NBD artifact saved  → {bgf_path}")
        logging.info(f"Gamma-Gamma artifact saved → {ggf_path}")

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 6c: Save predictions to Postgres ────────────────────────────────────

def _save_predictions(clv_df: pd.DataFrame, engine) -> None:
    try:
        # Select only the columns needed downstream
        output_cols = [
            COL_CUSTOMER_ID,
            "frequency",
            "recency",
            "T",
            "monetary_value",
            "ProbAlive",
            "ExpectedPurchases_90D",
            "PredictedCLV_12M",
            "CLVTier",
        ]
        output = clv_df[
            [c for c in output_cols if c in clv_df.columns]
        ].copy()

        full_table = f"{DB_SCHEMA}.{TABLE_CLV_PREDICTIONS}"

        output.to_sql(
            name      = TABLE_CLV_PREDICTIONS,
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

        if actual != len(output):
            raise ValueError(
                f"Row count mismatch: expected {len(output):,}, got {actual:,}"
            )

        logging.info(
            f"CLV predictions saved ✓ — "
            f"{actual:,} rows → {full_table}"
        )

    except Exception as e:
        raise CSRException(e, sys)


if __name__ == "__main__":
    run_clv_pipeline()