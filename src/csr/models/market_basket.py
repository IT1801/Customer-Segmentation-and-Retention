"""
src/csr/models/market_basket.py

Market basket analysis using Apriori / FP-Growth association rules.

Finds products that are frequently purchased together, generating
actionable rules of the form:
    "Customers who bought X also bought Y"

Steps:
    1. Load cleaned transactions from PostgreSQL
    2. Build basket matrix (Invoice × StockCode binary pivot)
    3. Run FP-Growth to find frequent itemsets
    4. Generate association rules (lift, confidence, support)
    5. Filter high-quality rules
    6. Log to MLflow
    7. Save rules to PostgreSQL

Run directly:
    python -m src.csr.models.market_basket
"""

import sys
import time

import mlflow
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sqlalchemy import text

from csr.exception.exception import CSRException
from csr.logging.logger import logging
from csr.config.configuration import ConfigurationManager
from csr.constants import (
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE,
    COL_STOCK_CODE,
    DB_INSERT_CHUNKSIZE,
    DB_SCHEMA,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TABLE_CLEANED_TRANSACTIONS,
)
from csr.etl.load import get_engine

# ─── Thresholds ───────────────────────────────────────────────────────────────
MIN_SUPPORT     = 0.02   # itemset must appear in ≥ 2% of baskets
MIN_CONFIDENCE  = 0.3    # rule must be correct ≥ 30% of the time
MIN_LIFT        = 1.5    # rule must be 1.5x better than random co-occurrence
MAX_RULES       = 500    # cap output to avoid bloating Postgres


def run_market_basket_pipeline() -> None:
    """
    Run the full market basket analysis pipeline end to end.
    """
    try:
        start = time.time()
        logging.info("=" * 60)
        logging.info("MARKET BASKET PIPELINE STARTED")
        logging.info("=" * 60)

        # ── Step 0: Config & engine ───────────────────────────────────────────
        cfg    = ConfigurationManager()
        db_cfg = cfg.get_database_config()
        engine = get_engine(db_config=db_cfg)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="fpgrowth_market_basket"):

            # ── Step 1: Load transactions ─────────────────────────────────────
            logging.info("Step 1/5 — Loading transactions from Postgres")
            df = _load_transactions(engine)

            # ── Step 2: Build basket matrix ───────────────────────────────────
            logging.info("Step 2/5 — Building basket matrix")
            basket, product_map = _build_basket_matrix(df)

            # ── Step 3: Run FP-Growth ─────────────────────────────────────────
            logging.info(
                f"Step 3/5 — Running FP-Growth "
                f"(min_support={MIN_SUPPORT})"
            )
            itemsets = _run_fpgrowth(basket)

            # ── Step 4: Generate association rules ────────────────────────────
            logging.info(
                f"Step 4/5 — Generating rules "
                f"(min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT})"
            )
            rules = _generate_rules(itemsets, product_map)

            # ── Step 5: Log + save ────────────────────────────────────────────
            logging.info("Step 5/5 — MLflow logging and saving rules")
            _log_mlflow(basket, itemsets, rules)
            _save_rules(rules, engine)

        elapsed = time.time() - start
        logging.info("=" * 60)
        logging.info("MARKET BASKET PIPELINE COMPLETE")
        logging.info(f"  Baskets analysed : {len(basket):,}")
        logging.info(f"  Frequent itemsets: {len(itemsets):,}")
        logging.info(f"  Rules generated  : {len(rules):,}")
        logging.info(f"  Total time       : {elapsed:.1f}s")
        logging.info("=" * 60)

    except Exception as e:
        logging.error("MARKET BASKET PIPELINE FAILED")
        raise CSRException(e, sys)


# ─── Step 1: Load transactions ────────────────────────────────────────────────

def _load_transactions(engine) -> pd.DataFrame:
    try:
        # Load only the columns needed for basket analysis
        query = text(
            f"SELECT "
            f'"{COL_INVOICE}", '
            f'"{COL_STOCK_CODE}", '
            f'"{COL_DESCRIPTION}", '
            f'"{COL_CUSTOMER_ID}" '
            f"FROM {DB_SCHEMA}.{TABLE_CLEANED_TRANSACTIONS}"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        logging.info(
            f"Transactions loaded — "
            f"{len(df):,} rows | "
            f"{df[COL_INVOICE].nunique():,} invoices | "
            f"{df[COL_STOCK_CODE].nunique():,} unique SKUs"
        )
        return df

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 2: Build basket matrix ──────────────────────────────────────────────

def _build_basket_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Build a binary Invoice × StockCode matrix where:
        1 = product was in that basket
        0 = product was not in that basket

    Also returns a product_map dict: StockCode → Description
    for human-readable rule output.

    Filters:
    - UK only (largest market, avoids sparse cross-market baskets)
    - StockCodes with letters excluded (service/postage codes)
    - Top 150 products by frequency (keeps matrix tractable)
    """
    try:
        # ── Build product description map ─────────────────────────────────────
        product_map = (
            df.dropna(subset=[COL_DESCRIPTION])
            .groupby(COL_STOCK_CODE)[COL_DESCRIPTION]
            .agg(lambda x: x.mode()[0])   # most common description per SKU
            .to_dict()
        )

        # ── Filter to top N products by frequency ─────────────────────────────
        top_products = (
            df[COL_STOCK_CODE]
            .value_counts()
            .head(150)
            .index.tolist()
        )
        df_filtered = df[df[COL_STOCK_CODE].isin(top_products)].copy()

        logging.info(
            f"Filtered to top 150 products — "
            f"{len(df_filtered):,} rows | "
            f"{df_filtered[COL_INVOICE].nunique():,} invoices"
        )

        # ── Build basket: group SKUs per invoice ──────────────────────────────
        basket_sets = (
            df_filtered
            .groupby(COL_INVOICE)[COL_STOCK_CODE]
            .apply(list)
            .tolist()
        )

        # ── TransactionEncoder → binary DataFrame ──────────────────────────────
        te       = TransactionEncoder()
        te_array = te.fit_transform(basket_sets)
        basket   = pd.DataFrame(te_array, columns=te.columns_)

        # Remove single-item baskets (can't generate pair rules)
        basket = basket[basket.sum(axis=1) > 1]

        logging.info(
            f"Basket matrix built — "
            f"shape: {basket.shape} | "
            f"density: {basket.values.mean()*100:.2f}%"
        )

        return basket, product_map

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 3: Run FP-Growth ────────────────────────────────────────────────────

def _run_fpgrowth(basket: pd.DataFrame) -> pd.DataFrame:
    """
    Run FP-Growth algorithm to find frequent itemsets.
    FP-Growth is used over Apriori because it doesn't generate
    candidate itemsets — significantly faster on large sparse matrices.
    """
    try:
        itemsets = fpgrowth(
            basket,
            min_support     = MIN_SUPPORT,
            use_colnames    = True,
            max_len         = 3,    # pairs and triplets only
        )

        itemsets = itemsets.sort_values("support", ascending=False)

        logging.info(
            f"FP-Growth complete — "
            f"{len(itemsets):,} frequent itemsets | "
            f"top support: {itemsets['support'].max():.4f}"
        )

        if itemsets.empty:
            logging.warning(
                f"No frequent itemsets found at min_support={MIN_SUPPORT}. "
                f"Try lowering MIN_SUPPORT."
            )

        return itemsets

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 4: Generate association rules ───────────────────────────────────────

def _generate_rules(
    itemsets: pd.DataFrame,
    product_map: dict,
) -> pd.DataFrame:
    """
    Generate and filter association rules from frequent itemsets.

    Adds human-readable antecedent/consequent descriptions
    using the product_map from Step 2.
    """
    try:
        if itemsets.empty:
            logging.warning("No itemsets to generate rules from — returning empty DataFrame")
            return pd.DataFrame()

        # ── Generate raw rules ────────────────────────────────────────────────
        rules = association_rules(
            itemsets,
            metric    = "lift",
            min_threshold = MIN_LIFT,
        )

        # ── Apply confidence filter ───────────────────────────────────────────
        rules = rules[rules["confidence"] >= MIN_CONFIDENCE].copy()

        if rules.empty:
            logging.warning(
                f"No rules passed confidence={MIN_CONFIDENCE} + lift={MIN_LIFT} filters"
            )
            return rules

        # ── Sort by lift descending ───────────────────────────────────────────
        rules = rules.sort_values("lift", ascending=False)

        # ── Cap to MAX_RULES ──────────────────────────────────────────────────
        if len(rules) > MAX_RULES:
            logging.info(f"Capping rules from {len(rules):,} to {MAX_RULES:,}")
            rules = rules.head(MAX_RULES)

        # ── Convert frozensets to readable strings ────────────────────────────
        rules["antecedents_codes"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        rules["consequents_codes"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        rules["antecedents_names"] = rules["antecedents"].apply(
            lambda x: " + ".join(
                product_map.get(code, code) for code in sorted(x)
            )
        )
        rules["consequents_names"] = rules["consequents"].apply(
            lambda x: " + ".join(
                product_map.get(code, code) for code in sorted(x)
            )
        )

        # ── Round metrics ─────────────────────────────────────────────────────
        for col in ["support", "confidence", "lift",
                    "leverage", "conviction"]:
            if col in rules.columns:
                rules[col] = rules[col].round(4)

        # ── Drop frozenset columns (not Postgres-serialisable) ────────────────
        rules = rules.drop(columns=["antecedents", "consequents"])

        logging.info(
            f"Rules generated — "
            f"{len(rules):,} rules | "
            f"max lift: {rules['lift'].max():.4f} | "
            f"max confidence: {rules['confidence'].max():.4f}"
        )

        # Log top 5 rules
        logging.info("Top 5 rules by lift:")
        for _, row in rules.head(5).iterrows():
            logging.info(
                f"  [{row['antecedents_names']}] → "
                f"[{row['consequents_names']}] "
                f"lift={row['lift']:.3f} conf={row['confidence']:.3f}"
            )

        return rules

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 5a: Log to MLflow ───────────────────────────────────────────────────

def _log_mlflow(
    basket: pd.DataFrame,
    itemsets: pd.DataFrame,
    rules: pd.DataFrame,
) -> None:
    try:
        mlflow.log_params({
            "min_support"    : MIN_SUPPORT,
            "min_confidence" : MIN_CONFIDENCE,
            "min_lift"       : MIN_LIFT,
            "max_rules"      : MAX_RULES,
            "n_products"     : basket.shape[1],
            "algorithm"      : "FP-Growth",
        })

        metrics = {
            "n_baskets"   : len(basket),
            "n_itemsets"  : len(itemsets),
            "n_rules"     : len(rules),
        }

        if not rules.empty:
            metrics.update({
                "max_lift"       : round(float(rules["lift"].max()), 4),
                "mean_lift"      : round(float(rules["lift"].mean()), 4),
                "max_confidence" : round(float(rules["confidence"].max()), 4),
                "mean_confidence": round(float(rules["confidence"].mean()), 4),
                "max_support"    : round(float(rules["support"].max()), 4),
            })

        mlflow.log_metrics(metrics)
        logging.info("MLflow logging complete ✓")

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 5b: Save rules to Postgres ─────────────────────────────────────────

def _save_rules(rules: pd.DataFrame, engine) -> None:
    try:
        if rules.empty:
            logging.warning("No rules to save — skipping DB write")
            return

        table      = "association_rules"
        full_table = f"{DB_SCHEMA}.{table}"

        rules.to_sql(
            name      = table,
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

        if actual != len(rules):
            raise ValueError(
                f"Row count mismatch: expected {len(rules):,}, got {actual:,}"
            )

        logging.info(
            f"Association rules saved ✓ — "
            f"{actual:,} rows → {full_table}"
        )

    except Exception as e:
        raise CSRException(e, sys)


if __name__ == "__main__":
    run_market_basket_pipeline()