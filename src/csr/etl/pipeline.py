"""
Orchestrates the full ETL pipeline:
    Extract  → load raw .xlsx into DataFrame
    Transform → clean and validate
    Load     → persist to PostgreSQL (retail.cleaned_transactions)
"""

import sys
import time

from csr.logging.logger import logging
from csr.exception.exception import CSRException
from csr.config.configuration import ConfigurationManager
from csr.etl.extract import extract
from csr.etl.transform import transform
from csr.etl.load import get_engine, save_interim


def run_etl_pipeline() -> None:
    try:
        start = time.time()
        logging.info("=" * 60)
        logging.info("ETL PIPELINE STARTED")
        logging.info("=" * 60)

        # ── Step 0: Load config once, pass through to all steps ──────────────
        logging.info("Step 0/3 — Loading configuration")
        cfg        = ConfigurationManager()
        etl_config = cfg.get_etl_config()
        db_config  = cfg.get_database_config()
        logging.info("Configuration loaded ✓")

        # ── Step 1: Extract ───────────────────────────────────────────────────
        logging.info("Step 1/3 — Extract")
        t1      = time.time()
        df_raw  = extract(etl_config=etl_config)
        logging.info(
            f"Extract complete ✓ — "
            f"{len(df_raw):,} rows | "
            f"{time.time() - t1:.1f}s"
        )

        # ── Step 2: Transform ─────────────────────────────────────────────────
        logging.info("Step 2/3 — Transform")
        t2       = time.time()
        df_clean = transform(df_raw, etl_config=etl_config)
        logging.info(
            f"Transform complete ✓ — "
            f"{len(df_clean):,} rows retained "
            f"({len(df_clean)/len(df_raw)*100:.1f}% of raw) | "
            f"{time.time() - t2:.1f}s"
        )

        # ── Step 3: Load ──────────────────────────────────────────────────────
        logging.info("Step 3/3 — Load")
        t3     = time.time()
        engine = get_engine(db_config=db_config)
        save_interim(
            df        = df_clean,
            engine    = engine,
            chunksize = etl_config.chunksize,
        )
        logging.info(
            f"Load complete ✓ — "
            f"retail.cleaned_transactions | "
            f"{time.time() - t3:.1f}s"
        )

        # ── Summary ───────────────────────────────────────────────────────────
        elapsed = time.time() - start
        logging.info("=" * 60)
        logging.info("ETL PIPELINE COMPLETE")
        logging.info(f"  Raw rows       : {len(df_raw):,}")
        logging.info(f"  Cleaned rows   : {len(df_clean):,}")
        logging.info(f"  Rows dropped   : {len(df_raw) - len(df_clean):,}")
        logging.info(f"  Total time     : {elapsed:.1f}s")
        logging.info("=" * 60)

    except Exception as e:
        logging.error("ETL PIPELINE FAILED")
        raise CSRException(e, sys)


if __name__ == "__main__":
    run_etl_pipeline()