"""
FastAPI application entry point.

Responsibilities:
    - Create the FastAPI app
    - Load all model artifacts once at startup into app.state
    - Load DB engine once at startup into app.state
    - Mount the router
    - Expose startup/shutdown lifecycle events

Run directly:
    uvicorn src.csr.api.main:app --host 0.0.0.0 --port 8000 --reload

Or via Makefile:
    make serve
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from sqlalchemy.engine import Engine
from sqlalchemy import text

from csr.exception.exception import CSRException
from csr.logging.logger import logging
from csr.api.router import router
from csr.config.configuration import ConfigurationManager
from csr.constants import (
    ARTIFACTS_DIR,
    BGF_ARTIFACT,
    CHURN_MODEL_ARTIFACT,
    DB_SCHEMA,
    GGF_ARTIFACT,
    KMEANS_ARTIFACT,
    SCALER_RFM_ARTIFACT,
)
from csr.etl.load import get_engine


# ─── Model store dataclass ────────────────────────────────────────────────────

@dataclass
class ModelStore:
    """
    Holds all loaded model artifacts.
    Loaded once at startup and injected into every request via app.state.
    """
    kmeans      : Optional[object] = None
    scaler      : Optional[object] = None
    churn_model : Optional[object] = None
    bgf         : Optional[object] = None
    ggf         : Optional[object] = None
    rules       : Optional[pd.DataFrame] = None


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Customer Segmentation & Retention API",
        description = (
            "Serves customer segmentation, churn prediction, CLV forecasting, "
            "and product recommendations from the Online Retail II dataset."
        ),
        version     = "0.1.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── Startup event ─────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup():
        try:
            logging.info("API startup — loading artifacts and DB engine...")

            cfg    = ConfigurationManager()
            db_cfg = cfg.get_database_config()

            # ── DB engine ─────────────────────────────────────────────────────
            engine = get_engine(db_config=db_cfg)
            app.state.engine = engine
            logging.info("DB engine ready ✓")

            # ── Model artifacts ───────────────────────────────────────────────
            store = ModelStore()
            store = _load_artifacts(store)
            store = _load_rules(store, engine)
            app.state.model_store = store

            logging.info("All artifacts loaded ✓ — API ready")

        except Exception as e:
            logging.error(f"Startup failed: {e}")
            raise CSRException(e, sys)

    # ── Shutdown event ────────────────────────────────────────────────────────
    @app.on_event("shutdown")
    async def shutdown():
        logging.info("API shutting down — releasing DB connections")
        if hasattr(app.state, "engine"):
            app.state.engine.dispose()

    # ── Mount router ──────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    return app


# ─── Load model artifacts ─────────────────────────────────────────────────────

def _load_artifacts(store: ModelStore) -> ModelStore:
    """
    Load all .joblib artifacts from the artifacts directory.
    Logs a warning (not a crash) for any missing artifact so the API
    can still serve the models that are available.
    """
    try:
        artifact_map = {
            "kmeans"     : KMEANS_ARTIFACT,
            "scaler"     : SCALER_RFM_ARTIFACT,
            "churn_model": CHURN_MODEL_ARTIFACT,
            "bgf"        : BGF_ARTIFACT,
            "ggf"        : GGF_ARTIFACT,
        }

        for attr, path in artifact_map.items():
            if path.exists():
                setattr(store, attr, joblib.load(path))
                logging.info(f"Loaded artifact: {path.name} ✓")
            else:
                logging.warning(
                    f"Artifact not found: {path} — "
                    f"/{attr} endpoints will return 503 until model is trained"
                )

        return store

    except Exception as e:
        raise CSRException(e, sys)


def _load_rules(store: ModelStore, engine: Engine) -> ModelStore:
    """
    Load association rules from PostgreSQL into memory.
    Kept in-memory for fast basket lookup without a DB round-trip per request.
    """
    try:
        rules_table = f"{DB_SCHEMA}.association_rules"

        with engine.connect() as conn:
            # Check table exists before querying
            exists = conn.execute(text(
                f"SELECT EXISTS ("
                f"SELECT FROM information_schema.tables "
                f"WHERE table_schema = '{DB_SCHEMA}' "
                f"AND table_name = 'association_rules'"
                f")"
            )).scalar()

        if exists:
            with engine.connect() as conn:
                store.rules = pd.read_sql(
                    text(f"SELECT * FROM {rules_table}"),
                    conn,
                )
            logging.info(
                f"Association rules loaded — "
                f"{len(store.rules):,} rules ✓"
            )
        else:
            logging.warning(
                "association_rules table not found in DB — "
                "/predict/basket will return 503 until market_basket.py is run"
            )

        return store

    except Exception as e:
        raise CSRException(e, sys)


# ─── App instance ─────────────────────────────────────────────────────────────

app = create_app()