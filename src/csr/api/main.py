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
    CHURN_MODEL_ARTIFACT,
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


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Customer Segmentation & Retention API",
        description = (
            "Serves customer segmentation and churn prediction "
            "from the Online Retail II dataset."
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



# ─── App instance ─────────────────────────────────────────────────────────────

app = create_app()