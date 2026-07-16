"""
All API route handlers. Imports artifacts from the ModelStore
(loaded once at startup in main.py) and serves predictions.

Routes:
    GET  /health                        — liveness check
    POST /predict/segment               — RFM → segment label
    POST /predict/churn                 — features → churn probability
    POST /predict/clv                   — RFM-T → 12M CLV
    POST /predict/basket                — basket → product recommendations
    GET  /customer/{customer_id}        — full customer profile from DB
"""

import sys

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from csr.exception.exception import CSRException
from csr.logging.logger import logging
from csr.api.schemas import (
    ChurnRequest,
    ChurnResponse,
    CustomerProfileResponse,
    HealthResponse,
    SegmentRequest,
    SegmentResponse,
)
from csr.constants import (
    CHURN_RISK_BINS,
    CHURN_RISK_LABELS,
    COL_CUSTOMER_ID,
    DB_SCHEMA,
    SEGMENT_LABELS,
    TABLE_CHURN_PREDICTIONS,
    TABLE_SEGMENT_RESULTS,
)
from sqlalchemy import text

router = APIRouter()

APP_VERSION = "0.1.0"


# ─── Dependency: model store ──────────────────────────────────────────────────

def get_model_store(request: Request):
    """Retrieve the ModelStore injected at startup from app.state."""
    return request.app.state.model_store


def get_engine(request: Request):
    """Retrieve the DB engine injected at startup from app.state."""
    return request.app.state.engine


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health(model_store=Depends(get_model_store)):
    """Liveness check — confirms API is up and which models are loaded."""
    return HealthResponse(
        status        = "ok",
        version       = APP_VERSION,
        models_loaded = {
            "kmeans"      : model_store.kmeans is not None,
            "rfm_scaler"  : model_store.scaler is not None,
            "churn_xgb"   : model_store.churn_model is not None,
        },
    )


# ─── Segmentation ─────────────────────────────────────────────────────────────

@router.post("/predict/segment", response_model=SegmentResponse, tags=["Segmentation"])
def predict_segment(
    body         : SegmentRequest,
    model_store  = Depends(get_model_store),
):
    """
    Assign a customer to a segment based on RFM values.

    Input  : Recency, Frequency, Monetary
    Output : Segment ID and human-readable label
    """
    try:
        if model_store.kmeans is None or model_store.scaler is None:
            raise HTTPException(status_code=503, detail="Segmentation models not loaded")

        import numpy as np

        # Log-transform to match training distribution
        log_freq = np.log1p(body.frequency)
        log_mon  = np.log1p(body.monetary)

        X = np.array([[body.recency, log_freq, log_mon]])
        X_scaled = model_store.scaler.transform(X)

        segment_id    = int(model_store.kmeans.predict(X_scaled)[0])
        segment_label = SEGMENT_LABELS.get(segment_id, "Unknown")

        logging.info(
            f"Segment predicted — "
            f"R={body.recency} F={body.frequency} M={body.monetary} "
            f"→ {segment_label}"
        )

        return SegmentResponse(
            segment_id    = segment_id,
            segment_label = segment_label,
            recency       = body.recency,
            frequency     = body.frequency,
            monetary      = body.monetary,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(CSRException(e, sys)))


# ─── Churn Prediction ─────────────────────────────────────────────────────────

@router.post("/predict/churn", response_model=ChurnResponse, tags=["Churn"])
def predict_churn(
    body        : ChurnRequest,
    model_store = Depends(get_model_store),
):
    """
    Predict churn probability for a customer.

    Input  : Full feature vector (RFM + behavioural + segment)
    Output : Churn probability, binary label, risk tier
    """
    try:
        if model_store.churn_model is None:
            raise HTTPException(status_code=503, detail="Churn model not loaded")

        feature_values = {
            "Recency"                 : body.recency,
            "Frequency"               : body.frequency,
            "Monetary"                : body.monetary,
            "AOV"                     : body.aov,
            "SpendStd"                : body.spend_std,
            "UniqueSKUs"              : body.unique_skus,
            "TotalItems"              : body.total_items,
            "RepeatSKURatio"          : body.repeat_sku_ratio,
            "AvgGap"                  : body.avg_gap,
            "StdGap"                  : body.std_gap,
            "WeekendRatio"            : body.weekend_ratio,
            "PreferredDayOfWeek"      : body.preferred_day_of_week,
            "ActiveMonths"            : body.active_months,
            "DaysSinceFirstPurchase"  : body.days_since_first_purchase,
            "ReturnRate"              : body.return_rate,
            "Segment"                 : body.segment,
        }

        X = pd.DataFrame([feature_values])

        churn_prob = float(model_store.churn_model.predict_proba(X)[0][1])
        churned    = int(model_store.churn_model.predict(X)[0])
        churn_risk = pd.cut(
            [churn_prob],
            bins   = CHURN_RISK_BINS,
            labels = CHURN_RISK_LABELS,
        )[0]

        logging.info(
            f"Churn predicted — customer: {body.customer_id} | "
            f"prob: {churn_prob:.4f} | risk: {churn_risk}"
        )

        return ChurnResponse(
            customer_id = body.customer_id,
            churn_prob  = round(churn_prob, 4),
            churned     = churned,
            churn_risk  = str(churn_risk),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(CSRException(e, sys)))





# ─── Customer Profile ─────────────────────────────────────────────────────────

@router.get(
    "/customer/{customer_id}",
    response_model=CustomerProfileResponse,
    tags=["Customer"],
)
def get_customer_profile(
    customer_id : str,
    engine      = Depends(get_engine),
):
    """
    Return a full customer profile by joining segment, churn,
    and CLV prediction tables from PostgreSQL.
    """
    try:
        query = text(f"""
            SELECT
                s."{COL_CUSTOMER_ID}",
                s."Recency",
                s."Frequency",
                s."Monetary",
                s."Segment"         AS segment_id,
                s."SegmentLabel"    AS segment_label,
                s."CohortMonth"     AS cohort_month,
                s."ActiveMonths"    AS active_months,
                c."ChurnProb"       AS churn_prob,
                c."ChurnRisk"       AS churn_risk
            FROM {DB_SCHEMA}.{TABLE_SEGMENT_RESULTS} s
            LEFT JOIN {DB_SCHEMA}.{TABLE_CHURN_PREDICTIONS} c
                ON s."{COL_CUSTOMER_ID}" = c."{COL_CUSTOMER_ID}"
            WHERE s."{COL_CUSTOMER_ID}" = :customer_id
        """)

        with engine.connect() as conn:
            result = conn.execute(query, {"customer_id": customer_id})
            row    = result.mappings().first()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{customer_id}' not found"
            )

        return CustomerProfileResponse(**dict(row))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(CSRException(e, sys)))