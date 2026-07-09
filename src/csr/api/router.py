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
    BasketRequest,
    BasketResponse,
    ChurnRequest,
    ChurnResponse,
    CLVRequest,
    CLVResponse,
    CustomerProfileResponse,
    HealthResponse,
    RecommendedProduct,
    SegmentRequest,
    SegmentResponse,
)
from csr.constants import (
    CHURN_RISK_BINS,
    CHURN_RISK_LABELS,
    CLV_DISCOUNT_RATE,
    CLV_HORIZON_MONTHS,
    COL_CUSTOMER_ID,
    DB_SCHEMA,
    SEGMENT_LABELS,
    TABLE_CHURN_PREDICTIONS,
    TABLE_CLV_PREDICTIONS,
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
            "bgf"         : model_store.bgf is not None,
            "ggf"         : model_store.ggf is not None,
            "rules"       : model_store.rules is not None,
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


# ─── CLV Prediction ───────────────────────────────────────────────────────────

@router.post("/predict/clv", response_model=CLVResponse, tags=["CLV"])
def predict_clv(
    body        : CLVRequest,
    model_store = Depends(get_model_store),
):
    """
    Predict 12-month CLV for a customer using BG/NBD + Gamma-Gamma.

    Input  : frequency, recency, T (customer age), monetary_value
    Output : predicted CLV, CLV tier, P(alive), expected purchases
    """
    try:
        if model_store.bgf is None or model_store.ggf is None:
            raise HTTPException(status_code=503, detail="CLV models not loaded")

        import pandas as pd

        summary = pd.DataFrame([{
            "frequency"      : body.frequency,
            "recency"        : body.recency,
            "T"              : body.T,
            "monetary_value" : body.monetary_value,
        }])

        # Probability alive
        prob_alive = float(
            model_store.bgf.conditional_probability_alive(
                summary["frequency"],
                summary["recency"],
                summary["T"],
            ).iloc[0]
        )

        # Expected purchases next 90 days
        expected_purchases = float(
            model_store.bgf.conditional_expected_number_of_purchases_up_to_time(
                90,
                summary["frequency"],
                summary["recency"],
                summary["T"],
            ).iloc[0]
        )

        # CLV — only for repeat buyers
        if body.frequency > 0:
            predicted_clv = float(
                model_store.ggf.customer_lifetime_value(
                    model_store.bgf,
                    summary["frequency"],
                    summary["recency"],
                    summary["T"],
                    summary["monetary_value"],
                    time          = CLV_HORIZON_MONTHS,
                    discount_rate = CLV_DISCOUNT_RATE,
                    freq          = "D",
                ).iloc[0]
            )
        else:
            predicted_clv = 0.0

        # CLV tier based on predicted value
        if predicted_clv == 0:
            clv_tier = "Bronze"
        elif predicted_clv < 500:
            clv_tier = "Silver"
        elif predicted_clv < 2000:
            clv_tier = "Gold"
        else:
            clv_tier = "Platinum"

        logging.info(
            f"CLV predicted — customer: {body.customer_id} | "
            f"CLV: £{predicted_clv:.2f} | tier: {clv_tier} | "
            f"P(alive): {prob_alive:.4f}"
        )

        return CLVResponse(
            customer_id            = body.customer_id,
            predicted_clv_12m      = round(predicted_clv, 2),
            clv_tier               = clv_tier,
            prob_alive             = round(prob_alive, 4),
            expected_purchases_90d = round(expected_purchases, 4),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(CSRException(e, sys)))


# ─── Market Basket ────────────────────────────────────────────────────────────

@router.post("/predict/basket", response_model=BasketResponse, tags=["Market Basket"])
def predict_basket(
    body        : BasketRequest,
    model_store = Depends(get_model_store),
):
    """
    Recommend products based on basket contents using association rules.

    Input  : List of StockCodes currently in basket
    Output : Ranked product recommendations with confidence and lift
    """
    try:
        if model_store.rules is None or model_store.rules.empty:
            raise HTTPException(status_code=503, detail="Association rules not loaded")

        rules      = model_store.rules
        input_set  = set(body.stock_codes)
        matches    = []

        for _, row in rules.iterrows():
            antecedent_codes = set(row["antecedents_codes"].split(", "))

            # Rule fires if antecedent is a subset of the current basket
            if antecedent_codes.issubset(input_set):
                consequent_codes = row["consequents_codes"].split(", ")

                for code in consequent_codes:
                    # Don't recommend items already in basket
                    if code not in input_set:
                        matches.append({
                            "stock_code" : code,
                            "description": row["consequents_names"],
                            "confidence" : row["confidence"],
                            "lift"       : row["lift"],
                        })

        if not matches:
            logging.info(
                f"No rules matched for basket: {body.stock_codes}"
            )
            return BasketResponse(
                input_codes     = body.stock_codes,
                recommendations = [],
            )

        # Deduplicate by stock_code, keep highest-lift match
        recs_df = (
            pd.DataFrame(matches)
            .sort_values("lift", ascending=False)
            .drop_duplicates(subset="stock_code")
            .head(body.top_n)
        )

        recommendations = [
            RecommendedProduct(
                stock_code  = row["stock_code"],
                description = row["description"],
                confidence  = round(row["confidence"], 4),
                lift        = round(row["lift"], 4),
            )
            for _, row in recs_df.iterrows()
        ]

        logging.info(
            f"Basket recommendations — "
            f"input: {body.stock_codes} | "
            f"recs: {[r.stock_code for r in recommendations]}"
        )

        return BasketResponse(
            input_codes     = body.stock_codes,
            recommendations = recommendations,
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
                c."ChurnRisk"       AS churn_risk,
                v."PredictedCLV_12M" AS predicted_clv_12m,
                v."CLVTier"         AS clv_tier,
                v."ProbAlive"       AS prob_alive
            FROM {DB_SCHEMA}.{TABLE_SEGMENT_RESULTS} s
            LEFT JOIN {DB_SCHEMA}.{TABLE_CHURN_PREDICTIONS} c
                ON s."{COL_CUSTOMER_ID}" = c."{COL_CUSTOMER_ID}"
            LEFT JOIN {DB_SCHEMA}.{TABLE_CLV_PREDICTIONS} v
                ON s."{COL_CUSTOMER_ID}" = v."{COL_CUSTOMER_ID}"
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