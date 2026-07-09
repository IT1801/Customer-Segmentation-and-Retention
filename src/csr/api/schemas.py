"""
Pydantic request and response models for all API endpoints.
Every endpoint has a dedicated Input and Output schema —
no raw dicts cross the API boundary.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Segmentation ─────────────────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    recency: float   = Field(..., ge=0,   description="Days since last purchase")
    frequency: float = Field(..., ge=1,   description="Number of unique orders")
    monetary: float  = Field(..., gt=0,   description="Total revenue in GBP")

    @field_validator("recency")
    @classmethod
    def recency_not_negative(cls, v):
        if v < 0:
            raise ValueError("Recency cannot be negative")
        return v


class SegmentResponse(BaseModel):
    segment_id   : int
    segment_label: str
    recency      : float
    frequency    : float
    monetary     : float


# ─── Churn Prediction ─────────────────────────────────────────────────────────

class ChurnRequest(BaseModel):
    customer_id             : str
    recency                 : float = Field(..., ge=0)
    frequency               : float = Field(..., ge=1)
    monetary                : float = Field(..., gt=0)
    aov                     : float = Field(..., gt=0,   description="Average order value")
    spend_std               : float = Field(..., ge=0,   description="Std deviation of order value")
    unique_skus             : int   = Field(..., ge=1,   description="Number of unique SKUs purchased")
    total_items             : int   = Field(..., ge=1,   description="Total items purchased")
    repeat_sku_ratio        : float = Field(..., ge=0, le=1)
    avg_gap                 : float = Field(..., ge=0,   description="Average days between orders")
    std_gap                 : float = Field(..., ge=0,   description="Std dev of days between orders")
    weekend_ratio           : float = Field(..., ge=0, le=1)
    preferred_day_of_week   : int   = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    active_months           : int   = Field(..., ge=1)
    days_since_first_purchase: float = Field(..., ge=0)
    return_rate             : float = Field(..., ge=0, le=1)
    segment                 : int   = Field(..., ge=0,   description="Cluster label from segmentation")


class ChurnResponse(BaseModel):
    customer_id  : str
    churn_prob   : float = Field(..., description="Probability of churn (0–1)")
    churned      : int   = Field(..., description="Binary churn prediction (0 or 1)")
    churn_risk   : str   = Field(..., description="Low / Medium / High")


# ─── CLV Prediction ───────────────────────────────────────────────────────────

class CLVRequest(BaseModel):
    customer_id    : str
    frequency      : float = Field(..., ge=0, description="Number of repeat purchases")
    recency        : float = Field(..., ge=0, description="Days between first and last purchase")
    T              : float = Field(..., ge=0, description="Customer age in days")
    monetary_value : float = Field(..., ge=0, description="Mean revenue per transaction")


class CLVResponse(BaseModel):
    customer_id         : str
    predicted_clv_12m   : float = Field(..., description="Predicted 12-month CLV in GBP")
    clv_tier            : str   = Field(..., description="Bronze / Silver / Gold / Platinum")
    prob_alive          : float = Field(..., description="Probability customer is still active")
    expected_purchases_90d: float = Field(..., description="Expected purchases in next 90 days")


# ─── Customer Lookup ──────────────────────────────────────────────────────────

class CustomerProfileResponse(BaseModel):
    customer_id           : str
    # RFM
    recency               : float
    frequency             : float
    monetary              : float
    # Segment
    segment_id            : Optional[int]   = None
    segment_label         : Optional[str]   = None
    # Churn
    churn_prob            : Optional[float] = None
    churn_risk            : Optional[str]   = None
    # CLV
    predicted_clv_12m     : Optional[float] = None
    clv_tier              : Optional[str]   = None
    prob_alive            : Optional[float] = None
    # Cohort
    cohort_month          : Optional[str]   = None
    active_months         : Optional[int]   = None


# ─── Market Basket ────────────────────────────────────────────────────────────

class BasketRequest(BaseModel):
    stock_codes: List[str] = Field(
        ...,
        min_length=1,
        description="List of StockCodes currently in the basket"
    )
    top_n: int = Field(5, ge=1, le=20, description="Max recommendations to return")


class RecommendedProduct(BaseModel):
    stock_code  : str
    description : str
    confidence  : float
    lift        : float


class BasketResponse(BaseModel):
    input_codes     : List[str]
    recommendations : List[RecommendedProduct]


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status  : str
    version : str
    models_loaded: dict