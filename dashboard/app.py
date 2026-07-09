"""
dashboard/app.py

Streamlit main entry point.
Handles DB connection, shared state, and sidebar navigation.

Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# ── Make src importable from dashboard/ ───────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.csr.config.configuration import ConfigurationManager
from src.csr.etl.load import get_engine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Customer Segmentation & Retention",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ── DB engine — cached so it's built once per session ─────────────────────────
@st.cache_resource
def load_engine():
    cfg = ConfigurationManager()
    return get_engine(db_config=cfg.get_database_config())


# ── Data loaders — cached with TTL so dashboard stays fresh ───────────────────
@st.cache_data(ttl=600)
def load_segments(_engine) -> pd.DataFrame:
    with _engine.connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM retail.segment_results"),
            conn,
        )

@st.cache_data(ttl=600)
def load_churn(_engine) -> pd.DataFrame:
    with _engine.connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM retail.churn_predictions"),
            conn,
        )

@st.cache_data(ttl=600)
def load_clv(_engine) -> pd.DataFrame:
    with _engine.connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM retail.clv_predictions"),
            conn,
        )

@st.cache_data(ttl=600)
def load_transactions(_engine) -> pd.DataFrame:
    with _engine.connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM retail.cleaned_transactions"),
            conn,
        )

@st.cache_data(ttl=600)
def load_rules(_engine) -> pd.DataFrame:
    with _engine.connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM retail.association_rules"),
            conn,
        )


# ── Load data ─────────────────────────────────────────────────────────────────
engine = load_engine()

with st.spinner("Loading data..."):
    segments     = load_segments(engine)
    churn        = load_churn(engine)
    clv          = load_clv(engine)
    transactions = load_transactions(engine)
    rules        = load_rules(engine)

# Merge into one master DataFrame for convenience
master = (
    segments
    .merge(churn[["CustomerID","ChurnProb","ChurnRisk","Churned"]], on="CustomerID", how="left")
    .merge(clv[["CustomerID","PredictedCLV_12M","CLVTier","ProbAlive"]], on="CustomerID", how="left")
)

# Store in session state so pages can access without reloading
st.session_state["master"]       = master
st.session_state["segments"]     = segments
st.session_state["churn"]        = churn
st.session_state["clv"]          = clv
st.session_state["transactions"] = transactions
st.session_state["rules"]        = rules
st.session_state["engine"]       = engine


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 CSR Dashboard")
    st.caption("Customer Segmentation & Retention")
    st.divider()

    st.markdown("### Navigation")
    st.page_link("app.py",                        label="🏠 Overview",        icon="🏠")
    st.page_link("pages/01_segments.py",           label="Segments",           icon="🎯")
    st.page_link("pages/02_churn.py",              label="Churn Analysis",     icon="⚠️")
    st.page_link("pages/03_clv.py",                label="CLV Forecast",       icon="💰")
    st.page_link("pages/04_cohort.py",             label="Cohort Retention",   icon="📅")
    st.page_link("pages/05_market_basket.py",      label="Market Basket",      icon="🛒")
    st.page_link("pages/06_customer_lookup.py",    label="Customer Lookup",    icon="🔍")

    st.divider()
    st.caption(f"Customers: **{len(master):,}**")

    start = transactions["InvoiceDate"].min().strftime("%Y-%m-%d")
    end = transactions["InvoiceDate"].max().strftime("%Y-%m-%d")

    st.caption(f"Date range: **{start}** → **{end}**")


# ── Overview page ─────────────────────────────────────────────────────────────
st.title("🏠 Overview")
st.markdown("High-level KPIs across all customers.")
st.divider()

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    label = "Total Customers",
    value = f"{len(master):,}",
)
k2.metric(
    label = "Total Revenue",
    value = f"£{master['Monetary'].sum():,.0f}",
)
k3.metric(
    label = "Avg CLV (12M)",
    value = f"£{master['PredictedCLV_12M'].mean():,.0f}",
)
k4.metric(
    label = "Churn Rate",
    value = f"{master['Churned'].mean()*100:.1f}%",
)
k5.metric(
    label = "Avg P(Alive)",
    value = f"{master['ProbAlive'].mean()*100:.1f}%",
)

st.divider()

# ── Segment breakdown ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customers by Segment")
    seg_counts = master["SegmentLabel"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    st.bar_chart(seg_counts.set_index("Segment"))

with col2:
    st.subheader("Revenue by Segment")
    rev_by_seg = (
        master.groupby("SegmentLabel")["Monetary"]
        .sum()
        .reset_index()
        .rename(columns={"Monetary": "Revenue"})
    )
    st.bar_chart(rev_by_seg.set_index("SegmentLabel"))

st.divider()

# ── Churn risk + CLV tier ─────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Churn Risk Distribution")
    risk_counts = master["ChurnRisk"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Customers"]
    st.bar_chart(risk_counts.set_index("Risk"))

with col4:
    st.subheader("CLV Tier Distribution")
    clv_counts = master["CLVTier"].value_counts().reset_index()
    clv_counts.columns = ["Tier", "Customers"]
    st.bar_chart(clv_counts.set_index("Tier"))