"""
dashboard/pages/01_segments.py
Customer Segmentation Explorer
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sidebar import render_sidebar
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np

st.set_page_config(page_title="Segments", page_icon="🎯", layout="wide")
render_sidebar()

st.title("🎯 Customer Segments")
st.markdown("K-Means segmentation on RFM features.")
st.divider()

master   = st.session_state["master"]
segments = st.session_state["segments"]

# ── Segment filter ────────────────────────────────────────────────────────────
all_labels = sorted(master["SegmentLabel"].dropna().unique())
selected   = st.multiselect("Filter segments", all_labels, default=all_labels)
df = master[master["SegmentLabel"].isin(selected)]

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers",       f"{len(df):,}")
k2.metric("Avg Recency",     f"{df['Recency'].mean():.0f} days")
k3.metric("Avg Frequency",   f"{df['Frequency'].mean():.1f} orders")
k4.metric("Avg Monetary",    f"£{df['Monetary'].mean():,.0f}")

st.divider()

# ── Segment profile table ─────────────────────────────────────────────────────
st.subheader("Segment Profile")

profile = (
    df.groupby("SegmentLabel")
    .agg(
        Customers     = ("CustomerID",    "count"),
        Avg_Recency   = ("Recency",       "mean"),
        Avg_Frequency = ("Frequency",     "mean"),
        Avg_Monetary  = ("Monetary",      "mean"),
        Avg_AOV       = ("AOV",           "mean"),
        Revenue_Share = ("Monetary",      "sum"),
    )
    .reset_index()
)
profile["Revenue_Share"] = (
    profile["Revenue_Share"] / profile["Revenue_Share"].sum() * 100
).round(1)

for col in ["Avg_Recency","Avg_Frequency","Avg_Monetary","Avg_AOV"]:
    profile[col] = profile[col].round(1)

st.dataframe(
    profile.rename(columns={
        "SegmentLabel" : "Segment",
        "Avg_Recency"  : "Avg Recency (days)",
        "Avg_Frequency": "Avg Orders",
        "Avg_Monetary" : "Avg Revenue (£)",
        "Avg_AOV"      : "Avg Order Value (£)",
        "Revenue_Share": "Revenue Share (%)",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── PCA scatter ───────────────────────────────────────────────────────────────
st.subheader("RFM Clusters — PCA Projection")

rfm_cols = ["Recency","Log_Frequency","Log_Monetary"]
available = [c for c in rfm_cols if c in df.columns]

if len(available) == 3:
    X      = df[available].fillna(0).values
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    scatter_df = pd.DataFrame({
        "PC1"         : coords[:, 0],
        "PC2"         : coords[:, 1],
        "Segment"     : df["SegmentLabel"].values,
        "CustomerID"  : df["CustomerID"].values,
        "Recency"     : df["Recency"].values,
        "Frequency"   : df["Frequency"].values,
        "Monetary"    : df["Monetary"].values,
    })

    fig = px.scatter(
        scatter_df,
        x          = "PC1",
        y          = "PC2",
        color      = "Segment",
        hover_data = ["CustomerID","Recency","Frequency","Monetary"],
        opacity    = 0.6,
        title      = f"PCA — {pca.explained_variance_ratio_.sum()*100:.1f}% variance explained",
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── RFM box plots ─────────────────────────────────────────────────────────────
st.subheader("RFM Distribution by Segment")
metric = st.selectbox("Metric", ["Recency","Frequency","Monetary"])

fig2 = px.box(
    df,
    x      = "SegmentLabel",
    y      = metric,
    color  = "SegmentLabel",
    points = "outliers",
    title  = f"{metric} by Segment",
)
st.plotly_chart(fig2, use_container_width=True)