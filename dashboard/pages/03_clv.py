"""
dashboard/pages/03_clv.py
CLV Forecast
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CLV", page_icon="💰", layout="wide")
st.title("💰 Customer Lifetime Value")
st.markdown("12-month CLV predictions from BG/NBD + Gamma-Gamma models.")
st.divider()

master = st.session_state["master"]
clv    = st.session_state["clv"]

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Predicted CLV",  f"£{clv['PredictedCLV_12M'].sum():,.0f}")
k2.metric("Median CLV",           f"£{clv['PredictedCLV_12M'].median():,.0f}")
k3.metric("Avg P(Alive)",         f"{clv['ProbAlive'].mean()*100:.1f}%")
k4.metric("Avg Expected Purchases (90D)", f"{clv['ExpectedPurchases_90D'].mean():.2f}")

st.divider()

# ── CLV distribution ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("CLV Tier Distribution")
    tier_order = ["Bronze","Silver","Gold","Platinum"]
    tier_df    = (
        clv["CLVTier"]
        .value_counts()
        .reindex(tier_order)
        .reset_index()
    )
    tier_df.columns = ["Tier","Customers"]
    fig = px.bar(
        tier_df, x="Tier", y="Customers",
        color="Tier",
        color_discrete_map={
            "Bronze":"#cd7f32",
            "Silver":"#c0c0c0",
            "Gold":"#ffd700",
            "Platinum":"#e5e4e2",
        },
        title="Customers per CLV Tier",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("CLV Distribution")
    clv_clipped = clv[clv["PredictedCLV_12M"] > 0].copy()
    fig2 = px.histogram(
        clv_clipped,
        x      = "PredictedCLV_12M",
        nbins  = 60,
        color_discrete_sequence=["#f39c12"],
        title  = "Predicted 12M CLV (repeat buyers only)",
        labels = {"PredictedCLV_12M":"CLV (£)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── CLV by segment ────────────────────────────────────────────────────────────
st.subheader("CLV by Segment")

clv_seg = (
    master.groupby("SegmentLabel")
    .agg(
        Customers    = ("CustomerID",        "count"),
        Median_CLV   = ("PredictedCLV_12M",  "median"),
        Mean_CLV     = ("PredictedCLV_12M",  "mean"),
        Total_CLV    = ("PredictedCLV_12M",  "sum"),
        Avg_PAlive   = ("ProbAlive",         "mean"),
    )
    .reset_index()
)
for col in ["Median_CLV","Mean_CLV","Total_CLV","Avg_PAlive"]:
    clv_seg[col] = clv_seg[col].round(2)

fig3 = px.bar(
    clv_seg,
    x     = "SegmentLabel",
    y     = "Mean_CLV",
    color = "SegmentLabel",
    text  = "Mean_CLV",
    title = "Mean Predicted 12M CLV by Segment (£)",
    labels= {"Mean_CLV":"Mean CLV (£)"},
)
fig3.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
st.plotly_chart(fig3, use_container_width=True)

st.dataframe(
    clv_seg.rename(columns={
        "SegmentLabel" : "Segment",
        "Median_CLV"   : "Median CLV (£)",
        "Mean_CLV"     : "Mean CLV (£)",
        "Total_CLV"    : "Total CLV (£)",
        "Avg_PAlive"   : "Avg P(Alive)",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── P(Alive) heatmap ──────────────────────────────────────────────────────────
st.subheader("Probability Alive vs Expected Purchases")
st.caption("Customers in the top-right quadrant are your most engaged active buyers")

fig4 = px.scatter(
    clv.sample(min(2000, len(clv)), random_state=42),
    x          = "ProbAlive",
    y          = "ExpectedPurchases_90D",
    color      = "CLVTier",
    opacity    = 0.5,
    color_discrete_map={
        "Bronze":"#cd7f32",
        "Silver":"#c0c0c0",
        "Gold":"#ffd700",
        "Platinum":"#a0a0a0",
    },
    labels={
        "ProbAlive"              : "Probability Alive",
        "ExpectedPurchases_90D"  : "Expected Purchases (90 days)",
    },
    title="P(Alive) vs Expected Purchases by CLV Tier",
)
fig4.add_vline(x=0.5, line_dash="dash", line_color="grey")
fig4.add_hline(y=clv["ExpectedPurchases_90D"].median(), line_dash="dash", line_color="grey")
st.plotly_chart(fig4, use_container_width=True)

# ── Top customers by CLV ──────────────────────────────────────────────────────
st.divider()
st.subheader("🏆 Top 50 Customers by Predicted CLV")

top_clv = (
    master[["CustomerID","SegmentLabel","PredictedCLV_12M","CLVTier",
            "ProbAlive","Monetary","ChurnRisk"]]
    .sort_values("PredictedCLV_12M", ascending=False)
    .head(50)
)
top_clv["PredictedCLV_12M"] = top_clv["PredictedCLV_12M"].round(0)
top_clv["ProbAlive"]        = (top_clv["ProbAlive"] * 100).round(1)

st.dataframe(
    top_clv.rename(columns={
        "SegmentLabel"     : "Segment",
        "PredictedCLV_12M" : "12M CLV (£)",
        "ProbAlive"        : "P(Alive) %",
        "Monetary"         : "Historic Revenue (£)",
        "ChurnRisk"        : "Churn Risk",
    }),
    use_container_width=True,
    hide_index=True,
)