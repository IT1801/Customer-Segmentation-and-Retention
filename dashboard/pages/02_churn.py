"""
dashboard/pages/02_churn.py
Churn Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Churn", page_icon="⚠️", layout="wide")
st.title("⚠️ Churn Analysis")
st.markdown("Churn probability and risk tiers across customer segments.")
st.divider()

master = st.session_state["master"]

# ── KPI row ───────────────────────────────────────────────────────────────────
total     = len(master)
churned   = master["Churned"].sum()
high_risk = (master["ChurnRisk"] == "High").sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers",     f"{total:,}")
k2.metric("Churned",             f"{churned:,}", f"{churned/total*100:.1f}%")
k3.metric("High Risk",           f"{high_risk:,}", f"{high_risk/total*100:.1f}%")
k4.metric("Avg Churn Prob",      f"{master['ChurnProb'].mean()*100:.1f}%")

st.divider()

# ── Risk tier breakdown ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Risk Distribution")
    risk_order = ["Low","Medium","High"]
    risk_df    = (
        master["ChurnRisk"]
        .value_counts()
        .reindex(risk_order)
        .reset_index()
    )
    risk_df.columns = ["Risk","Count"]
    fig = px.bar(
        risk_df, x="Risk", y="Count",
        color="Risk",
        color_discrete_map={"Low":"#2ecc71","Medium":"#f39c12","High":"#e74c3c"},
        title="Customers by Churn Risk Tier",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Churn Probability Distribution")
    fig2 = px.histogram(
        master, x="ChurnProb",
        nbins     = 50,
        color_discrete_sequence=["#e74c3c"],
        title="Distribution of Churn Probability",
        labels={"ChurnProb": "Churn Probability"},
    )
    fig2.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Medium threshold")
    fig2.add_vline(x=0.6, line_dash="dash", line_color="red",    annotation_text="High threshold")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Churn by segment ──────────────────────────────────────────────────────────
st.subheader("Churn Rate by Segment")

churn_seg = (
    master.groupby("SegmentLabel")
    .agg(
        Customers  = ("CustomerID",  "count"),
        Churned    = ("Churned",     "sum"),
        Avg_Prob   = ("ChurnProb",   "mean"),
    )
    .reset_index()
)
churn_seg["Churn_Rate"] = (churn_seg["Churned"] / churn_seg["Customers"] * 100).round(1)
churn_seg["Avg_Prob"]   = (churn_seg["Avg_Prob"] * 100).round(1)

fig3 = px.bar(
    churn_seg,
    x     = "SegmentLabel",
    y     = "Churn_Rate",
    color = "SegmentLabel",
    text  = "Churn_Rate",
    title = "Churn Rate (%) by Segment",
    labels= {"Churn_Rate":"Churn Rate (%)"},
)
fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── High risk customer table ───────────────────────────────────────────────────
st.subheader("🔴 High Risk Customers")
st.caption("Customers with churn probability > 60% — prioritise for retention campaigns")

high_risk_df = (
    master[master["ChurnRisk"] == "High"]
    [[
        "CustomerID","SegmentLabel",
        "Recency","Frequency","Monetary",
        "ChurnProb","CLVTier","PredictedCLV_12M"
    ]]
    .sort_values("ChurnProb", ascending=False)
    .head(100)
)
high_risk_df["ChurnProb"]        = (high_risk_df["ChurnProb"] * 100).round(1)
high_risk_df["PredictedCLV_12M"] = high_risk_df["PredictedCLV_12M"].round(0)

st.dataframe(
    high_risk_df.rename(columns={
        "SegmentLabel"     : "Segment",
        "ChurnProb"        : "Churn Prob (%)",
        "PredictedCLV_12M" : "12M CLV (£)",
    }),
    use_container_width=True,
    hide_index=True,
)

# ── Churn vs CLV scatter ──────────────────────────────────────────────────────
st.divider()
st.subheader("Churn Probability vs CLV — Risk Prioritisation")
st.caption("Top-right = highest priority: high value customers at risk of churning")

fig4 = px.scatter(
    master.sample(min(2000, len(master)), random_state=42),
    x          = "ChurnProb",
    y          = "PredictedCLV_12M",
    color      = "SegmentLabel",
    hover_data = ["CustomerID","Recency","Monetary"],
    opacity    = 0.6,
    labels     = {
        "ChurnProb"        : "Churn Probability",
        "PredictedCLV_12M" : "Predicted 12M CLV (£)",
    },
    title = "Churn Risk vs Customer Value",
)
fig4.add_vline(x=0.6, line_dash="dash", line_color="red",   annotation_text="High churn threshold")
fig4.add_hline(y=master["PredictedCLV_12M"].median(),
               line_dash="dash", line_color="orange", annotation_text="Median CLV")
st.plotly_chart(fig4, use_container_width=True)