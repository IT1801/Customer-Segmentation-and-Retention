"""
dashboard/pages/05_market_basket.py
Market Basket & Association Rules Explorer
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Market Basket", page_icon="🛒", layout="wide")
st.title("🛒 Market Basket Analysis")
st.markdown("Association rules from FP-Growth — products frequently bought together.")
st.divider()

rules = st.session_state["rules"]

if rules.empty:
    st.warning("No association rules found. Run `market_basket.py` first.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Rules",      f"{len(rules):,}")
k2.metric("Max Lift",         f"{rules['lift'].max():.2f}")
k3.metric("Max Confidence",   f"{rules['confidence'].max()*100:.1f}%")
k4.metric("Avg Support",      f"{rules['support'].mean()*100:.2f}%")

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
min_lift  = col1.slider("Min Lift",       1.0, float(rules["lift"].max()),       1.5, 0.1)
min_conf  = col2.slider("Min Confidence", 0.1, 1.0,                              0.3, 0.05)
top_n     = col3.slider("Show top N rules", 10, 100, 25)

filtered = rules[
    (rules["lift"]       >= min_lift) &
    (rules["confidence"] >= min_conf)
].sort_values("lift", ascending=False).head(top_n)

st.caption(f"Showing {len(filtered)} rules after filters")

# ── Rules table ───────────────────────────────────────────────────────────────
st.subheader("Association Rules")

display_cols = [
    "antecedents_names",
    "consequents_names",
    "support",
    "confidence",
    "lift",
]
display = filtered[display_cols].copy()
display["confidence"] = (display["confidence"] * 100).round(1)
display["support"]    = (display["support"]    * 100).round(2)

st.dataframe(
    display.rename(columns={
        "antecedents_names" : "If customer buys...",
        "consequents_names" : "They also buy...",
        "support"           : "Support (%)",
        "confidence"        : "Confidence (%)",
        "lift"              : "Lift",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Lift vs Confidence scatter ─────────────────────────────────────────────────
st.subheader("Lift vs Confidence")

fig = px.scatter(
    filtered,
    x          = "confidence",
    y          = "lift",
    size       = "support",
    hover_data = ["antecedents_names","consequents_names","support"],
    color      = "lift",
    color_continuous_scale="Viridis",
    title      = "Rules — Lift vs Confidence (bubble = support)",
    labels     = {"confidence":"Confidence","lift":"Lift"},
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Product recommendation tool ───────────────────────────────────────────────
st.subheader("🔍 Product Recommendation Tool")
st.caption("Enter a product name to see what customers also buy with it.")

search = st.text_input("Search product (partial name match)")

if search:
    mask = rules["antecedents_names"].str.contains(search, case=False, na=False)
    recs = (
        rules[mask]
        [["antecedents_names","consequents_names","confidence","lift"]]
        .sort_values("lift", ascending=False)
        .head(10)
    )
    if recs.empty:
        st.info(f"No rules found matching '{search}'")
    else:
        recs["confidence"] = (recs["confidence"] * 100).round(1)
        st.dataframe(
            recs.rename(columns={
                "antecedents_names" : "If customer buys...",
                "consequents_names" : "They also buy...",
                "confidence"        : "Confidence (%)",
                "lift"              : "Lift",
            }),
            use_container_width=True,
            hide_index=True,
        )