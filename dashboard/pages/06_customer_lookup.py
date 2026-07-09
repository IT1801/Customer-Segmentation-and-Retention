"""
Individual Customer Lookup
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customer Lookup", page_icon="🔍", layout="wide")
st.title("🔍 Customer Lookup")
st.markdown("Full profile for any individual customer.")
st.divider()
if "master" not in st.session_state:
    st.switch_page("app.py")
master = st.session_state["master"]

transactions = st.session_state["transactions"]
rules        = st.session_state["rules"]

# ── Search ────────────────────────────────────────────────────────────────────
customer_id = st.text_input(
    "Enter CustomerID",
    placeholder="e.g. 12747",
)

if not customer_id:
    st.info("Enter a CustomerID above to view their profile.")
    st.stop()

# ── Fetch customer ─────────────────────────────────────────────────────────────
row = master[master["CustomerID"] == customer_id]

if row.empty:
    st.error(f"CustomerID '{customer_id}' not found.")
    st.stop()

row = row.iloc[0]

# ── Profile header ─────────────────────────────────────────────────────────────
st.subheader(f"Customer {customer_id}")
col1, col2, col3 = st.columns(3)

col1.markdown(f"**Segment:** {row.get('SegmentLabel','—')}")
col1.markdown(f"**Cohort:** {row.get('CohortMonth','—')}")
col1.markdown(f"**Active months:** {row.get('ActiveMonths','—')}")

col2.markdown(f"**Churn risk:** {row.get('ChurnRisk','—')}")
col2.markdown(f"**Churn prob:** {row.get('ChurnProb',0)*100:.1f}%")
col2.markdown(f"**P(Alive):** {row.get('ProbAlive',0)*100:.1f}%")

col3.markdown(f"**CLV Tier:** {row.get('CLVTier','—')}")
col3.markdown(f"**12M CLV:** £{row.get('PredictedCLV_12M',0):,.0f}")
col3.markdown(f"**Historic revenue:** £{row.get('Monetary',0):,.2f}")

st.divider()

# ── RFM metrics ───────────────────────────────────────────────────────────────
st.subheader("RFM Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Recency",   f"{row.get('Recency',0):.0f} days")
m2.metric("Frequency", f"{row.get('Frequency',0):.0f} orders")
m3.metric("Monetary",  f"£{row.get('Monetary',0):,.2f}")
m4.metric("AOV",       f"£{row.get('AOV',0):,.2f}")

st.divider()

# ── Transaction history ───────────────────────────────────────────────────────
st.subheader("Transaction History")

cust_txns = (
    transactions[transactions["CustomerID"] == customer_id]
    .copy()
)
cust_txns["InvoiceDate"] = pd.to_datetime(cust_txns["InvoiceDate"])
cust_txns = cust_txns.sort_values("InvoiceDate", ascending=False)

st.caption(f"{len(cust_txns):,} line items across {cust_txns['InvoiceNo'].nunique()} orders")

# Monthly spend chart
monthly = (
    cust_txns.groupby(cust_txns["InvoiceDate"].dt.to_period("M").astype(str))["Revenue"]
    .sum()
    .reset_index()
    .rename(columns={"InvoiceDate":"Month","Revenue":"Revenue"})
)

fig = px.bar(
    monthly,
    x     = "Month",
    y     = "Revenue",
    title = f"Monthly Spend — Customer {customer_id}",
    color_discrete_sequence=["#3498db"],
    labels= {"Revenue":"Revenue (£)"},
)
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, use_container_width=True)

# Raw transactions table
with st.expander("View raw transactions"):
    st.dataframe(
        cust_txns[["InvoiceDate","InvoiceNo","StockCode","Description","Quantity","UnitPrice","Revenue"]]
        .rename(columns={"InvoiceDate":"Date"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Product recommendations ────────────────────────────────────────────────────
st.subheader("🛒 Product Recommendations")
st.caption("Based on this customer's purchase history and association rules")

if not rules.empty:
    bought_codes = set(cust_txns["StockCode"].unique())
    matches      = []

    for _, r in rules.iterrows():
        ant_codes = set(r["antecedents_codes"].split(", "))
        if ant_codes.issubset(bought_codes):
            con_codes = r["consequents_codes"].split(", ")
            for code in con_codes:
                if code not in bought_codes:
                    matches.append({
                        "StockCode"  : code,
                        "Description": r["consequents_names"],
                        "Confidence" : r["confidence"],
                        "Lift"       : r["lift"],
                    })

    if matches:
        recs = (
            pd.DataFrame(matches)
            .sort_values("Lift", ascending=False)
            .drop_duplicates("StockCode")
            .head(5)
        )
        recs["Confidence"] = (recs["Confidence"] * 100).round(1)
        st.dataframe(
            recs.rename(columns={"Confidence":"Confidence (%)"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No recommendations available for this customer's purchase history.")
else:
    st.info("Run market_basket.py to enable recommendations.")