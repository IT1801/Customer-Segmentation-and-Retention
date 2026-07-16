"""
Individual Customer Lookup
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sidebar import render_sidebar
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customer Lookup", page_icon="🔍", layout="wide")
render_sidebar()

st.title("🔍 Customer Lookup")
st.markdown("Full profile for any individual customer.")
st.divider()
if "master" not in st.session_state:
    st.switch_page("app.py")
master = st.session_state["master"]

transactions = st.session_state["transactions"]

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
