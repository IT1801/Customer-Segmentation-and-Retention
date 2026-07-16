import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("📊 CSR Dashboard")
        st.caption("Customer Segmentation & Retention")
        st.divider()

        st.markdown("### Navigation")
        st.page_link("app.py",                        label="Overview",           icon="🏠")
        st.page_link("pages/01_segments.py",           label="Segments",           icon="🎯")
        st.page_link("pages/02_churn.py",              label="Churn Analysis",     icon="⚠️")
        st.page_link("pages/03_customer_lookup.py",    label="Customer Lookup",    icon="🔍")

        if "master" in st.session_state and "transactions" in st.session_state:
            st.divider()
            master = st.session_state["master"]
            transactions = st.session_state["transactions"]
            st.caption(f"Customers: **{len(master):,}**")

            start = transactions["InvoiceDate"].min().strftime("%Y-%m-%d")
            end = transactions["InvoiceDate"].max().strftime("%Y-%m-%d")

            st.caption(f"Date range: **{start}** → **{end}**")
