"""Page 1 — Executive Overview.

KPI cards across the top (Total Revenue, Orders, Customers, AOV)
plus a revenue-by-country bar chart. Reads the sidebar filters.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from neuralretail.dashboard import data as d
from neuralretail.dashboard.components import format_currency, format_int, kpi_card, section_header
from neuralretail.dashboard.theme import ACCENT, ACCENT_DARK, ACCENT_LIGHT

st.set_page_config(page_title="Executive Overview · NeuralRetail", page_icon="📊", layout="wide")

st.title("📊 Executive Overview")
st.caption("Headline KPIs and revenue distribution by country.")

# --- Data ---
df = d.load_transactions()
df = d.filter_by_sidebar(df, date_col="InvoiceDate")

if df.empty:
    st.warning("No transactions match the current filter. Widen the country or date range.")
    st.stop()

# --- KPIs ---
total_revenue = float(df["TotalPrice"].sum())
n_orders = int(df["InvoiceNo"].nunique())
n_customers = int(df["CustomerID"].nunique())
aov = total_revenue / n_orders if n_orders else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Total Revenue", format_currency(total_revenue))
with c2:
    kpi_card("Orders", format_int(n_orders))
with c3:
    kpi_card("Customers", format_int(n_customers))
with c4:
    kpi_card("Avg Order Value", format_currency(aov))

st.write("")  # breathing room

# --- Revenue by country bar chart ---
section_header("Revenue by country")

by_country = (
    df.groupby("Country", as_index=False)["TotalPrice"]
    .sum()
    .rename(columns={"TotalPrice": "Revenue"})
    .sort_values("Revenue", ascending=False)
)

fig = px.bar(
    by_country,
    x="Country",
    y="Revenue",
    color="Revenue",
    color_continuous_scale=[ACCENT_LIGHT, ACCENT, ACCENT_DARK],
    text=by_country["Revenue"].map(lambda v: format_currency(v)),
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(
    coloraxis_showscale=False,
    xaxis_title=None,
    yaxis_title="Revenue",
    margin=dict(t=20, b=20, l=10, r=10),
    height=420,
)
st.plotly_chart(fig, width="stretch")

# --- Country detail table ---
with st.expander("View country detail table"):
    by_country["Share %"] = (by_country["Revenue"] / total_revenue * 100).round(2)
    st.dataframe(
        by_country.style.format({"Revenue": "{:,.2f}", "Share %": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
