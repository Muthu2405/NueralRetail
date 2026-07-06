"""Page 2 — Sales Analytics.

Monthly revenue trend and top-10 products by revenue. Both charts
respect the sidebar country/date filters.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from neuralretail.dashboard import data as d
from neuralretail.dashboard.components import format_currency, format_int, section_header
from neuralretail.dashboard.theme import ACCENT

st.set_page_config(page_title="Sales Analytics · NeuralRetail", page_icon="📈", layout="wide")

st.title("📈 Sales Analytics")
st.caption("Monthly revenue trend and the top revenue-driving products.")

df = d.load_transactions()
df = d.filter_by_sidebar(df, date_col="InvoiceDate")

if df.empty:
    st.warning("No transactions match the current filter. Widen the country or date range.")
    st.stop()

# --- Monthly revenue trend ---
section_header("Monthly revenue trend")

# `ME` (month-end) is the modern pandas alias for the deprecated `M`.
monthly = (
    df.assign(Month=pd.to_datetime(df["InvoiceDate"]).dt.to_period("M").dt.to_timestamp())
    .groupby("Month", as_index=False)["TotalPrice"]
    .sum()
    .rename(columns={"TotalPrice": "Revenue"})
    .sort_values("Month")
)

fig_trend = px.line(
    monthly,
    x="Month",
    y="Revenue",
    markers=True,
    color_discrete_sequence=[ACCENT],
)
fig_trend.update_traces(line=dict(width=3), marker=dict(size=7))
fig_trend.update_layout(
    xaxis_title=None,
    yaxis_title="Revenue",
    margin=dict(t=10, b=10, l=10, r=10),
    height=380,
    hovermode="x unified",
)
st.plotly_chart(fig_trend, width="stretch")

# --- Top-10 products ---
section_header("Top 10 products by revenue")

top_n = st.slider("How many products to show", min_value=5, max_value=20, value=10, step=1)

top_products = (
    df.groupby(["StockCode", "Description"], as_index=False)["TotalPrice"]
    .sum()
    .rename(columns={"TotalPrice": "Revenue"})
    .sort_values("Revenue", ascending=False)
    .head(top_n)
    .sort_values("Revenue")  # for horizontal bar, ascending so the largest sits on top
)

fig_top = px.bar(
    top_products,
    x="Revenue",
    y="Description",
    orientation="h",
    color="Revenue",
    color_continuous_scale=[ACCENT, ACCENT],
    text=top_products["Revenue"].map(lambda v: format_currency(v)),
)
fig_top.update_traces(textposition="outside", cliponaxis=False)
fig_top.update_layout(
    coloraxis_showscale=False,
    xaxis_title="Revenue",
    yaxis_title=None,
    margin=dict(t=10, b=10, l=10, r=10),
    height=max(320, 32 * len(top_products)),
)
st.plotly_chart(fig_top, width="stretch")

# --- Detail table ---
with st.expander("View product detail table"):
    detail = top_products.sort_values("Revenue", ascending=False).reset_index(drop=True)
    detail["Share %"] = (detail["Revenue"] / detail["Revenue"].sum() * 100).round(2)
    st.dataframe(
        detail.style.format({"Revenue": "{:,.2f}", "Share %": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
