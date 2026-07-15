"""Page 3 — Customer Hub.

RFM cluster scatter, persona distribution pie, and a per-persona
summary table. RFM is a snapshot at training time, so the sidebar
country/date filters don't apply here — we keep the page global.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from neuralretail.dashboard import data as d
from neuralretail.dashboard.components import format_int, section_header
from neuralretail.dashboard.theme import PALETTE

st.set_page_config(page_title="Customer Hub · NeuralRetail", page_icon="👥", layout="wide")

st.title("👥 Customer Hub")
st.caption("RFM clusters, personas, and segment distribution. Global view — filters don't apply.")

# --- Score RFM ---
rfm = d.load_rfm()
pipe = d.load_segmentation_model()
scored = d.score_rfm_clusters(rfm, pipe)

# Persona order — stable so the pie/donut and legend are consistent.
persona_order = ["Champions", "Loyal Customers", "Regular", "At Risk", "Hibernating"]
present = [p for p in persona_order if p in scored["persona"].unique()]
scored["persona"] = pd.Categorical(scored["persona"], categories=present, ordered=True)

# --- Headline numbers ---
total = len(scored)
n_clusters = scored["cluster"].nunique()
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Customers", format_int(total))
with c2:
    st.metric("Clusters", format_int(n_clusters))
with c3:
    st.metric("Personas", format_int(scored["persona"].nunique()))

st.write("")

# --- Scatter: Recency vs Monetary, coloured by persona ---
section_header("Recency vs Monetary (coloured by persona)")

fig_scatter = px.scatter(
    scored,
    x="Recency",
    y="Monetary",
    color="persona",
    color_discrete_sequence=PALETTE,
    category_orders={"persona": present},
    hover_data={"Frequency": True, "cluster": True},
    log_x=True,
    log_y=True,
    opacity=0.65,
)
fig_scatter.update_layout(
    xaxis_title="Recency (days, log scale)",
    yaxis_title="Monetary (log scale)",
    margin=dict(t=10, b=10, l=10, r=10),
    height=480,
    legend_title="Persona",
)
st.plotly_chart(fig_scatter, width="stretch")

# --- Persona distribution pie ---
section_header("Customer distribution by persona")

counts = (
    scored["persona"].value_counts()
    .reindex(present)
    .dropna()
    .reset_index()
)
counts.columns = ["persona", "customers"]
counts["share_pct"] = (counts["customers"] / counts["customers"].sum() * 100).round(2)

fig_pie = px.pie(
    counts,
    names="persona",
    values="customers",
    color="persona",
    color_discrete_sequence=PALETTE,
    hole=0.4,
)
fig_pie.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent}")
fig_pie.update_layout(
    showlegend=True,
    margin=dict(t=10, b=10, l=10, r=10),
    height=420,
)
st.plotly_chart(fig_pie, width="stretch")

# --- Per-persona summary table ---
section_header("Per-persona summary")

summary = (
    scored.groupby("persona", observed=True)
    .agg(
        n_customers=("CustomerID", "count"),
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
        total_monetary=("Monetary", "sum"),
    )
    .reindex(present)
    .dropna()
    .reset_index()
)
summary["share_pct"] = (summary["n_customers"] / summary["n_customers"].sum() * 100).round(2)
summary = summary.sort_values("total_monetary", ascending=False).reset_index(drop=True)

st.dataframe(
    summary.style.format(
        {
            "n_customers": "{:,.0f}",
            "avg_recency": "{:.1f}",
            "avg_frequency": "{:.1f}",
            "avg_monetary": "{:,.2f}",
            "total_monetary": "{:,.2f}",
            "share_pct": "{:.2f}",
        }
    ),
    width="stretch",
    hide_index=True,
)
