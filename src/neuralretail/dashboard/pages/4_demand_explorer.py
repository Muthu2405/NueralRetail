"""Page 4 — Demand Explorer.

Actual historical revenue vs. Prophet forecast, with a 95 % confidence
band. A slider controls the forecast horizon. Filters in the sidebar
only affect the *historical* overlay (the forecast is forward-looking
and doesn't depend on country/date filters).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from neuralretail.dashboard import data as d
from neuralretail.dashboard.components import format_currency, section_header
from neuralretail.dashboard.theme import ACCENT, ACCENT_LIGHT, NEUTRAL, NEUTRAL_PALE

st.set_page_config(page_title="Demand Explorer · NeuralRetail", page_icon="🔮", layout="wide")

st.title("🔮 Demand Explorer")
st.caption("Daily revenue history vs. Prophet forecast with 95 % confidence band.")

# --- Inputs ---
horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)

# --- Data ---
daily = d.load_daily_revenue()
model = d.load_prophet()
forecast = d.prophet_forecast(model, horizon=horizon)

# Sidebar filters affect only the historical overlay.
start = st.session_state.get("start")
end = st.session_state.get("end")
history = daily.copy()
if start:
    history = history[history.index >= pd.to_datetime(start)]
if end:
    history = history[history.index <= pd.to_datetime(end) + pd.Timedelta(days=1)]

if history.empty and forecast.empty:
    st.warning("No data and no forecast. Run `make train` to fit the Prophet model.")
    st.stop()

# --- Headline metrics ---
last_actual = float(history["Revenue"].iloc[-1]) if not history.empty else float("nan")
avg_forecast = float(forecast["yhat"].tail(horizon).mean())
delta = (avg_forecast - last_actual) / last_actual * 100 if last_actual else 0.0

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Last actual (filtered)", format_currency(last_actual))
with c2:
    st.metric(
        f"Avg forecast next {horizon}d",
        format_currency(avg_forecast),
        delta=f"{delta:+.1f}%",
    )
with c3:
    st.metric("Horizon days", f"{horizon}")

st.write("")

# --- Forecast chart ---
section_header(f"Actual vs forecast (next {horizon} days)")

fig = go.Figure()

# Confidence band: lower → upper, filled to next y (upper).
fig.add_trace(
    go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor=ACCENT_LIGHT,
        opacity=0.25,
        line=dict(color="rgba(0,0,0,0)"),
        name="95% confidence",
        hoverinfo="skip",
    )
)

# Forecast line.
fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        line=dict(color=ACCENT, width=3),
        name="Forecast (yhat)",
    )
)

# Historical actuals (only the in-range slice from the sidebar filters).
if not history.empty:
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Revenue"],
            mode="markers",
            marker=dict(color=NEUTRAL, size=6),
            name="Actual",
        )
    )

# Vertical separator at the boundary between actuals and forecast.
# Plotly's add_vline fails on pandas Timestamps in some versions because
# it tries to take the mean; convert to a unix-ms integer which is what
# plotly wants on a date axis anyway.
boundary_dt = (
    history.index.max()
    if not history.empty
    else pd.to_datetime(forecast["ds"].iloc[0]) - pd.Timedelta(days=1)
)
boundary_ms = int(pd.Timestamp(boundary_dt).timestamp() * 1000)
fig.add_vline(
    x=boundary_ms,
    line=dict(color=NEUTRAL_PALE, width=1, dash="dash"),
    annotation_text="Forecast →",
    annotation_position="top right",
)

fig.update_layout(
    xaxis_title=None,
    yaxis_title="Revenue",
    margin=dict(t=20, b=10, l=10, r=10),
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, width="stretch")

# --- Forecast table ---
section_header("Forecast detail (next horizon days)")

table = (
    forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    .assign(
        yhat=lambda d: d["yhat"].map(lambda v: format_currency(v)),
        yhat_lower=lambda d: d["yhat_lower"].map(lambda v: format_currency(v)),
        yhat_upper=lambda d: d["yhat_upper"].map(lambda v: format_currency(v)),
    )
    .rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower 95%", "yhat_upper": "Upper 95%"})
)
st.dataframe(table, width="stretch", hide_index=True)
