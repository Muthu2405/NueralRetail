"""Cached data + model loaders for the Streamlit dashboard.

Tables are loaded with ``@st.cache_data`` (pickled once per change of
the input arg, otherwise returned by reference), and fitted models
with ``@st.cache_resource`` (loaded exactly once per Streamlit
session — they are heavy and unpicklable to recompute per call).

Everything reads from the on-disk processed dataset and on-disk
``models/`` artifacts. No HTTP calls; no FastAPI dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from neuralretail.config import get_settings
from neuralretail.models import forecasting as fc_mod
from neuralretail.models import segmentation as seg_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tabular loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading transactions…")
def load_transactions() -> pd.DataFrame:
    """Cleaned transaction table (≈9 500 rows)."""
    path = get_settings().processed_dir / "cleaned.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading RFM table…")
def load_rfm() -> pd.DataFrame:
    """RFM table (4 780 customers, snapshot at training time)."""
    path = get_settings().processed_dir / "rfm.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading daily revenue…")
def load_daily_revenue() -> pd.DataFrame:
    """Daily revenue series (≈373 days)."""
    path = get_settings().processed_dir / "daily_revenue.parquet"
    df = pd.read_parquet(path)
    # Guarantee a DatetimeIndex for downstream plotly/Prophet code.
    if "ds" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["ds"]))
    return df


@st.cache_data(show_spinner="Loading inventory table…")
def load_inventory() -> pd.DataFrame:
    """Per-SKU ABC + EOQ + dead-stock table (≈9 500 SKUs)."""
    path = get_settings().models_dir / "inventory_table.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Fitted-model loaders (one-per-session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading KMeans segmentation pipeline…")
def load_segmentation_model() -> Any:
    """The fitted sklearn ``Pipeline`` (StandardScaler → KMeans)."""
    path = str(get_settings().models_dir / "segmentation_kmeans.joblib")
    return seg_mod.load_latest(path)


@st.cache_resource(show_spinner="Loading Prophet demand model…")
def load_prophet() -> Any:
    """The fitted Prophet model."""
    path = str(get_settings().models_dir / "prophet_demand.json")
    return fc_mod.load_latest(path)


# ---------------------------------------------------------------------------
# Derived helpers (no caching — cheap and parameter-dependent)
# ---------------------------------------------------------------------------


def score_rfm_clusters(rfm: pd.DataFrame, pipe: Any) -> pd.DataFrame:
    """Attach a ``cluster`` and ``persona`` column to an RFM frame.

    The persona map is re-derived from the cluster centroids at score
    time, so labels always match the training-time convention (see
    ``neuralretail.models.segmentation._assign_personas``).
    """
    feats = rfm[seg_mod.SEGMENT_FEATURES].fillna(0).to_numpy(dtype=float)
    labels = pipe.predict(feats)

    out = rfm.copy()
    out["cluster"] = labels.astype(int)

    centroids = pd.DataFrame(
        pipe.named_steps["scaler"].inverse_transform(
            pipe.named_steps["kmeans"].cluster_centers_
        ),
        columns=seg_mod.SEGMENT_FEATURES,
    )
    persona_map = seg_mod._assign_personas(centroids)
    out["persona"] = out["cluster"].map(persona_map).fillna("Regular")
    return out


def prophet_forecast(model: Any, horizon: int) -> pd.DataFrame:
    """Forecast ``horizon`` days forward; return ``(ds, yhat, yhat_lower, yhat_upper)``."""
    return fc_mod.predict(model, periods=horizon)


# ---------------------------------------------------------------------------
# Sidebar filter plumbing
# ---------------------------------------------------------------------------


def filter_by_sidebar(
    df: pd.DataFrame,
    *,
    date_col: str | None = "InvoiceDate",
) -> pd.DataFrame:
    """Apply the country + date filters stored in ``st.session_state``.

    Pages that have a country or InvoiceDate column call this. Pages
    that don't (Customer Hub, Demand Explorer) skip it.
    """
    countries: list[str] = st.session_state.get("countries") or []
    if countries and "Country" in df.columns:
        df = df[df["Country"].isin(countries)]

    start = st.session_state.get("start")
    end = st.session_state.get("end")
    if date_col and date_col in df.columns and (start or end):
        ts = pd.to_datetime(df[date_col])
        if start:
            df = df[ts >= pd.to_datetime(start)]
        if end:
            df = df[ts <= pd.to_datetime(end) + pd.Timedelta(days=1)]
    return df
