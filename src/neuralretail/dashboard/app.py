"""Streamlit entry point for the NeuralRetail dashboard.

Sets the page config, paints the global sidebar (country + date
filters), seeds ``st.session_state`` so each page can read the
filters, and configures the default plotly template.

The five pages under ``pages/`` are picked up automatically by
Streamlit's native multi-page routing — they appear in the left
sidebar in filename order.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.io as pio
import streamlit as st

from neuralretail import __version__
from neuralretail.dashboard import data as d
from neuralretail.dashboard.theme import ACCENT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="NeuralRetail",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Plotly default template
# ---------------------------------------------------------------------------


# Pin the simple_white template once so every page's charts inherit a
# clean, report-style look. Pages can still pass color_discrete_sequence
# overrides to nudge a single trace into the accent.
pio.templates.default = "simple_white"


# ---------------------------------------------------------------------------
# Sidebar filters — seeded once, then read by each page
# ---------------------------------------------------------------------------


def _seed_session_state() -> None:
    """Initialise sidebar filter defaults on the very first run.

    We have to know the data range to default the date pickers
    sensibly, so we load the cleaned transactions frame here. The
    loader is cached so the cost is paid once per session.
    """
    if "_initialised" in st.session_state:
        return

    tx = d.load_transactions()
    min_d = pd.to_datetime(tx["InvoiceDate"]).min().date()
    max_d = pd.to_datetime(tx["InvoiceDate"]).max().date()
    countries = sorted(tx["Country"].unique().tolist())

    st.session_state.setdefault("countries", countries)  # all selected by default
    st.session_state.setdefault("start", min_d)
    st.session_state.setdefault("end", max_d)
    st.session_state.setdefault("_min_date", min_d)
    st.session_state.setdefault("_max_date", max_d)
    st.session_state.setdefault("_country_options", countries)
    st.session_state["_initialised"] = True


_seed_session_state()


with st.sidebar:
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.2rem;'>🛒 NeuralRetail</h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"v{__version__} — AI retail intelligence")
    st.divider()

    st.subheader("Filters")
    st.multiselect(
        "Countries",
        options=st.session_state["_country_options"],
        key="countries",
    )
    st.date_input(
        "Start date",
        min_value=st.session_state["_min_date"],
        max_value=st.session_state["_max_date"],
        key="start",
    )
    st.date_input(
        "End date",
        min_value=st.session_state["_min_date"],
        max_value=st.session_state["_max_date"],
        key="end",
    )

    # Clamp end >= start in case the user picked an out-of-order range.
    if st.session_state["end"] < st.session_state["start"]:
        st.warning("End date is before start date — pages will be empty until you fix this.")

    st.divider()
    st.caption(
        "Filters apply to pages 1, 2, and 5. "
        "Customer Hub and Demand Explorer are global."
    )


# ---------------------------------------------------------------------------
# Landing copy
# ---------------------------------------------------------------------------


st.title("NeuralRetail — Retail Sales Intelligence")
st.markdown(
    """
    Welcome. Use the sidebar to navigate:

    1. **Executive Overview** — headline KPIs and revenue by country.
    2. **Sales Analytics** — monthly revenue trend and top products.
    3. **Customer Hub** — RFM clusters, personas, and segment breakdown.
    4. **Demand Explorer** — actual vs. forecast with a 95 % confidence band.
    5. **Inventory Health** — ABC classification and recommended reorders.

    All charts are rendered against the on-disk processed dataset; the
    data is the same as what was used to train the models.
    """
)
