"""Reusable Streamlit widgets for the dashboard.

The two helpers here are deliberately small and side-effect-free:

- ``kpi_card`` renders one KPI in a styled box (title + value + optional
  delta). All five pages use it for the headline metrics.
- ``section_header`` is a thin wrapper around ``st.subheader`` that
  injects a coloured left border so the section title is visually
  anchored to the accent.

Plotly figures are built with ``color_discrete_sequence=[ACCENT]`` at
the call site (in each page) — this module is for layout primitives,
not chart styling.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from neuralretail.dashboard.theme import (
    ACCENT,
    DIVIDER,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
)


def kpi_card(title: str, value: str, delta: Optional[str] = None) -> None:
    """Render a single KPI tile.

    Parameters
    ----------
    title:
        Short label, e.g. "Total Revenue".
    value:
        Pre-formatted value string, e.g. "$1.83M". The page is
        responsible for formatting (we don't pull in locale here).
    delta:
        Optional secondary line, e.g. "+12.4% vs last month".
    """
    delta_html = (
        f'<div style="color:{ACCENT};font-size:0.85rem;margin-top:0.2rem;">{delta}</div>'
        if delta
        else ""
    )
    st.markdown(
        f"""
        <div style="
            border:1px solid {DIVIDER};
            border-left:4px solid {ACCENT};
            border-radius:6px;
            padding:0.8rem 1rem;
            background:{SURFACE};
            min-height:96px;
        ">
            <div style="color:{TEXT_MUTED};font-size:0.8rem;text-transform:uppercase;
                        letter-spacing:0.04em;">{title}</div>
            <div style="color:{TEXT_PRIMARY};font-size:1.7rem;font-weight:600;
                        margin-top:0.25rem;line-height:1.1;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    """Subheader with the accent colour as a left border."""
    st.markdown(
        f"""
        <div style="
            border-left:4px solid {ACCENT};
            padding-left:0.6rem;
            margin: 0.5rem 0 0.4rem 0;
            color:{TEXT_PRIMARY};
            font-size:1.1rem;
            font-weight:600;
        ">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    """Compact currency formatter, e.g. 1_834_355 → '$1.83M'."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}k"
    return f"${value:,.0f}"


def format_int(value: int | float) -> str:
    """Integer with thousands separators, e.g. 4780 → '4,780'."""
    return f"{int(value):,}"
