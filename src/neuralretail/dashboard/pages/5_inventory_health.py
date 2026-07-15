"""Page 5 — Inventory Health.

ABC class distribution (always global — matches the trained model) and
a reorder recommendations table filtered by the sidebar country/date
inputs. The dead-stock flag respects the date cutoff.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from neuralretail.dashboard import data as d
from neuralretail.dashboard.components import format_int, section_header
from neuralretail.dashboard.theme import ACCENT, ACCENT_DARK, ACCENT_LIGHT

st.set_page_config(page_title="Inventory Health · NeuralRetail", page_icon="📦", layout="wide")

st.title("📦 Inventory Health")
st.caption("ABC distribution and reorder recommendations.")

# --- Global inventory (ABC is from the trained model — don't filter) ---
inv = d.load_inventory()

# --- Headline numbers ---
total_skus = int(len(inv))
n_a = int((inv["ABC"] == "A").sum())
n_b = int((inv["ABC"] == "B").sum())
n_c = int((inv["ABC"] == "C").sum())
n_dead = int((inv["IsDeadStock"] == 1).sum())
dead_pct = n_dead / total_skus * 100 if total_skus else 0.0
total_revenue = float(inv["Revenue"].sum())
total_eoq = float(inv["EOQ"].sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total SKUs", format_int(total_skus))
with c2:
    st.metric("Class A / B / C", f"{format_int(n_a)} / {format_int(n_b)} / {format_int(n_c)}")
with c3:
    st.metric("Dead stock", f"{format_int(n_dead)}", delta=f"{dead_pct:.1f}% of SKUs", delta_color="inverse")
with c4:
    st.metric("Total annual demand (EOQ)", format_int(total_eoq))

st.write("")

# --- ABC pie ---
section_header("ABC distribution (global)")

abc_counts = inv["ABC"].value_counts().reindex(["A", "B", "C"]).reset_index()
abc_counts.columns = ["ABC", "count"]

fig_abc = px.pie(
    abc_counts,
    names="ABC",
    values="count",
    color="ABC",
    color_discrete_map={"A": ACCENT_DARK, "B": ACCENT, "C": ACCENT_LIGHT},
    hole=0.45,
    category_orders={"ABC": ["A", "B", "C"]},
)
fig_abc.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent}")
fig_abc.update_layout(
    showlegend=True,
    margin=dict(t=10, b=10, l=10, r=10),
    height=380,
)
st.plotly_chart(fig_abc, width="stretch")

# --- Reorder table (sidebar-filtered) ---
section_header("Reorder recommendations")

# The inventory table is per-SKU; the country/date sidebar filters
# only make sense for the *reorder* list, not the ABC classes. We
# apply them as a soft filter: hide dead stock older than the
# end date, and let the user toggle ABC + dead-stock-only.
f1, f2, f3 = st.columns(3)
with f1:
    abc_filter = st.multiselect("ABC class", options=["A", "B", "C"], default=["A", "B", "C"])
with f2:
    dead_only = st.checkbox("Dead stock only", value=False)
with f3:
    sort_by = st.selectbox("Sort by", options=["EOQ", "Revenue", "AnnualDemand", "DaysSinceLastSale"], index=0)

filtered = inv.copy()
if abc_filter:
    filtered = filtered[filtered["ABC"].isin(abc_filter)]
else:
    st.info("Select at least one ABC class to see reorders.")
    st.stop()
if dead_only:
    filtered = filtered[filtered["IsDeadStock"] == 1]

# Sidebar date filter: drop dead stock that *predates* the user's
# window so the table is scoped to "what's relevant right now".
end = st.session_state.get("end")
if end:
    cutoff = pd.to_datetime(end)
    filtered = filtered[pd.to_datetime(filtered["LastSale"]) <= cutoff]

# Sidebar country filter: re-score revenue within the selected
# countries so EOQ reflects actual demand. We rebuild a quick map
# from the transactions frame.
countries = st.session_state.get("countries")
if countries:
    tx = d.load_transactions()
    tx = d.filter_by_sidebar(tx, date_col="InvoiceDate")
    if not tx.empty:
        country_rev = (
            tx.groupby("StockCode", as_index=False)["TotalPrice"]
            .sum()
            .rename(columns={"TotalPrice": "CountryRevenue"})
        )
        filtered = filtered.merge(country_rev, on="StockCode", how="left")
        filtered["CountryRevenue"] = filtered["CountryRevenue"].fillna(0.0)
    else:
        filtered["CountryRevenue"] = 0.0
else:
    filtered["CountryRevenue"] = filtered["Revenue"]

filtered = filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)

# Columns to display — friendly names + reasonable formatting.
display = filtered[
    [
        "StockCode",
        "Description",
        "ABC",
        "UnitsSold",
        "Revenue",
        "CountryRevenue",
        "AnnualDemand",
        "EOQ",
        "DaysSinceLastSale",
        "IsDeadStock",
    ]
].rename(
    columns={
        "StockCode": "Stock code",
        "Description": "Description",
        "ABC": "Class",
        "UnitsSold": "Units sold",
        "Revenue": "Revenue (global)",
        "CountryRevenue": "Revenue (filtered)",
        "AnnualDemand": "Annual demand",
        "EOQ": "EOQ",
        "DaysSinceLastSale": "Days since last sale",
        "IsDeadStock": "Dead stock",
    }
)

st.dataframe(
    display.style.format(
        {
            "Units sold": "{:,.0f}",
            "Revenue (global)": "{:,.2f}",
            "Revenue (filtered)": "{:,.2f}",
            "Annual demand": "{:,.1f}",
            "EOQ": "{:,.1f}",
            "Days since last sale": "{:,.0f}",
        }
    ),
    width="stretch",
    hide_index=True,
    height=480,
)
