# Phase 6 — Streamlit Dashboard

Built: 2026-07-03

This is the human-facing surface of the NeuralRetail platform: a
multi-page Streamlit app that lets a business user read the model's
output, understand the customer base, and act on reorder
recommendations.

## Pages

| # | Page | Source | What it shows |
|---|---|---|---|
| 1 | Executive Overview | `pages/1_executive_overview.py` | 4 KPI cards (Total Revenue, Orders, Customers, AOV) + revenue-by-country bar |
| 2 | Sales Analytics | `pages/2_sales_analytics.py` | Monthly revenue trend line + top-N products bar |
| 3 | Customer Hub | `pages/3_customer_hub.py` | RFM cluster scatter (Recency vs Monetary, log-log, coloured by persona) + persona pie + per-persona summary |
| 4 | Demand Explorer | `pages/4_demand_explorer.py` | Actual vs Prophet forecast with 95 % confidence band + horizon slider (7–90 days) + forecast table |
| 5 | Inventory Health | `pages/5_inventory_health.py` | ABC pie + reorder table filtered by ABC class / dead-stock / country / date |

Navigation: Streamlit auto-picks up files in `pages/` and renders them
in the left sidebar in filename order. The global sidebar (set up in
`app.py`) provides country + date-range filters that pages 1, 2, and
5 read; pages 3 and 4 are global because RFM and the forward-looking
forecast don't have a meaningful country/date scope.

## Architecture

```
src/neuralretail/dashboard/
├── app.py                # entry point — page config, sidebar, session_state
├── theme.py              # ACCENT (#0E8388) + 5 tints + neutrals — single source of truth
├── data.py               # @st.cache_data tables + @st.cache_resource fitted models
├── components.py         # kpi_card, section_header, format_currency/int
└── pages/
    ├── 1_executive_overview.py
    ├── 2_sales_analytics.py
    ├── 3_customer_hub.py
    ├── 4_demand_explorer.py
    └── 5_inventory_health.py
```

| Module | Role |
|---|---|
| `theme.py` | Defines the accent colour (`#0E8388`) and 4 derived tints. The "one accent" rule is enforced by a grep-based test that fails if any `dashboard/*.py` outside `theme.py` contains a raw `#RRGGBB` literal. |
| `data.py` | Wraps every `pd.read_parquet`, `pd.read_csv`, and `model.load_latest(...)` with the right `st.cache_*` decorator. Models are loaded once per session, tables once per change. |
| `components.py` | Reusable widgets (KPI cards, section headers) and formatters. Inline CSS pulls colours from `theme.py`. |
| `app.py` | Page config, sidebar filters, plotly default template, landing copy. |
| `pages/*.py` | One file per page, each ~80–120 lines. |

## Caching strategy

- **Tabular data** (`cleaned.parquet`, `rfm.parquet`, `daily_revenue.parquet`, `inventory_table.csv`): `@st.cache_data`. Streamlit pickles the result and re-uses it across reruns; the cache key includes the function arguments, so filters (different country lists) are memoised.
- **Fitted models** (KMeans pipeline, Prophet): `@st.cache_resource`. They are unpicklable for re-load and expensive to fit, so we hold a single instance in memory for the whole session.
- **Derived helpers** (`score_rfm_clusters`, `prophet_forecast`, `filter_by_sidebar`): uncached — they're cheap and parameter-dependent.

## Data sources (no API calls)

The dashboard reads directly from disk:
- `data/processed/*.parquet` (mounted read-only into the docker container at `/app/data/processed:ro`)
- `models/*.json`, `models/*.joblib`, `models/*.csv` (mounted read-only at `/app/models:ro`)

Decoupling from the FastAPI service means the dashboard can boot
before the API is up. The docker-compose `dashboard` service no
longer has `depends_on: api`.

## Styling rules

- Accent: `#0E8388` (teal). All other colours are derived tints.
- Theme is set three ways at once:
  1. `.streamlit/config.toml` paints every Streamlit chrome element.
  2. `theme.py` exports the constant for plotly `color_discrete_sequence` and inline CSS.
  3. `plotly.io.templates.default = "simple_white"` in `app.py` for clean report-style charts.
- One grep test (`tests/test_dashboard.py::test_no_raw_hex_outside_theme`) enforces the rule — if a future page adds a hard-coded hex, the test fails.

## Verification

```
$ make dashboard-test
======================= 16 passed in 40.39s =======================
```

The smoke test covers:
- 6 page-level renders via `streamlit.testing.v1.AppTest` (1 entry + 5 pages) — each runs to completion, asserts no `at.exception`, and that at least one title is set.
- 10 grep-based hex-code guards — one per `*.py` under `dashboard/`, asserting no raw `#RRGGBB` outside `theme.py`.

Manual verification:
```
$ make dashboard        # starts on :8501
# Click each page in the sidebar — no traceback in the terminal,
# every chart renders, filters actually change the data.
```

A `use_container_width` → `width="stretch"` deprecation sweep was
applied across all 5 pages; no streamlit deprecation warnings remain.

## Known limitations / follow-ups

- The synthetic dataset's metrics (Prophet MAPE ≈ 0.34, KMeans silhouette ≈ 0.23) are below the spec targets. The dashboard renders the model output regardless — but on a real Online Retail II ingest, the numbers would land much closer to the targets.
- The persona summary in Customer Hub uses the same `_assign_personas` heuristic as training, so labels are stable across runs.
- Phase 7 (drift monitoring) and Phase 8 (containerize + docs polish) are still pending.
