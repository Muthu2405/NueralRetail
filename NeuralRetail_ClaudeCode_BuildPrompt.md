# NeuralRetail — Claude Code Build Prompt


Paste this whole prompt into Claude Code in an empty project folder. It describes the full scope. Claude Code should work through it phase by phase, committing after each phase, rather than trying to do everything in one shot.

---

## PROJECT CONTEXT

Build **NeuralRetail**, an AI-powered retail sales intelligence platform for a company called Amdox Technologies. It ingests retail transaction data and produces: demand forecasts, customer segmentation, churn predictions, and inventory reorder recommendations — all exposed through a multi-page Streamlit dashboard and a FastAPI scoring service, with MLflow experiment tracking and basic drift monitoring.

This is a portfolio/internship-grade project, not a hyperscale production system — so implement everything for real, keep it runnable on a single laptop via `docker-compose`, but structure the code as if it were production (clean modules, tests, typed schemas, logging), so it can later scale.

**Dataset:** Use the "Online Retail II" dataset from the UCI Machine Learning Repository (or its Kaggle mirror). Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country. Write the data loader to gracefully handle either the UCI XLSX or a CSV export.

---

## GOALS / SUCCESS METRICS

- Demand forecast MAPE ≤ 10% on a 30-day holdout
- Churn classifier AUC-ROC ≥ 0.90
- Customer segmentation silhouette score ≥ 0.55, 4–8 interpretable clusters
- Dashboard loads and every page renders without error against the processed dataset
- API returns predictions for all four core endpoints
- End-to-end pipeline runs with one command: `make pipeline` (or `docker-compose up`)

---

## TECH STACK

- **Language:** Python 3.12, managed with `uv` or `poetry`
- **Data processing:** pandas, numpy (Polars optional for speed, not required)
- **Data quality:** Great Expectations (a lightweight suite, not full production config)
- **ML:** scikit-learn, XGBoost, LightGBM, Prophet
- **Explainability:** SHAP
- **Experiment tracking:** MLflow (local file-store or SQLite backend is fine)
- **Serving API:** FastAPI + Pydantic v2 + Uvicorn
- **Dashboard:** Streamlit + Plotly
- **Drift monitoring:** Evidently AI (generate an HTML report; a simple scheduled check is enough, no need for a live alerting pipeline)
- **Orchestration:** a simple Python/Makefile pipeline is fine; add an optional Airflow DAG stub (`dags/neuralretail_pipeline.py`) that mirrors the same steps, but it doesn't need to run in this environment
- **Containerization:** Docker + docker-compose (services: `mlflow`, `api`, `dashboard`)
- **Testing:** pytest for feature engineering and model utility functions
- **Config:** `.env` / `pydantic-settings`, no hardcoded paths or secrets

Skip (or stub as documentation only, not real infra) unless I explicitly ask: Spark, Delta Lake, Feast, Kafka, Kubernetes/Helm, Terraform, DoWhy/EconML causal inference, TimeGPT, OpenLineage. These are overkill for a single-node build — note them in the README as "future scale-up path" instead of implementing them.

---

## REPO STRUCTURE

```
neuralretail/
├── data/
│   ├── raw/                  # original downloaded dataset (gitignored)
│   └── processed/            # cleaned parquet/csv outputs
├── src/
│   └── neuralretail/
│       ├── config.py
│       ├── data/
│       │   ├── ingest.py         # load + validate raw data
│       │   └── clean.py          # cleaning: drop cancellations, nulls, negative qty/price
│       ├── features/
│       │   ├── rfm.py             # Recency, Frequency, Monetary
│       │   └── timeseries.py      # daily aggregation, lag/rolling features
│       ├── models/
│       │   ├── forecasting.py     # Prophet (+ optional simple LSTM) demand model
│       │   ├── churn.py           # XGBoost/LightGBM churn classifier + SHAP
│       │   ├── segmentation.py    # KMeans (+ optional DBSCAN) RFM clustering
│       │   └── inventory.py       # ABC classification + EOQ calculation
│       ├── monitoring/
│       │   └── drift.py           # Evidently AI report generation
│       ├── api/
│       │   ├── main.py            # FastAPI app
│       │   └── schemas.py         # Pydantic request/response models
│       └── dashboard/
│           ├── app.py             # Streamlit entrypoint with page router
│           └── pages/
│               ├── 1_executive_overview.py
│               ├── 2_sales_analytics.py
│               ├── 3_customer_hub.py
│               ├── 4_demand_explorer.py
│               └── 5_inventory_health.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_forecasting.ipynb
│   ├── 04_churn_model.ipynb
│   └── 05_segmentation_inventory.ipynb
├── dags/
│   └── neuralretail_pipeline.py   # documentation-only Airflow DAG stub
├── tests/
│   ├── test_clean.py
│   ├── test_rfm.py
│   └── test_inventory.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── models/                        # MLflow local artifact store (gitignored)
├── mlruns/                        # MLflow tracking store (gitignored)
├── .env.example
├── Makefile
├── pyproject.toml
├── README.md
└── report/
    └── model_cards/                # one markdown model card per registered model
```

---

## PHASE-BY-PHASE BUILD PLAN

Work through these phases in order. After each phase, run the tests / smoke-check the output before moving on, and give me a short summary of what was built.

### Phase 1 — Project scaffolding & data pipeline
1. Set up the repo structure above, `pyproject.toml`, `.env.example`, `Makefile` with targets `install`, `data`, `features`, `train`, `api`, `dashboard`, `test`, `pipeline`.
2. `data/ingest.py`: download or load the Online Retail II dataset (support both a local file path and, if network access exists, a direct download; otherwise instruct the user where to place the file).
3. `data/clean.py`: drop cancelled invoices (InvoiceNo starting with "C"), drop rows with missing CustomerID, filter Quantity > 0 and UnitPrice > 0, parse InvoiceDate, compute TotalPrice. Write a small Great Expectations suite that checks row count, null CustomerID rate, and value ranges; fail loudly (non-zero exit) if checks fail.
4. Output a cleaned parquet file to `data/processed/`.
5. Write `tests/test_clean.py` covering the cleaning rules.

### Phase 2 — Feature engineering
1. `features/rfm.py`: compute Recency, Frequency, Monetary per CustomerID relative to a snapshot date.
2. `features/timeseries.py`: daily revenue aggregation, lag features (t-1, t-7, t-14), rolling mean/std (7/14/30 day windows), day-of-week and seasonality flags.
3. `tests/test_rfm.py` covering RFM calculation correctness on a small synthetic dataframe.

### Phase 3 — Models
1. **Forecasting** (`models/forecasting.py`): train a Prophet model on daily revenue; log MAPE/RMSE to MLflow; save the model as an MLflow artifact; generate a 30–90 day forecast with confidence intervals. Keep the LSTM ensemble optional/behind a flag — Prophet alone must work end-to-end first.
2. **Churn** (`models/churn.py`): define churn as inactive > 90 days; train XGBoost (and optionally LightGBM) on RFM + behavioral features; report AUC-ROC, precision/recall; add SHAP TreeExplainer output (summary + a helper to get a per-customer waterfall); log everything to MLflow; register the model.
3. **Segmentation** (`models/segmentation.py`): StandardScaler + KMeans on RFM (k selected via silhouette score, try k=3–8); label clusters with human-readable personas (e.g. Champions, At Risk, Hibernating, Loyal) based on cluster centroid characteristics — don't hardcode labels, derive them programmatically from centroid rank.
4. **Inventory** (`models/inventory.py`): ABC classification by revenue contribution; EOQ formula given an assumed holding cost % and ordering cost; flag dead-stock (SKUs with no sales in last N days).
5. Each model module should expose a clean `train()` / `predict()` / `load_latest()` interface so the API and dashboard can both call it.

### Phase 4 — MLflow integration
1. Local MLflow tracking server config (SQLite backend, local artifact store) via docker-compose.
2. Every model training run above logs params, metrics, and the model artifact; promote the best run to a "Production" alias in the MLflow Model Registry.
3. Write one markdown model card per model into `report/model_cards/` (training data summary, metrics, intended use, limitations).

### Phase 5 — FastAPI service
1. Endpoints: `POST /predict/demand`, `POST /predict/churn`, `POST /segment/score`, `POST /inventory/reorder`, `GET /health`.
2. Pydantic v2 request/response schemas with field validation.
3. Load models from the MLflow registry at startup (not retrained per-request).
4. Basic API key auth via header, configurable through `.env`.
5. Write a quick smoke test that hits each endpoint with sample payloads.

### Phase 6 — Streamlit dashboard
Build a 5-page app matching the report's screenshots:
1. **Executive Overview** — KPI cards (Total Revenue, Orders, Customers, AOV), revenue-by-country bar chart.
2. **Sales Analytics** — monthly revenue trend line, top-10 products bar chart.
3. **Customer Hub** — RFM cluster scatter (Recency vs Monetary colored by cluster), segment distribution pie chart.
4. **Demand Explorer** — actual vs forecast line chart with confidence interval band, forecast table.
5. **Inventory Health** — ABC pie chart, recommended reorder quantities table.

Use a sidebar for navigation and country/date filters. Cache expensive data loads with `st.cache_data`. Keep styling clean and consistent (pick one accent color, use it throughout — don't hardcode a specific brand's exact hex codes).

### Phase 7 — Monitoring
1. `monitoring/drift.py`: generate an Evidently AI data-drift HTML report comparing a "reference" slice of the data to a "current" slice; save to `report/drift_report.html`.
2. Document (README section) how this would plug into a scheduled Airflow job with an alert threshold in a real deployment.

### Phase 8 — Containerization & docs
1. `Dockerfile.api`, `Dockerfile.dashboard`, `docker-compose.yml` wiring up `mlflow`, `api`, `dashboard` services with the processed data mounted as a volume.
2. `README.md`: setup instructions, architecture diagram (Mermaid is fine), how to run the pipeline end-to-end, screenshots placeholder, model metrics summary, and a "Future Scale-Up Path" section briefly noting how this would evolve into the full Spark/Feast/Kubernetes stack.
3. `make pipeline` should run ingest → clean → features → train all models → generate drift report, in order, using the cleaned data.

---

## WORKING STYLE INSTRUCTIONS FOR CLAUDE CODE

- Work phase by phase. After finishing a phase, run relevant tests/scripts to confirm it actually works before moving to the next phase.
- Prefer real, runnable code over stubs — but it's fine to gate optional heavy pieces (LSTM ensemble, DBSCAN, LightGBM) behind config flags so the core pipeline stays fast to iterate on.
- Use type hints and docstrings throughout `src/`.
- Don't invent business metrics or claims in the README that the code doesn't actually produce — report whatever MAPE/AUC/silhouette the trained models actually achieve on this dataset.
- If the UCI dataset can't be downloaded automatically in this environment, clearly tell me where to place the file manually and continue building against a small synthetic sample so the pipeline stays testable.
- Ask me before making any destructive changes or before choosing between two materially different design options (e.g. SQLite vs Postgres for MLflow backend).
