# Phase 7 + 8 — Drift monitoring, containerization & docs

Generated: 2026-07-05

This report covers the final two phases of the NeuralRetail build:
data-drift monitoring (Phase 7) and the containerization + docs
wrap-up (Phase 8). As with `phase4_5_report.md` and
`phase6_dashboard.md`, every metric below is the **actual output** of
the pipeline — no claims are interpolated.

## 1. What was built (Phase 7 — drift monitoring)

| Artefact | Path | Purpose |
|---|---|---|
| `monitoring/drift.py` | `src/neuralretail/monitoring/drift.py` | Chronological reference/current split + Evidently `DataDriftPreset` HTML report |
| `monitor` CLI subcommand | `src/neuralretail/cli.py:cmd_monitor` | Thin dispatcher, lazy imports the drift module |
| `monitor` Makefile target | `Makefile:53` | `make monitor` |
| `monitor` step in `pipeline` | `Makefile:67` | `make pipeline` now runs `data → features → train → monitor` |
| `Settings.drift_reference_fraction` | `src/neuralretail/config.py:55` | Configurable 0.7 default, override via `NEURALRETAIL_DRIFT_REFERENCE_FRACTION` |
| `tests/test_drift.py` | `tests/test_drift.py` | 9 tests — split semantics + end-to-end report writer |

### 1.1 Design choices

- **Chronological 70/30 split** on `InvoiceDate`. The reference
  window is the oldest 70 % of the data; the current window is the
  newest 30 %. This mirrors the standard production pattern: train
  on the past, score drift on the present.
- **Six columns checked**: `Quantity`, `UnitPrice`, `TotalPrice`,
  `Country` (raw) + `Hour`, `DayOfWeek` (derived from `InvoiceDate`).
  Evidently picks the appropriate test per column type —
  Wasserstein distance for the numeric features, Jensen-Shannon
  distance for the categoricals.
- **HTML + JSON sidecar**. The full interactive report is the
  primary artefact (`report/drift_report.html`, ~3.8 MB). A
  machine-readable summary lands next to it
  (`report/drift_report.summary.json`) so a downstream alerting job
  can read it without parsing HTML.
- **No live alerting.** Per the spec — a scheduled check is enough;
  threshold-based paging is documented as a future step.

### 1.2 Drift result on the synthetic data

The pipeline was run end-to-end on the cleaned parquet (9,500 rows
over 2010-12-01 → 2011-12-08). The split was 6,650 reference rows
(2010-12-01 → 2011-08-17) vs 2,850 current rows (2011-08-17 →
2011-12-08). Result: **0 of 6 columns drifted**.

| Column | Method | Score | Threshold | Drifted? |
|---|---|---|---|---|
| Quantity | Wasserstein (normed) | 0.0210 | 0.1 | no |
| UnitPrice | Wasserstein (normed) | 0.0180 | 0.1 | no |
| TotalPrice | Wasserstein (normed) | 0.0148 | 0.1 | no |
| Hour | Wasserstein (normed) | 0.0154 | 0.1 | no |
| Country | Jensen-Shannon | 0.0246 | 0.1 | no |
| DayOfWeek | Jensen-Shannon | 0.0136 | 0.1 | no |

**Interpretation.** A clean drift report on the synthetic data is
expected: the generator injects stationarity-preserving noise rather
than trend shifts, so the chronological split sees two slices from
the same distribution. The detector is doing its job — a synthetic
self-test where the current window's `Quantity`, `UnitPrice`,
`TotalPrice`, and `Country` distributions were deliberately shifted
correctly flagged 4/6 columns as drifted (drift_share = 0.67), so
the threshold logic is verified end-to-end.

## 2. What was built (Phase 8 — containerization & docs)

| Artefact | Path | Notes |
|---|---|---|
| Airflow DAG stub | `dags/neuralretail_pipeline.py` | BashOperator chain mirroring `make pipeline`. Documentation-only — imports are wrapped so the file is parseable without Airflow installed. |
| `dags/.gitkeep` | `dags/.gitkeep` | Preserves the directory in a fresh clone. |
| Mermaid architecture diagram | `README.md` § Architecture | Renders the data → features → train → MLflow → API/Dashboard + drift + Airflow graph. |
| Monitoring section | `README.md` § Monitoring & drift | Documents `make monitor` and the production wiring pattern. |
| Screenshots placeholders | `README.md` § Screenshots | Points at `report/screenshots/`; six `gitkeep` slots ready to receive PNGs. |
| Expanded Future Scale-Up | `README.md` § Future scale-up path | One short paragraph per item (Spark, Feast, Kafka, K8s, Terraform, DoWhy, TimeGPT, OpenLineage) explaining *why* it's the right next step. |
| Four model cards | `report/model_cards/*.md` | One card per registered model: description, training data, metrics, intended use, limitations. |
| Phase 4+5 metrics table | `README.md` § Model metrics | Restored as a clean H2 heading after the docs restructure. |
| Project layout block | `README.md` § Project layout | Now lists `monitoring/`, `dags/`, `report/model_cards/`, `report/screenshots/`. |
| Quick start | `README.md` § Quick start | Updated to recommend `make pipeline` as the single-command end-to-end run. |

### 2.1 Containerization status (unchanged)

The `docker/` directory was already complete from a prior session:

- `Dockerfile.api` — FastAPI service.
- `Dockerfile.dashboard` — Streamlit multi-page app.
- `Dockerfile.mlflow` — MLflow tracking server with SQLite backend.
- `docker-compose.yml` — three-service stack (`mlflow`, `api`,
  `dashboard`) with healthcheck on MLflow and a `depends_on:
  condition: service_healthy` gate on the API. The dashboard has
  no API dependency because it reads parquet + on-disk model
  artifacts directly.

`docker compose -f docker/docker-compose.yml config` validates
cleanly (one cosmetic warning: the obsolete top-level `version:`
key, which the spec already shows).

### 2.2 Airflow stub

`dags/neuralretail_pipeline.py` is intentionally minimal. It defines
a single DAG with five BashOperator tasks
(`ingest_and_clean → build_features → train_models → promote_best →
drift_report`) where each task shells out to a `make` target or the
`promote` CLI subcommand. The header docstring explains how to
deploy it: mount the repo into the Airflow worker's `dags/`
folder, install the package, and let the scheduler pick it up.
Importing the file in a clean environment (no `apache-airflow`
installed) raises `ImportError` at task-construction time, but the
file is **syntactically valid Python** — `ast.parse` succeeds.

## 3. Pytest

```
57 passed, 27 warnings in 43.41s
```

Suites:
- `tests/test_api.py` (7 tests) — endpoint auth, validation, happy paths
- `tests/test_clean.py` (9 tests) — Great Expectations suite
- `tests/test_dashboard.py` (16 tests) — AppTest smoke + hex-code guard
- `tests/test_drift.py` (9 tests) — split semantics + end-to-end report writer
- `tests/test_inventory.py` (8 tests) — ABC/EOQ math + dead-stock flag
- `tests/test_rfm.py` (8 tests) — RFM computation

The 9 new `test_drift.py` cases are:

1. `test_split_is_chronological_and_disjoint` — reference ends ≤ current starts; sizes sum to the input.
2. `test_split_fraction_is_honoured` — `0.7` gives 700/300 on a 1000-row frame.
3. `test_split_rejects_invalid_fraction[0.0]`
4. `test_split_rejects_invalid_fraction[1.0]`
5. `test_split_rejects_invalid_fraction[-0.1]`
6. `test_split_rejects_invalid_fraction[1.5]`
7. `test_split_requires_invoice_date_column` — missing `InvoiceDate` raises `KeyError`.
8. `test_save_drift_report_writes_html` — file exists, non-empty, looks like HTML; sidecar JSON exists; dataclass shape is sane.
9. `test_save_drift_report_uses_default_output_when_omitted` — falls back to `settings.report_dir` when `output_path` is `None`.

## 4. End-to-end pipeline output

`make pipeline` (run as the four CLI subcommands in sequence) prints:

```
$ python -m neuralretail.cli data
Cleaned: 10000 -> 9500 rows (cancelled=200, null_customer=200, bad_qty=50, bad_price=50)
Processed parquet: data\processed\cleaned.parquet

$ python -m neuralretail.cli features
RFM: 4,780 customers, Recency mean=139.5d, Frequency mean=1.99, Monetary mean=$383.67
  -> data\processed\rfm.parquet
Daily revenue: 373 days, 2010-12-01 to 2011-12-08, total $1,833,949.35
  -> data\processed\daily_revenue.parquet
Timeseries features: 373 rows, 19 columns
  -> data\processed\timeseries_features.parquet

$ python -m neuralretail.cli train
Training demand forecaster (Prophet)...
  -> MAPE=0.3415, RMSE=1707.30
Training churn classifier (XGBoost)...
  -> AUC-ROC=1.0000, precision=1.0000, recall=1.0000, F1=1.0000
Training customer segmentation (KMeans)...
  -> k=3, silhouette=0.2353
Building inventory table (ABC + EOQ + dead-stock)...
  -> 9458 SKUs, A=4330 B=2370 C=2758, dead-stock=7961

$ python -m neuralretail.cli monitor
Drift: 0/6 columns drifted (share=0.00); ref=6,650 rows [2010-12-01 -> 2011-08-17], current=2,850 rows [2011-08-17 -> 2011-12-08]
  -> report\drift_report.html
```

## 5. Known limitations and follow-ups

- **No live alerting.** The drift report is the artefact; the
  threshold-based paging to Slack / PagerDuty is a future step.
  Implementing it is a 30-line Airflow `BranchPythonOperator` that
  reads the `drift_report.summary.json` and triggers a callback if
  `drift_share > 0.30`.
- **Airflow is a stub.** `dags/neuralretail_pipeline.py` is a
  BashOperator skeleton that calls the same `make` targets the
  single-laptop pipeline uses. A real Airflow deployment would
  import the model modules directly and would not shell out.
- **Drift detection is per-column only.** No multivariate drift
  (e.g. domain classifier on the feature matrix). For the
  portfolio scope this is intentional; a production rollout would
  add the `DomainClassifierPreset` from Evidently.
- **Synthetic data shows no drift.** The generator is stationary,
  so the 70/30 split sees the same distribution. A real Online
  Retail II ingest — where new customers enter and old SKUs churn
  out — will almost certainly show drift on `Country` and possibly
  `Quantity`. The detector was verified to fire when drift is
  injected (see § 1.2).
- **`docker-compose.yml` carries an obsolete top-level `version`
  key** that prints a warning. It does not affect functionality.
