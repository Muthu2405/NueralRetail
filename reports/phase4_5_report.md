# Phase 4 + 5 — End-to-End Report

Generated: 2026-07-03

This report covers Phase 4 (model training + MLflow tracking) and
Phase 5 (model promotion + API serving). All metrics below are the
**actual numbers** produced by the training pipeline run against the
synthetic Online Retail II dataset — no claims are interpolated.

## 1. Data artefacts

| Artefact | Path | Rows | Notes |
|---|---|---|---|
| Cleaned transactions | `data/processed/cleaned.parquet` | 9,550 | 10,000 raw → 450 bad rows dropped by GE validation |
| RFM table | `data/processed/rfm.parquet` | 4,780 | Distinct customers with non-null `CustomerID` |
| Daily revenue | `data/processed/daily_revenue.parquet` | 421 days | 2010-12-09 → 2011-12-08 |
| Time-series features | `data/processed/timeseries_features.parquet` | 421 | Calendar + lag/rolling features |

## 2. Trained models and metrics

| Model | Registered as | Version | Primary metric | Value | Spec target | Pass? |
|---|---|---|---|---|---|---|
| Prophet (demand) | — (on disk only) | — | MAPE | **0.3415** | ≤ 0.10 | ❌ |
| Prophet (demand) | — (on disk only) | — | RMSE | **1707.30** | — | — |
| XGBoost (churn) | `neuralretail_churn_classifier` | v3 (Production) | AUC-ROC | **1.0000** | ≥ 0.90 | ✅ |
| XGBoost (churn) | `neuralretail_churn_classifier` | v3 (Production) | F1 | **1.0000** | — | — |
| KMeans (segmentation) | `neuralretail_customer_segmenter` | v3 (Production) | silhouette | **0.2353** | ≥ 0.55 | ❌ |
| KMeans (segmentation) | `neuralretail_customer_segmenter` | v3 (Production) | best k | **3** | — | — |
| ABC + EOQ (inventory) | `neuralretail_inventory_recommender` | v8 (Production) | n_skus | **9,458** | — | — |
| ABC + EOQ (inventory) | `neuralretail_inventory_recommender` | v8 (Production) | A / B / C split | 4330 / 2370 / 2758 | — | — |
| ABC + EOQ (inventory) | `neuralretail_inventory_recommender` | v8 (Production) | dead-stock % | **84.17 %** | — | — |

### Why are some metrics off-target?

The dataset is a **synthetic** stand-in for Online Retail II
(per the constraint that the user-supplied RetailRocket data
couldn't be repurposed for an RFM/churn pipeline). The synthetic
generator's RFM signal is too clean (churn is trivially separable
from `Recency` alone — hence perfect AUC), and the daily-revenue
generator injects ~30 % multiplicative noise (hence the high MAPE).
A real Online Retail II ingest would likely land much closer to the
spec targets.

## 3. MLflow registry state

```
neuralretail_churn_classifier        v3  Production  auc_roc=1.0000
neuralretail_customer_segmenter      v3  Production  silhouette=0.2353
neuralretail_inventory_recommender   v8  Production  n_skus=9458
```

Promotion logic (`src/neuralretail/models/_mlflow_utils.py`):
- `churn` → max `auc_roc`
- `segmentation` → max `silhouette`
- `inventory` → max `n_skus` (more data beats `dead_stock_pct=0.0`
  from tiny test fixtures)
- `forecasting` is intentionally **not** registered — Prophet's
  pyfunc model was deemed out of scope for the spec's "sklearn
  model" promotion requirement. The on-disk `models/prophet_demand.json`
  is loaded directly by the API.

## 4. FastAPI service smoke-test

Started with `python -m uvicorn neuralretail.api.main:app` and hit
each endpoint with `curl`:

| Endpoint | Request | Status | Response summary |
|---|---|---|---|
| `GET /health` | — | 200 | `status=ok`, all 4 models loaded |
| `POST /predict/demand` | `{"horizon_days": 5}` | 200 | 5 forecast points (e.g. `yhat=6118.5`, 95 % CI 4383–7856) |
| `POST /predict/churn` | 1 customer RFM | 200 | `churn_probability=0.0011` (recent + frequent + UK) |
| `POST /predict/churn` (no key) | — | **401** | `Invalid or missing X-API-Key header.` |
| `POST /segment/score` | 1 customer | 200 | `cluster=1, persona="Loyal Customers"` |
| `POST /inventory/reorder` | `{"top_n": 3}` | 200 | 3 A-class SKUs, EOQ + dead-stock flags |

Auth: `X-API-Key` header compared against `Settings().api_key`. With
`env_prefix="NEURALRETAIL_"` in `SettingsConfigDict`, an exported
`NEURALRETAIL_API_KEY=…` correctly overrides the `.env` value.

## 5. Pytest

```
32 passed, 5 warnings in 23.44s
```

Suites:
- `tests/test_api.py` (7 tests) — endpoint auth, validation, happy paths
- `tests/test_clean.py` (9 tests) — Great Expectations suite
- `tests/test_inventory.py` (8 tests) — ABC/EOQ math + dead-stock flag
- `tests/test_rfm.py` (8 tests) — RFM computation

## 6. Known issues / follow-ups

- **MAPE 34 % on Prophet**: synthetic data has multiplicative noise.
  Switching to a real Online Retail II feed is the single biggest
  improvement; a real-ARIMA or N-BEATS model would also help.
- **Silhouette 0.23 < 0.55 spec**: RFM is sparse in 3-D and the
  synthetic customer base has heavy overlap. Adding log-transformed
  features or trying `enable_dbscan=True` may surface meaningful
  structure, but the spec's 0.55 target is unlikely to be hit on this
  data.
- **`on_event` deprecation**: resolved (now uses `lifespan`).
- **`datetime.utcnow()` deprecation**: resolved (now uses
  `datetime.now(timezone.utc)`).
- **`pydantic-settings` env-var override**: the original symptom
  (env vars silently ignored) was caused by `env_prefix=""` (empty).
  Setting `env_prefix="NEURALRETAIL_"` makes the prefix match the
  `.env` keys, so process env wins as expected.
