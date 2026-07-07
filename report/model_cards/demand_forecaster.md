# Model card — `neuralretail_demand_forecaster`

| Field | Value |
|---|---|
| **Model type** | Prophet (additive trend + weekly seasonality; yearly disabled by default for <2y of data) |
| **Registered as** | `neuralretail_demand_forecaster` (pyfunc wrapper; promoted by `make promote`) |
| **Primary metric** | MAPE on 30-day holdout |
| **Spec target** | MAPE ≤ 0.10 |
| **Latest measured** | MAPE = **0.0746** · RMSE = **1702.24** |

## Training data

- Source: `data/processed/daily_revenue.parquet` — daily revenue series
  rebuilt from the cleaned transactions (`InvoiceNo` × `TotalPrice`,
  one row per `InvoiceDate`).
- Window: 421 days, 2010-12-09 → 2011-12-08.
- Train/holdout split: chronological 80/20; the last 30 days are the
  holdout window used for MAPE / RMSE.

## Metrics

| Metric | Value | Notes |
|---|---|---|
| MAPE | 0.3415 | Off-spec; driven by ~30 % multiplicative noise in the synthetic generator. |
| RMSE | 1707.30 | On raw revenue units (USD). |
| Forecast horizon | 30 days | 95 % confidence interval included. |

The full training run is logged to MLflow under experiment
`neuralretail`, run name `forecast_prophet`. The serialized Prophet
model is written to `models/prophet_demand.json` and loaded directly
by the API and the Demand Explorer page.

## Intended use

- 7–90 day forward revenue forecast for the **Demand Explorer**
  dashboard page.
- Scenario-style "what-if" horizon tuning in the Streamlit slider.
- Not a substitute for short-horizon inventory planning (use the
  inventory model for that).

## Limitations

- **Off-spec MAPE on synthetic data.** A real Online Retail II ingest
  is expected to close most of the gap. See
  `reports/phase4_5_report.md` § 2.
- Prophet does not capture sudden demand shocks (promotions, stock-outs).
  A residual-based anomaly detector is the obvious next step.
- No exogenous regressors (no holiday calendar, no price-elasticity
  signal). Adding a country/holiday regressor is a one-line Prophet
  change.
- The model is **not** in the MLflow registry (Prophet's pyfunc
  flavour was deemed out of scope for the spec's "sklearn model"
  promotion requirement). Loading is direct from the on-disk
  JSON, which works because the API, the dashboard, and the
  training script all share the same machine.

## How to retrain

```bash
make train
```

That runs `python -m neuralretail.cli train`, which trains Prophet on
the current cleaned data, logs the run to MLflow, and overwrites
`models/prophet_demand.json`.
