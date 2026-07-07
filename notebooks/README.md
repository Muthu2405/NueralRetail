# NeuralRetail — Notebooks

Five Jupyter notebooks that mirror the production pipeline. Each
notebook reads the parquet artefacts produced by `make pipeline` and
calls the actual `neuralretail.*` modules — they are not standalone
analysis scripts.

| # | Notebook | Reads from | Trains |
|---|----------|------------|--------|
| 01 | [`01_eda.ipynb`](01_eda.ipynb) | `data/processed/cleaned.parquet` | — |
| 02 | [`02_feature_engineering.ipynb`](02_feature_engineering.ipynb) | `rfm.parquet`, `timeseries_features.parquet` | — |
| 03 | [`03_forecasting.ipynb`](03_forecasting.ipynb) | `daily_revenue.parquet` | `models.forecasting.train` |
| 04 | [`04_churn_model.ipynb`](04_churn_model.ipynb) | `cleaned.parquet` + RFM | `models.churn.train` |
| 05 | [`05_segmentation_inventory.ipynb`](05_segmentation_inventory.ipynb) | `cleaned.parquet` + RFM | `models.segmentation.train`, `models.inventory.train` |

## Run order

```bash
# 1. Build the artefacts (writes the parquet files the notebooks read)
make pipeline

# 2. Open the notebooks in Jupyter
jupyter lab notebooks/
```

If you have a real Online Retail II file in `data/raw/online_retail.csv`,
the `make pipeline` will read it instead of the synthetic fallback — the
notebooks work against either source.

## Notes

- Each notebook is intentionally short (≤ 10 cells) and focused on one
  slice of the pipeline. They re-use the production code paths so the
  numbers in the notebook output match the dashboard, the API, and the
  MLflow runs.
- The notebooks write figures to the active matplotlib backend; if
  you're running headless, set `MPLBACKEND=Agg` in the environment
  before starting Jupyter.
- The 30-day holdout MAPE in notebook 03 is the headline forecasting
  spec metric; the silhouette in notebook 05 is the headline
  segmentation spec metric. If either drifts outside the spec band,
  check the drift report at `report/drift_report.html` first.
