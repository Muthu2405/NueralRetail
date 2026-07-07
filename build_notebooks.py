"""Build the 5 deliverable notebooks for the NeuralRetail platform.

The notebooks are plain ``.ipynb`` JSON files with a few code + markdown
cells each, focused on running the actual production modules on the
artefacts produced by ``make pipeline``.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path("notebooks")
NB_DIR.mkdir(exist_ok=True)


def cell(cell_type: str, source: str, *, outputs: list | None = None) -> dict:
    """Build a notebook cell. Markdown cells have no outputs."""
    out: dict = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True) if isinstance(source, str) else source,
    }
    if cell_type == "code":
        out["execution_count"] = None
        out["outputs"] = outputs or []
    return out


def md(text: str) -> dict:
    return cell("markdown", text)


def code(text: str) -> dict:
    return cell("code", text)


def notebook(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NB_DIR / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote notebooks/{name} ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# 01_eda.ipynb — exploratory data analysis on cleaned.parquet
# ---------------------------------------------------------------------------

notebook(
    "01_eda.ipynb",
    [
        md(
            "# 01 — Exploratory Data Analysis\n\n"
            "Loads `data/processed/cleaned.parquet` (produced by `make data`)\n"
            "and inspects the schema, distributions, top countries, and the\n"
            "daily revenue series.\n\n"
            "**Prerequisite:** run `make pipeline` first."
        ),
        code(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "\n"
            "plt.rcParams['figure.figsize'] = (10, 4)\n"
            "CLEAN = Path('data/processed/cleaned.parquet')\n"
            "df = pd.read_parquet(CLEAN)\n"
            "print(f'Rows: {len(df):,}')\n"
            "df.head()\n"
        ),
        md(
            "## Quantity, UnitPrice, TotalPrice distributions\n\n"
            "Truncate Quantity at the 99th percentile so the histogram is not\n"
            "dominated by bulk-order outliers."
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "for ax, col in zip(axes, ['Quantity', 'UnitPrice', 'TotalPrice']):\n"
            "    s = df[col]\n"
            "    upper = s.quantile(0.99)\n"
            "    s = s[s <= upper]\n"
            "    ax.hist(s, bins=40, color='steelblue', edgecolor='white')\n"
            "    ax.set_title(f'{col} (≤p99 = {upper:.2f})')\n"
            "    ax.set_xlabel(col)\n"
            "    ax.set_ylabel('rows')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md("## Top 10 countries by row count"),
        code(
            "top = df['Country'].value_counts().head(10)\n"
            "ax = top.plot.barh(color='teal')\n"
            "ax.set_xlabel('rows')\n"
            "ax.set_title('Top 10 countries (rows)')\n"
            "ax.invert_yaxis()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "top\n"
        ),
        md("## Daily revenue over the cleaned window"),
        code(
            "daily = (\n"
            "    df.assign(Revenue=df['Quantity'] * df['UnitPrice'])\n"
            "      .set_index('InvoiceDate')\n"
            "      .resample('D')['Revenue']\n"
            "      .sum()\n"
            ")\n"
            "ax = daily.plot(color='coral', linewidth=1.2)\n"
            "ax.set_title('Daily revenue (cleaned)')\n"
            "ax.set_ylabel('GBP')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print(f'Days: {len(daily)}, mean: {daily.mean():.0f}, std: {daily.std():.0f}')\n"
        ),
    ],
)

# ---------------------------------------------------------------------------
# 02_feature_engineering.ipynb — RFM + time-series features
# ---------------------------------------------------------------------------

notebook(
    "02_feature_engineering.ipynb",
    [
        md(
            "# 02 — Feature Engineering\n\n"
            "Loads `rfm.parquet` and `timeseries_features.parquet` and\n"
            "visualises the per-customer RFM distribution plus the time-series\n"
            "lag/rolling features used by the forecasting model."
        ),
        code(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "plt.rcParams['figure.figsize'] = (10, 4)\n"
            "rfm = pd.read_parquet('data/processed/rfm.parquet')\n"
            "ts = pd.read_parquet('data/processed/timeseries_features.parquet')\n"
            "print('rfm:', rfm.shape, '|', list(rfm.columns))\n"
            "print('ts :', ts.shape, '|', list(ts.columns))\n"
        ),
        md("## RFM distributions"),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "for ax, col in zip(axes, ['Recency', 'Frequency', 'Monetary']):\n"
            "    s = rfm[col]\n"
            "    if col == 'Monetary':\n"
            "        s = s[s <= s.quantile(0.99)]\n"
            "    ax.hist(s, bins=40, color='steelblue', edgecolor='white')\n"
            "    ax.set_title(f'{col} (n={len(rfm):,})')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "## Time-series feature columns\n\n"
            "Lag/rolling features are built by `features/timeseries.py` from\n"
            "the cleaned daily revenue series."
        ),
        code(
            "lag_cols = [c for c in ts.columns if c.startswith('lag_') or c.startswith('roll_')]\n"
            "print('lag/roll columns:', lag_cols)\n"
            "if lag_cols:\n"
            "    ax = ts[lag_cols[:3]].plot(subplots=True, figsize=(10, 6), linewidth=1.0)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
        ),
    ],
)

# ---------------------------------------------------------------------------
# 03_forecasting.ipynb — Prophet fit + 30-day holdout
# ---------------------------------------------------------------------------

notebook(
    "03_forecasting.ipynb",
    [
        md(
            "# 03 — Demand Forecasting\n\n"
            "Runs the production `forecasting.train` on the cleaned daily\n"
            "revenue series and visualises the actual vs forecast with the\n"
            "confidence band. The 30-day holdout MAPE is the headline spec\n"
            "metric for this model."
        ),
        code(
            "import logging\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "\n"
            "from neuralretail.models.forecasting import train as fc_train, _mape, _prepare\n"
            "\n"
            "logging.getLogger('neuralretail').setLevel(logging.WARNING)\n"
            "logging.getLogger('cmdstanpy').setLevel(logging.WARNING)\n"
            "\n"
            "daily = pd.read_parquet('data/processed/daily_revenue.parquet').reset_index()\n"
            "daily = daily.rename(columns={'InvoiceDate': 'ds', 'Revenue': 'y'})\n"
            "print('daily shape:', daily.shape, '| range:', daily['ds'].min(), '→', daily['ds'].max())\n"
        ),
        md(
            "## Fit Prophet with the same hyper-params as `make train`\n\n"
            "Holdout = last 30 days; training = everything before."
        ),
        code(
            "HORIZON = 30\n"
            "train_df = daily.iloc[:-HORIZON].copy()\n"
            "test_df = daily.iloc[-HORIZON:].copy()\n"
            "print('train rows:', len(train_df), '| test rows:', len(test_df))\n"
            "\n"
            "res = fc_train(train_df, horizon_days=0, run_name='nb_forecast')\n"
            "future = res.model.make_future_dataframe(periods=HORIZON, freq='D')\n"
            "fc = res.model.predict(future)\n"
            "fc_test = fc.tail(HORIZON).set_index('ds')\n"
            "mape = _mape(test_df.set_index('ds')['y'], fc_test['yhat'])\n"
            "print(f'30-day holdout MAPE = {mape:.4f}')\n"
        ),
        md("## Actual vs forecast with 80% confidence interval"),
        code(
            "ax = daily.set_index('ds')['y'].plot(label='actual', color='black', linewidth=1.0)\n"
            "fc.set_index('ds')['yhat'].plot(ax=ax, label='forecast', color='coral')\n"
            "ax.fill_between(\n"
            "    fc['ds'], fc['yhat_lower'], fc['yhat_upper'],\n"
            "    color='coral', alpha=0.2, label='80% CI',\n"
            ")\n"
            "ax.axvline(test_df['ds'].iloc[0], color='grey', linestyle='--', label='holdout start')\n"
            "ax.set_title(f'Prophet forecast (30-day holdout MAPE = {mape:.4f})')\n"
            "ax.set_ylabel('Revenue (GBP)')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
    ],
)

# ---------------------------------------------------------------------------
# 04_churn_model.ipynb — train + SHAP summary
# ---------------------------------------------------------------------------

notebook(
    "04_churn_model.ipynb",
    [
        md(
            "# 04 — Churn Classifier\n\n"
            "Trains the XGBoost churn model on RFM + behavioural features\n"
            "(see `models/churn.py`) and visualises the SHAP feature\n"
            "importance and the AUC-ROC curve."
        ),
        code(
            "import logging\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "from neuralretail.data.ingest import load_raw\n"
            "from neuralretail.data.clean import clean\n"
            "from neuralretail.features.rfm import compute_rfm\n"
            "from neuralretail.models.churn import train as churn_train\n"
            "from sklearn.metrics import roc_auc_score, roc_curve\n"
            "\n"
            "logging.getLogger('neuralretail').setLevel(logging.WARNING)\n"
        ),
        md("## Build the training table"),
        code(
            "raw = load_raw()\n"
            "cleaned, _ = clean(raw)\n"
            "rfm = compute_rfm(cleaned)\n"
            "res = churn_train(cleaned, rfm, run_name='nb_churn')\n"
            "print('metrics:', {k: round(v, 4) for k, v in res.metrics.items() if isinstance(v, float)})\n"
            "res.feature_importances.head()\n"
        ),
        md("## Feature importance (XGBoost gain)"),
        code(
            "imp = res.feature_importances.sort_values('importance')\n"
            "ax = imp.plot.barh(x='feature', y='importance', color='teal', legend=False)\n"
            "ax.set_title('XGBoost feature importance')\n"
            "ax.set_xlabel('gain')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "## AUC-ROC\n\n"
            "Re-fit on the same train/test split so we can plot the curve."
        ),
        code(
            "from neuralretail.models.churn import build_training_table, FEATURE_COLUMNS\n"
            "from sklearn.model_selection import train_test_split\n"
            "table = build_training_table(cleaned, rfm)\n"
            "X = table[FEATURE_COLUMNS].fillna(0)\n"
            "y = table['churned']\n"
            "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n"
            "proba = res.model.predict_proba(Xte)[:, 1]\n"
            "auc = roc_auc_score(yte, proba)\n"
            "fpr, tpr, _ = roc_curve(yte, proba)\n"
            "plt.plot(fpr, tpr, color='coral', label=f'AUC = {auc:.4f}')\n"
            "plt.plot([0, 1], [0, 1], color='grey', linestyle='--')\n"
            "plt.xlabel('FPR')\n"
            "plt.ylabel('TPR')\n"
            "plt.title('ROC — churn classifier')\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
    ],
)

# ---------------------------------------------------------------------------
# 05_segmentation_inventory.ipynb — KMeans + persona scatter + ABC
# ---------------------------------------------------------------------------

notebook(
    "05_segmentation_inventory.ipynb",
    [
        md(
            "# 05 — Segmentation & Inventory\n\n"
            "Trains the KMeans persona model and the inventory recommender\n"
            "in one pass so we can compare the ABC distribution and the\n"
            "top-reorder SKUs side by side."
        ),
        code(
            "import logging\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "from neuralretail.data.ingest import load_raw\n"
            "from neuralretail.data.clean import clean\n"
            "from neuralretail.features.rfm import compute_rfm\n"
            "from neuralretail.models.segmentation import train as seg_train\n"
            "from neuralretail.models.inventory import train as inv_train\n"
            "\n"
            "logging.getLogger('neuralretail').setLevel(logging.WARNING)\n"
        ),
        md("## Train segmentation"),
        code(
            "raw = load_raw()\n"
            "cleaned, _ = clean(raw)\n"
            "rfm = compute_rfm(cleaned)\n"
            "seg = seg_train(rfm, k_min=4, k_max=8, run_name='nb_seg')\n"
            "print(f'k = {seg.k} | silhouette = {seg.metrics[\"silhouette\"]:.4f}')\n"
            "seg.summary\n"
        ),
        md(
            "## Recency × Monetary scatter coloured by cluster\n\n"
            "Each point is one customer; the persona name comes from the\n"
            "post-training cluster → persona mapping."
        ),
        code(
            "scored = rfm.copy()\n"
            "scored['cluster'] = seg.labels\n"
            "scored['persona'] = scored['cluster'].map(seg.persona_map)\n"
            "fig, ax = plt.subplots(figsize=(8, 6))\n"
            "for persona, sub in scored.groupby('persona'):\n"
            "    ax.scatter(sub['Recency'], sub['Monetary'], label=persona, alpha=0.6, s=14)\n"
            "ax.set_xlabel('Recency (days)')\n"
            "ax.set_ylabel('Monetary (GBP)')\n"
            "ax.set_yscale('log')\n"
            "ax.set_title(f'Customer segments (k={seg.k}, silhouette={seg.metrics[\"silhouette\"]:.3f})')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md("## Train inventory and inspect ABC distribution"),
        code(
            "inv = inv_train(cleaned, run_name='nb_inv')\n"
            "print('inventory rows:', len(inv.table))\n"
            "print('metrics:', {k: round(v, 4) if isinstance(v, float) else v for k, v in inv.metrics.items()})\n"
            "abc_counts = inv.table['ABC'].value_counts()\n"
            "ax = abc_counts.plot.pie(autopct='%1.0f%%', colors=['#2a9d8f', '#e9c46a', '#f4a261'])\n"
            "ax.set_title('ABC class distribution')\n"
            "ax.set_ylabel('')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md("## Top 10 reorder candidates"),
        code(
            "top = inv.table.sort_values('Revenue', ascending=False).head(10)\n"
            "top[['StockCode', 'Description', 'ABC', 'UnitsSold', 'Revenue', 'EOQ', 'IsDeadStock']]\n"
        ),
    ],
)

print("done.")
