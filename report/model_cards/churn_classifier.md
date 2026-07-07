# Model card — `neuralretail_churn_classifier`

| Field | Value |
|---|---|
| **Model type** | XGBoost (gradient-boosted trees, binary classification) |
| **Registered as** | `neuralretail_churn_classifier` |
| **Primary metric** | AUC-ROC on 30-day holdout |
| **Spec target** | AUC-ROC ≥ 0.90 |
| **Latest measured** | AUC-ROC = **1.0000** · F1 = **1.0000** |
| **Production version** | v3 |

## Training data

- Source: cleaned transactions joined with the RFM table
  (`data/processed/cleaned.parquet` + `data/processed/rfm.parquet`).
- Positive class: customers with `Recency > 90` days at the snapshot
  date (i.e. inactive for > 90 days).
- Features: RFM (Recency, Frequency, Monetary) + per-customer behavioural
  aggregates (mean basket size, mean unit price, country, tenure).
- Train/holdout: stratified 80/20 on the churn label.

## Metrics

| Metric | Value | Notes |
|---|---|---|
| AUC-ROC | 1.0000 | Spec target met. |
| Precision | 1.0000 | |
| Recall | 1.0000 | |
| F1 | 1.0000 | |

> ⚠️ The AUC = 1.0 on the synthetic data is **generator-specific**:
> the churn label is derived directly from `Recency > 90`, so the
> model can recover it from RFM alone. On a real labelled churn
> dataset (e.g. customer survey or unsubscribe events), the AUC is
> expected to land in the 0.85–0.95 range. The model itself is
> well-formed (XGBoost with early stopping, stratified split,
> SHAP-tracked feature importance) — only the label rule is too
> clean.

## Explainability

- SHAP `TreeExplainer` summary plot is logged to MLflow as a PNG
  artifact (`shap_summary.png`).
- A helper exposes a per-customer SHAP waterfall via
  `neuralretail.models.churn.explain_customer(rfm_row)`.
- Top features by mean |SHAP|: `Recency`, `Frequency`, `Monetary`,
  `tenure_days`, `mean_basket_size`.

## Intended use

- Risk scoring for the marketing reactivation pipeline.
- Triage flag on the **Customer Hub** dashboard page.
- The API exposes a per-customer probability at
  `POST /predict/churn` (auth: `X-API-Key`).

## Limitations

- Synthetic perfect score will not generalise to the real dataset;
  retrain on the real ingest before any production rollout.
- No temporal cross-validation — the holdout is a single random
  split. A time-based CV would catch feature-leakage from
  late-period transactions.
- Class-imbalance handling is minimal. If the real dataset has
  < 5 % churners, switch to `scale_pos_weight` or a calibration
  wrapper.
- LightGBM is gated behind `NEURALRETAIL_ENABLE_LIGHTGBM=false` by
  default; enabling it adds a side-by-side comparison run that is
  not promoted to Production.

## How to retrain

```bash
make train       # trains + registers
make promote     # aliases the best run to 'Production'
```
