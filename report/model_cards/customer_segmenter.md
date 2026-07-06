# Model card — `neuralretail_customer_segmenter`

| Field | Value |
|---|---|
| **Model type** | KMeans on standardised RFM (silhouette-selected k ∈ 3..8) |
| **Registered as** | `neuralretail_customer_segmenter` |
| **Primary metric** | Silhouette score on the training set |
| **Spec target** | Silhouette ≥ 0.55, 4–8 interpretable clusters |
| **Latest measured** | Silhouette = **0.2353** · best k = **3** |
| **Production version** | v3 |

## Training data

- Source: `data/processed/rfm.parquet` (one row per CustomerID).
- Features: `Recency` (days since last purchase), `Frequency` (count
  of unique invoices), `Monetary` (sum of `TotalPrice`).
- Preprocessing: `StandardScaler` (zero mean, unit variance). The
  fitted scaler is bundled in the same pickle as KMeans so inference
  uses the exact same transform.
- k selection: best k by silhouette across `k ∈ {3, 4, 5, 6, 7, 8}`.

## Metrics

| Metric | Value | Notes |
|---|---|---|
| Silhouette | 0.2353 | **Off-spec.** RFM in 3-D is sparse; the synthetic customer base has heavy overlap. |
| Best k | 3 | Below the 4-cluster spec floor. |

> The 0.55 silhouette target is unlikely to be hit on this data
> without log-transformed features or DBSCAN-style density clustering
> (gated behind `NEURALRETAIL_ENABLE_DBSCAN=true`). The personas are
> still useful as a coarse triage signal — they're computed
> programmatically from centroid rank, not hardcoded.

## Personas

Persona labels are **derived from the cluster centroids**, not
hardcoded. The algorithm:

1. Rank clusters by `Monetary` (high → low).
2. The top-Money cluster → "Champions".
3. The most recent + high-Frequency cluster → "Loyal Customers".
4. The oldest-Recency cluster → "At Risk".
5. The remainder → "Hibernating".

In the latest run (k=3) the mapping was:

| Cluster | Persona | n customers |
|---|---|---|
| 0 | Hibernating | … |
| 1 | Loyal Customers | … |
| 2 | Champions | … |

## Intended use

- Visualisation on the **Customer Hub** dashboard page
  (Recency vs Monetary scatter, coloured by cluster).
- Persona-level marketing segmentation in the persona summary CSV
  (`persona_summary.csv`).
- API endpoint `POST /segment/score` returns the cluster + persona
  for a single customer's RFM.

## Limitations

- KMeans assumes spherical, equally-sized clusters — a poor fit
  for the long-tailed RFM distribution typical of retail.
- No temporal stability check. A customer assigned to "Champions"
  in one training run may move to "Loyal" in the next; in production
  this should be smoothed (e.g. via majority vote over the last N runs).
- Personas are coarse (3–8 clusters max). A real deployment would
  overlay behavioural features (basket size, channel) to sharpen
  the segments.

## How to retrain

```bash
make train        # fits KMeans + StandardScaler, logs to MLflow
make promote      # aliases the best-silhouette run to 'Production'
```
