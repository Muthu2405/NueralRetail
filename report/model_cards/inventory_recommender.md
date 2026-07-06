# Model card — `neuralretail_inventory_recommender`

| Field | Value |
|---|---|
| **Model type** | ABC classification + EOQ formula + dead-stock flag |
| **Registered as** | `neuralretail_inventory_recommender` |
| **Primary metric** | n_skus (data coverage — more SKUs = more representative) |
| **Spec target** | *(none — this is a deterministic table, not a learned model)* |
| **Latest measured** | n_skus = **9,458** · A/B/C = 4330 / 2370 / 2758 · dead-stock = **84.17 %** |
| **Production version** | v8 |

## Training data

- Source: cleaned transactions (`data/processed/cleaned.parquet`).
- Aggregation: per-`StockCode` revenue + unit counts over the full
  data window.
- Lookback for dead-stock: configurable (default 90 days since last sale).

## How it works

1. **ABC classification** — cumulative revenue % per SKU; top 80 % = A,
   next 15 % = B, last 5 % = C (Pareto-style).
2. **EOQ** — `sqrt(2 * annual_demand * order_cost / holding_cost_pct * unit_cost)`.
   Holding cost defaults to 20 %, ordering cost to $50 / order.
   Both are overridable via `Settings`.
3. **Dead-stock flag** — any SKU with zero sales in the lookback
   window.
4. **Reorder recommendation** — `max(0, safety_stock - current_stock)`
   where safety_stock is a function of lead time and demand variance.

The output is a CSV table (`inventory_table.csv`) and a registered
MLflow model (the recommendation function as a Python model).

## Metrics

| Metric | Value | Notes |
|---|---|---|
| n_skus | 9,458 | Distinct StockCodes in the cleaned data. |
| n_class_a | 4,330 | High-revenue SKUs (~46 %). |
| n_class_b | 2,370 | Mid-revenue. |
| n_class_c | 2,758 | Long tail. |
| dead-stock % | 84.17 % | SKUs with no sales in the last 90 days. |

> The high dead-stock % is partly a feature of the synthetic
> generator (it produces sparse, single-purchase SKUs to mirror
> the real dataset's long tail) and partly a real signal that
> Online Retail II has thousands of one-off gift items.

## Intended use

- **Inventory Health** dashboard page — ABC pie + reorder table.
- API endpoint `POST /inventory/reorder` returns the top-N A-class
  SKUs that need reordering now, ordered by EOQ.

## Limitations

- The model assumes `current_stock` and `lead_time_days` are
  available per SKU. In this single-laptop build they are derived
  from the synthetic data; in a real deployment they would come
  from the warehouse management system.
- No seasonality adjustment. A SKU with a December spike looks
  identical to a SKU with steady demand — a Prophet forecast
  per-SKU would fix this.
- EOQ assumes independent demand. For substitute / complementary
  SKUs the model can over- or under-order.
- Holding cost (20 %) and ordering cost ($50) are industry-typical
  defaults, not company-specific. They should be calibrated from
  the buyer's finance team.

## How to retrain

```bash
make train       # rebuilds the inventory table + re-registers the model
make promote     # aliases the run with the most SKUs to 'Production'
```
