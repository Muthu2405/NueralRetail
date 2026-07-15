"""Inventory analytics: ABC classification + EOQ + dead-stock flag.

ABC by revenue contribution (Pareto-style):
- A: top SKUs by revenue that together cover 80% of total revenue
- B: next 15% of cumulative revenue
- C: bottom 5%

EOQ (Economic Order Quantity) uses the Wilson formula:
    EOQ = sqrt(2 * D * S / H)
where D = annual demand (units), S = ordering cost per order,
H = holding cost per unit per year (= holding_pct * unit_cost).

Dead-stock: SKUs with zero sales in the last ``dead_stock_days`` days.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from neuralretail.config import get_settings
from neuralretail.models._mlflow_utils import log_metrics, log_params, start_run

logger = logging.getLogger(__name__)

DEFAULT_ORDERING_COST = 50.0  # $ per order
DEFAULT_HOLDING_PCT = 0.25  # 25% of unit cost per year
DEFAULT_DEAD_STOCK_DAYS = 60


@dataclass
class InventoryResult:
    table: pd.DataFrame
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


def _abc_classify(revenue_sorted: pd.DataFrame) -> pd.Series:
    """Assign A/B/C by Pareto-style cumulative revenue thresholds.

    Convention: classify each row by where its *cumulative* share lands
    when the input is already sorted by Revenue descending.

        A: cumulative_share <= 0.80
        B: 0.80 <  cumulative_share <= 0.95
        C: cumulative_share >  0.95   (long tail)

    Edge case: a single dominant SKU can sit above 80% on its own. We
    guarantee the top row is always class A (champion) so the highest-
    revenue SKU is never mis-labelled as a long-tail item.
    """
    total = revenue_sorted["Revenue"].sum()
    if total <= 0:
        return pd.Series("C", index=revenue_sorted.index)
    cum = revenue_sorted["Revenue"].cumsum() / total

    classes = pd.Series("C", index=revenue_sorted.index, dtype=str)
    classes[cum <= 0.80] = "A"
    classes[(cum > 0.80) & (cum <= 0.95)] = "B"
    # Top row is always A — the highest-revenue SKU is the "champion".
    if len(classes) > 0:
        classes.iloc[0] = "A"
    return classes


# ---------------------------------------------------------------------------
# EOQ
# ---------------------------------------------------------------------------


def _eoq(annual_demand: float, unit_cost: float, ordering_cost: float, holding_pct: float) -> float:
    """Wilson EOQ. Returns 0 if demand or cost is non-positive."""
    if annual_demand <= 0 or unit_cost <= 0 or ordering_cost <= 0 or holding_pct <= 0:
        return 0.0
    h = holding_pct * unit_cost
    return float(np.sqrt(2.0 * annual_demand * ordering_cost / h))


# ---------------------------------------------------------------------------
# Public train() / predict()
# ---------------------------------------------------------------------------


def train(
    transactions: pd.DataFrame,
    *,
    ordering_cost: float = DEFAULT_ORDERING_COST,
    holding_pct: float = DEFAULT_HOLDING_PCT,
    dead_stock_days: int = DEFAULT_DEAD_STOCK_DAYS,
    reference_date: pd.Timestamp | None = None,
    run_name: str = "inventory_abc_eoq",
) -> InventoryResult:
    """Build a SKU-level inventory table with ABC class, EOQ, and dead-stock flag."""
    required = {"StockCode", "Description", "Quantity", "UnitPrice", "InvoiceDate"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions is missing required columns: {missing}")

    df = transactions.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Span of the data, in years, for annualising demand.
    span_days = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days + 1
    span_years = max(span_days / 365.25, 0.01)

    grp = df.groupby(["StockCode", "Description"], sort=False)
    table = pd.DataFrame(
        {
            "UnitsSold": grp["Quantity"].sum(),
            "Revenue": grp.apply(lambda g: float((g["Quantity"] * g["UnitPrice"]).sum()), include_groups=False),
            "AvgUnitPrice": grp["UnitPrice"].mean(),
            "LastSale": grp["InvoiceDate"].max(),
        }
    ).reset_index()
    table = table.sort_values("Revenue", ascending=False).reset_index(drop=True)
    table["ABC"] = _abc_classify(table)

    # Annualise units sold
    table["AnnualDemand"] = (table["UnitsSold"] / span_years).round(1)
    table["EOQ"] = table.apply(
        lambda r: _eoq(r["AnnualDemand"], r["AvgUnitPrice"], ordering_cost, holding_pct),
        axis=1,
    ).round(1)
    table["DaysSinceLastSale"] = (reference_date - table["LastSale"]).dt.days
    table["IsDeadStock"] = (table["DaysSinceLastSale"] > dead_stock_days).astype(int)
    table["OrderingCost"] = ordering_cost
    table["HoldingPct"] = holding_pct

    metrics = {
        "n_skus": float(len(table)),
        "n_class_a": float((table["ABC"] == "A").sum()),
        "n_class_b": float((table["ABC"] == "B").sum()),
        "n_class_c": float((table["ABC"] == "C").sum()),
        "n_dead_stock": float(table["IsDeadStock"].sum()),
        "dead_stock_pct": float(table["IsDeadStock"].mean()),
        "total_revenue": float(table["Revenue"].sum()),
        "span_years": float(span_years),
    }

    with start_run(run_name=run_name, tags={"model": "inventory"}):
        log_params(
            {
                "ordering_cost": ordering_cost,
                "holding_pct": holding_pct,
                "dead_stock_days": dead_stock_days,
            }
        )
        log_metrics(metrics)
        table.to_csv("inventory_table.csv", index=False)
        import mlflow

        mlflow.log_artifact("inventory_table.csv")
        # No model artifact for inventory; just the table.
        try:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=_InventoryPyFunc(table),
                registered_model_name="neuralretail_inventory_recommender",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not register inventory model: %s", exc)

    logger.info(
        "Inventory: %d SKUs, A=%d B=%d C=%d, dead-stock=%d",
        int(metrics["n_skus"]),
        int(metrics["n_class_a"]),
        int(metrics["n_class_b"]),
        int(metrics["n_class_c"]),
        int(metrics["n_dead_stock"]),
    )
    return InventoryResult(table=table, metrics=metrics)


def save(
    table: pd.DataFrame,
    path: str | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    """Persist the per-SKU inventory table to disk.

    Parameters
    ----------
    table
        Per-SKU DataFrame (one row per ``(StockCode, Description)``).
    path
        Output CSV path. Defaults to ``settings.models_dir / "inventory_table.csv"``.
    metrics
        Optional aggregate-metrics dict (the ``InventoryResult.metrics`` from
        :func:`train`). When provided, also written as a sidecar JSON
        (``inventory_metrics.json`` next to the CSV) so the API's
        ``/inventory/reorder`` response can populate its ``summary`` block
        without recomputing aggregates from the per-SKU table.

    Returns
    -------
    str
        The CSV path that was written.
    """
    import json

    settings = get_settings()
    path = path or str(settings.models_dir / "inventory_table.csv")
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    if metrics is not None:
        sidecar = csv_path.with_name("inventory_metrics.json")
        sidecar.write_text(json.dumps(metrics, indent=2))
        logger.info("Wrote inventory metrics sidecar to %s", sidecar)
    return str(csv_path)


def load_latest(path: str | None = None) -> pd.DataFrame:
    settings = get_settings()
    path = path or str(settings.models_dir / "inventory_table.csv")
    return pd.read_csv(path)


class _InventoryPyFunc(mlflow.pyfunc.PythonModel):
    """Trivial pyfunc that just holds the inventory table for MLflow registry."""

    def __init__(self, table: pd.DataFrame) -> None:
        super().__init__()
        self._table = table

    def predict(self, context: Any, model_input: list[Any]) -> pd.DataFrame:
        return self._table
