"""Tests for models/inventory.py — ABC boundaries and EOQ formula.

These tests use small in-memory DataFrames so the classification logic
can be verified without running the full training pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from neuralretail.models.inventory import _abc_classify, _eoq, train


def _row(stock: str, qty: int, price: float, date: str, country: str = "UK") -> dict:
    return {
        "InvoiceNo": "X",
        "StockCode": stock,
        "Description": stock,
        "Quantity": qty,
        "InvoiceDate": pd.Timestamp(date),
        "UnitPrice": price,
        "CustomerID": 1,
        "Country": country,
        "TotalPrice": qty * price,
    }


def test_abc_classify_three_classes():
    """3 SKUs with revenues 100, 20, 5 — cumulative shares are
    80%, 96%, 100% — row 1 = A, rows 2 and 3 = C (the 96% skips the B band)."""
    df = pd.DataFrame(
        [
            {"StockCode": "A", "Revenue": 100.0},
            {"StockCode": "B", "Revenue": 20.0},
            {"StockCode": "C", "Revenue": 5.0},
        ]
    )
    classes = _abc_classify(df).tolist()
    assert classes == ["A", "C", "C"]


def test_abc_classify_b_band_filled_with_smooth_distribution():
    """A smooth distribution fills all three bands.

    Revenues 100, 30, 20, 10, 5 (total 165). Cumulative: 60.6%, 78.8%,
    90.9%, 97.0%, 100%. So rows 1, 2 = A (cum 78.8% still <= 80%),
    row 3 = B (90.9% > 80% and <= 95%), rows 4, 5 = C.
    """
    df = pd.DataFrame(
        [
            {"StockCode": "S1", "Revenue": 100.0},
            {"StockCode": "S2", "Revenue": 30.0},
            {"StockCode": "S3", "Revenue": 20.0},
            {"StockCode": "S4", "Revenue": 10.0},
            {"StockCode": "S5", "Revenue": 5.0},
        ]
    )
    classes = _abc_classify(df).tolist()
    assert classes == ["A", "A", "B", "C", "C"]


def test_abc_classify_all_class_a_when_first_dominates():
    """A single SKU with 100% of revenue -> A; rest get C (the long tail)."""
    df = pd.DataFrame(
        [
            {"StockCode": "X", "Revenue": 1000.0},
            {"StockCode": "Y", "Revenue": 1.0},
            {"StockCode": "Z", "Revenue": 1.0},
        ]
    ).sort_values("Revenue", ascending=False).reset_index(drop=True)
    classes = _abc_classify(df).tolist()
    assert classes[0] == "A"
    # Y and Z share the remaining tiny slice — both C
    assert set(classes[1:]) == {"C"}


def test_abc_classify_handles_zero_revenue():
    df = pd.DataFrame(
        [
            {"StockCode": "X", "Revenue": 0.0},
            {"StockCode": "Y", "Revenue": 0.0},
        ]
    )
    classes = _abc_classify(df).tolist()
    assert classes == ["C", "C"]


def test_eoq_formula_known_value():
    """D=1000 units/yr, S=$50, H=25% of $4 = $1. EOQ = sqrt(2*1000*50/1) = sqrt(100000) = 316.23."""
    eoq = _eoq(annual_demand=1000, unit_cost=4.0, ordering_cost=50.0, holding_pct=0.25)
    assert eoq == pytest.approx(316.23, rel=1e-3)


def test_eoq_zero_for_invalid_inputs():
    assert _eoq(0, 1.0, 50.0, 0.25) == 0.0
    assert _eoq(100, 0.0, 50.0, 0.25) == 0.0
    assert _eoq(100, 1.0, 0.0, 0.25) == 0.0
    assert _eoq(100, 1.0, 50.0, 0.0) == 0.0
    assert _eoq(-100, 1.0, 50.0, 0.25) == 0.0


def test_train_produces_expected_columns():
    rows = [
        _row("A", 10, 5.0, "2011-01-01"),
        _row("A", 5, 5.0, "2011-01-15"),
        _row("B", 2, 3.0, "2011-01-10"),
        _row("C", 1, 1.0, "2011-01-20"),
    ]
    df = pd.DataFrame(rows)
    res = train(
        df,
        ordering_cost=50.0,
        holding_pct=0.25,
        dead_stock_days=60,
        reference_date=pd.Timestamp("2011-02-01"),
    )
    table = res.table
    expected_cols = {
        "StockCode",
        "Description",
        "UnitsSold",
        "Revenue",
        "AvgUnitPrice",
        "LastSale",
        "ABC",
        "AnnualDemand",
        "EOQ",
        "DaysSinceLastSale",
        "IsDeadStock",
        "OrderingCost",
        "HoldingPct",
    }
    assert expected_cols.issubset(set(table.columns))
    assert len(table) == 3
    # A is the only one sold in quantity, and is the highest-revenue SKU.
    assert "A" in set(table["StockCode"])
    # ABC classes should be among A, B, C
    assert set(table["ABC"]).issubset({"A", "B", "C"})
    # EOQ is non-negative
    assert (table["EOQ"] >= 0).all()


def test_dead_stock_flag():
    """Two SKUs: one sold recently, one not. Dead-stock = the stale one."""
    rows = [
        _row("FRESH", 1, 1.0, "2011-06-01"),
        _row("STALE", 1, 1.0, "2010-01-01"),
    ]
    df = pd.DataFrame(rows)
    res = train(df, dead_stock_days=60, reference_date=pd.Timestamp("2011-07-01"))
    table = res.table.set_index("StockCode")
    assert int(table.loc["FRESH", "IsDeadStock"]) == 0
    assert int(table.loc["STALE", "IsDeadStock"]) == 1
    assert res.metrics["n_dead_stock"] == 1.0


# ---------------------------------------------------------------------------
# Sidecar metrics JSON (consumed by the API's /inventory/reorder summary)
# ---------------------------------------------------------------------------


def test_save_writes_sidecar_metrics(tmp_path):
    """When `metrics` is provided, save() writes inventory_metrics.json
    next to the CSV with the same keys and float values.
    """
    from neuralretail.models import inventory as inv_mod

    table = pd.DataFrame(
        [
            {
                "StockCode": "A",
                "Description": "A",
                "UnitsSold": 10,
                "Revenue": 100.0,
                "AvgUnitPrice": 5.0,
                "LastSale": pd.Timestamp("2011-01-01"),
                "ABC": "A",
                "AnnualDemand": 10.0,
                "EOQ": 5.0,
                "DaysSinceLastSale": 0,
                "IsDeadStock": 0,
                "OrderingCost": 50.0,
                "HoldingPct": 0.25,
            }
        ]
    )
    metrics = {
        "n_skus": 1.0,
        "n_class_a": 1.0,
        "n_class_b": 0.0,
        "n_class_c": 0.0,
        "n_dead_stock": 0.0,
        "dead_stock_pct": 0.0,
        "total_revenue": 100.0,
        "span_years": 1.025,
    }
    csv_path = tmp_path / "inventory_table.csv"
    inv_mod.save(table, path=str(csv_path), metrics=metrics)

    sidecar = tmp_path / "inventory_metrics.json"
    assert sidecar.exists(), "sidecar JSON was not written"
    import json

    loaded = json.loads(sidecar.read_text())
    assert loaded == metrics


def test_save_without_metrics_does_not_write_sidecar(tmp_path):
    """The `metrics` arg is optional; without it, only the CSV is written."""
    from neuralretail.models import inventory as inv_mod

    table = pd.DataFrame(
        [
            {
                "StockCode": "A",
                "Description": "A",
                "UnitsSold": 1,
                "Revenue": 5.0,
                "AvgUnitPrice": 5.0,
                "LastSale": pd.Timestamp("2011-01-01"),
                "ABC": "A",
                "AnnualDemand": 1.0,
                "EOQ": 1.0,
                "DaysSinceLastSale": 0,
                "IsDeadStock": 0,
                "OrderingCost": 50.0,
                "HoldingPct": 0.25,
            }
        ]
    )
    csv_path = tmp_path / "inventory_table.csv"
    inv_mod.save(table, path=str(csv_path))

    sidecar = tmp_path / "inventory_metrics.json"
    assert not sidecar.exists(), "sidecar should not be written without metrics"


def test_api_inventory_reorder_returns_populated_summary():
    """Integration: with a real table + sidecar loaded into _State, the
    API /inventory/reorder response populates the summary block with all
    eight expected keys.
    """
    import os

    os.environ.setdefault("NEURALRETAIL_API_KEY", "test-key")

    from fastapi.testclient import TestClient

    from neuralretail.api.main import _State, app
    from neuralretail.api.schemas import InventoryRequest

    table = pd.DataFrame(
        [
            {
                "StockCode": "A",
                "Description": "Top SKU",
                "UnitsSold": 100.0,
                "Revenue": 1000.0,
                "AvgUnitPrice": 10.0,
                "LastSale": pd.Timestamp("2011-01-01"),
                "ABC": "A",
                "AnnualDemand": 100.0,
                "EOQ": 10.0,
                "DaysSinceLastSale": 0,
                "IsDeadStock": 0,
                "OrderingCost": 50.0,
                "HoldingPct": 0.25,
            }
        ]
    )
    metrics = {
        "n_skus": 1.0,
        "n_class_a": 1.0,
        "n_class_b": 0.0,
        "n_class_c": 0.0,
        "n_dead_stock": 0.0,
        "dead_stock_pct": 0.0,
        "total_revenue": 1000.0,
        "span_years": 1.025,
    }
    # Save + restore _State so we don't leak across tests.
    saved = (
        _State.inventory_table,
        _State.inventory_metrics,
        dict(_State.loaded),
    )
    try:
        # TestClient(app) runs the lifespan handler, which calls
        # _load_models_into_state and overwrites _State. We have to set
        # the test fixtures AFTER entering the lifespan context.
        with TestClient(app) as client:
            _State.inventory_table = table
            _State.inventory_metrics = metrics
            _State.loaded = {
                "forecasting": False,
                "churn": False,
                "segmentation": False,
                "inventory": True,
            }
            r = client.post(
                "/inventory/reorder",
                headers={"X-API-Key": "test-key"},
                json={"top_n": 5, "abc_filter": "ALL", "dead_stock_only": False},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        for key in (
            "n_skus",
            "n_class_a",
            "n_class_b",
            "n_class_c",
            "n_dead_stock",
            "dead_stock_pct",
            "total_revenue",
            "span_years",
        ):
            assert key in body["summary"], f"missing summary key {key!r}"
        assert body["summary"]["n_skus"] == 1.0
        assert body["summary"]["total_revenue"] == 1000.0
    finally:
        (
            _State.inventory_table,
            _State.inventory_metrics,
            _State.loaded,
        ) = saved
