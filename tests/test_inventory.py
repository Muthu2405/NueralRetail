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
