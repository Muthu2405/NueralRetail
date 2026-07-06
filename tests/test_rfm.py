"""Tests for features/rfm.py.

These tests use small in-memory DataFrames with fixed dates so the
expected Recency / Frequency / Monetary values are easy to verify.
"""

from __future__ import annotations

import pandas as pd
import pytest

from neuralretail.features.rfm import compute_rfm


def _row(customer: int, invoice: str, qty: int, price: float, date: str) -> dict:
    return {
        "InvoiceNo": invoice,
        "StockCode": "X",
        "Description": "X",
        "Quantity": qty,
        "InvoiceDate": pd.Timestamp(date),
        "UnitPrice": price,
        "CustomerID": customer,
        "Country": "UK",
        "TotalPrice": qty * price,
    }


def test_recency_frequency_monetary_known_values():
    """Three customers with known purchase patterns, snapshot pinned."""
    rows = [
        # Customer 1: three invoices, last on 2011-01-10
        _row(1, "A1", 1, 1.0, "2011-01-01"),
        _row(1, "A2", 3, 1.0, "2011-01-10"),
        _row(1, "A3", 2, 1.0, "2011-01-10"),
        # Customer 2: one invoice, on 2011-01-15, $5
        _row(2, "B1", 1, 5.0, "2011-01-15"),
        # Customer 3: one invoice, on 2011-01-05, $2
        _row(3, "C1", 1, 2.0, "2011-01-05"),
    ]
    df = pd.DataFrame(rows)
    snapshot = pd.Timestamp("2011-01-20")
    rfm = compute_rfm(df, snapshot_date=snapshot).set_index("CustomerID")

    assert set(rfm.index.tolist()) == {1, 2, 3}
    # Customer 1: last on Jan 10, snapshot Jan 20 -> Recency 10
    assert rfm.loc[1, "Recency"] == 10
    # Customer 1: three distinct invoices A1, A2, A3
    assert rfm.loc[1, "Frequency"] == 3
    assert rfm.loc[1, "Monetary"] == pytest.approx(6.0)

    # Customer 2: last on Jan 15 -> Recency 5
    assert rfm.loc[2, "Recency"] == 5
    assert rfm.loc[2, "Frequency"] == 1
    assert rfm.loc[2, "Monetary"] == pytest.approx(5.0)

    # Customer 3: last on Jan 5 -> Recency 15
    assert rfm.loc[3, "Recency"] == 15
    assert rfm.loc[3, "Frequency"] == 1
    assert rfm.loc[3, "Monetary"] == pytest.approx(2.0)


def test_frequency_counts_unique_invoices():
    rows = [
        _row(1, "INV-1", 2, 1.0, "2011-02-01"),
        _row(1, "INV-1", 3, 1.0, "2011-02-01"),  # same invoice
        _row(1, "INV-2", 1, 1.0, "2011-02-05"),
    ]
    rfm = compute_rfm(pd.DataFrame(rows), snapshot_date=pd.Timestamp("2011-02-10"))
    assert rfm.iloc[0]["Frequency"] == 2


def test_monetary_sums_totalprice():
    rows = [
        _row(1, "X", 1, 2.5, "2011-03-01"),
        _row(1, "Y", 4, 1.0, "2011-03-02"),
        _row(1, "Z", 2, 3.0, "2011-03-03"),
    ]
    rfm = compute_rfm(pd.DataFrame(rows), snapshot_date=pd.Timestamp("2011-03-10"))
    assert rfm.iloc[0]["Monetary"] == pytest.approx(2.5 + 4.0 + 6.0)


def test_monetary_clipped_nonnegative_with_refunds():
    """If TotalPrice is negative (refund line) we floor Monetary at 0."""
    rows = [
        _row(1, "X", 1, 10.0, "2011-04-01"),
        _row(1, "Y", 1, -5.0, "2011-04-02"),
    ]
    # We don't normally see negatives after cleaning, but be defensive.
    df = pd.DataFrame(rows)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    rfm = compute_rfm(df, snapshot_date=pd.Timestamp("2011-04-10"))
    assert rfm.iloc[0]["Monetary"] == pytest.approx(5.0)


def test_recency_uses_snapshot_date():
    rows = [_row(1, "X", 1, 1.0, "2011-01-01")]
    rfm_early = compute_rfm(pd.DataFrame(rows), snapshot_date=pd.Timestamp("2011-01-05"))
    rfm_late = compute_rfm(pd.DataFrame(rows), snapshot_date=pd.Timestamp("2011-02-05"))
    assert rfm_early.iloc[0]["Recency"] == 4
    assert rfm_late.iloc[0]["Recency"] == 35


def test_default_snapshot_is_day_after_last_purchase():
    rows = [
        _row(1, "X", 1, 1.0, "2011-05-01"),
        _row(2, "Y", 1, 1.0, "2011-05-10"),
    ]
    rfm = compute_rfm(pd.DataFrame(rows))  # no snapshot_date
    # Default snapshot = 2011-05-11
    assert rfm.set_index("CustomerID").loc[1, "Recency"] == 10
    assert rfm.set_index("CustomerID").loc[2, "Recency"] == 1


def test_first_and_last_purchase_present():
    rows = [
        _row(1, "X", 1, 1.0, "2011-06-01"),
        _row(1, "Y", 1, 1.0, "2011-06-15"),
        _row(1, "Z", 1, 1.0, "2011-06-08"),
    ]
    rfm = compute_rfm(pd.DataFrame(rows), snapshot_date=pd.Timestamp("2011-06-30"))
    row = rfm.iloc[0]
    assert row["FirstPurchase"] == pd.Timestamp("2011-06-01")
    assert row["LastPurchase"] == pd.Timestamp("2011-06-15")


def test_missing_columns_raises():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_rfm(df)
