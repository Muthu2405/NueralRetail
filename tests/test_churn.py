"""Tests for the churn classifier.

Covers the feature builder (``_behavioural_features`` and
``build_training_table``), the label rule (Recency > inactivity
days), and end-to-end training on a small fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuralretail.models.churn import (
    FEATURE_COLUMNS,
    _behavioural_features,
    build_training_table,
)


def _row(customer, invoice, qty, price, date, country="United Kingdom", stock="X"):
    return {
        "InvoiceNo": invoice,
        "StockCode": stock,
        "Description": "X",
        "Quantity": qty,
        "InvoiceDate": pd.Timestamp(date),
        "UnitPrice": price,
        "CustomerID": customer,
        "Country": country,
        "TotalPrice": qty * price,
    }


def test_behavioural_features_known_values():
    """avg_basket_size = mean(TotalPrice), unique_products = nunique(StockCode)."""
    rows = [
        _row(1, "A1", 2, 1.0, "2011-01-01", stock="S1"),
        _row(1, "A2", 3, 1.0, "2011-01-05", stock="S1"),
        _row(1, "A3", 1, 1.0, "2011-01-10", stock="S2"),
    ]
    df = pd.DataFrame(rows)
    out = _behavioural_features(df).set_index("CustomerID")
    # 3 invoices, all 2*1=2, 3*1=3, 1*1=1 → mean 2.0
    assert out.loc[1, "avg_basket_size"] == pytest.approx(2.0)
    # 2 distinct StockCodes
    assert out.loc[1, "unique_products"] == 2
    # avg_days_between: between 2011-01-01/01-05/01-10 — gaps 4 and 5
    assert out.loc[1, "avg_days_between"] == pytest.approx(4.5)
    # is_uk
    assert out.loc[1, "is_uk"] == 1


def test_behavioural_features_non_uk_customer():
    rows = [_row(2, "B1", 1, 5.0, "2011-02-01", country="Germany")]
    df = pd.DataFrame(rows)
    out = _behavioural_features(df).set_index("CustomerID")
    assert out.loc[2, "is_uk"] == 0


def test_behavioural_features_handles_single_invoice():
    """A customer with one invoice should still get a row; avg_days_between is NaN."""
    rows = [_row(3, "C1", 1, 1.0, "2011-03-01")]
    df = pd.DataFrame(rows)
    out = _behavioural_features(df)
    assert len(out) == 1
    # NaN is fine here; build_training_table fills it with Recency.
    assert np.isnan(out.iloc[0]["avg_days_between"])


def test_churn_label_uses_recency_threshold():
    """churned = (Recency > inactivity_days)."""
    rows = [
        _row(1, "A1", 1, 1.0, "2011-01-01"),  # last = Jan 1
        _row(2, "B1", 1, 1.0, "2010-10-01"),  # last = Oct 1 (older)
    ]
    transactions = pd.DataFrame(rows)
    rfm = pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "Recency": [10, 100],  # customer 1 recent, customer 2 stale
            "Frequency": [1, 1],
            "Monetary": [1.0, 1.0],
        }
    )
    table = build_training_table(
        transactions, rfm, snapshot_date=pd.Timestamp("2011-01-11"), inactivity_days=90
    ).set_index("CustomerID")
    assert table.loc[1, "churned"] == 0
    assert table.loc[2, "churned"] == 1


def test_build_training_table_has_all_feature_columns():
    rows = [_row(1, "A1", 1, 1.0, "2011-01-01")]
    transactions = pd.DataFrame(rows)
    rfm = pd.DataFrame(
        {"CustomerID": [1], "Recency": [1], "Frequency": [1], "Monetary": [1.0]}
    )
    table = build_training_table(
        transactions, rfm, snapshot_date=pd.Timestamp("2011-01-11")
    )
    for col in FEATURE_COLUMNS + ["churned"]:
        assert col in table.columns


def test_build_training_table_fills_nans_for_single_buyer():
    """A customer with one invoice has NaN avg_days_between; it should
    be filled with their Recency by build_training_table so the model
    sees a numeric value."""
    rows = [_row(1, "A1", 1, 1.0, "2011-01-01")]
    transactions = pd.DataFrame(rows)
    rfm = pd.DataFrame(
        {"CustomerID": [1], "Recency": [10], "Frequency": [1], "Monetary": [1.0]}
    )
    table = build_training_table(
        transactions, rfm, snapshot_date=pd.Timestamp("2011-01-11")
    )
    row = table.iloc[0]
    assert pd.notna(row["avg_days_between"])
    assert row["avg_days_between"] == 10


def test_end_to_end_train_small_synthetic():
    """Quick sanity check: train on a tiny synthetic frame and assert
    the metrics are reasonable. AUC-ROC should be > 0.5 (better than
    random) on at least 30 customers with both classes."""
    from neuralretail.models.churn import train as churn_train

    rng = np.random.default_rng(0)
    rows = []
    for cid in range(50):
        # Churned customers (recency > 90)
        for d in range(rng.integers(120, 250)):
            rows.append(
                _row(
                    cid,
                    f"C{cid}-{d}",
                    rng.integers(1, 5),
                    round(float(rng.uniform(1, 20)), 2),
                    pd.Timestamp("2010-12-01") + pd.Timedelta(days=int(d)),
                )
            )
    # Active customers (recency < 30)
    for cid in range(50, 100):
        for d in range(rng.integers(5, 25)):
            rows.append(
                _row(
                    cid,
                    f"A{cid}-{d}",
                    rng.integers(1, 5),
                    round(float(rng.uniform(1, 20)), 2),
                    pd.Timestamp("2011-11-15") - pd.Timedelta(days=int(d)),
                )
            )
    transactions = pd.DataFrame(rows)
    # Build RFM from the transactions
    from neuralretail.features.rfm import compute_rfm

    rfm = compute_rfm(transactions, snapshot_date=pd.Timestamp("2011-12-09"))
    res = churn_train(transactions, rfm)
    assert res.metrics["auc_roc"] >= 0.5
    assert res.metrics["n_train"] > 0
    assert res.metrics["n_test"] > 0
