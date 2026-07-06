"""Tests for data/clean.py — verify the cleaning rules.

These tests use small in-memory DataFrames; they do not require a real
Online Retail II file or a working GE installation to run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from neuralretail.data.clean import clean, validate_with_great_expectations


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_drops_cancelled_invoices():
    df = _make_df(
        [
            {
                "InvoiceNo": "536365",
                "StockCode": "84029G",
                "Description": "X",
                "Quantity": 6,
                "InvoiceDate": "2010-12-01 08:26:00",
                "UnitPrice": 2.55,
                "CustomerID": 17850.0,
                "Country": "United Kingdom",
            },
            {
                "InvoiceNo": "C536365",  # cancellation
                "StockCode": "84029G",
                "Description": "X",
                "Quantity": -6,
                "InvoiceDate": "2010-12-01 08:26:00",
                "UnitPrice": 2.55,
                "CustomerID": 17850.0,
                "Country": "United Kingdom",
            },
        ]
    )
    out, report = clean(df)
    assert len(out) == 1
    assert report.cancelled_dropped == 1
    assert out.iloc[0]["InvoiceNo"] == "536365"


def test_drops_null_customer_id():
    df = _make_df(
        [
            {
                "InvoiceNo": "536365",
                "StockCode": "84029G",
                "Description": "X",
                "Quantity": 6,
                "InvoiceDate": "2010-12-01 08:26:00",
                "UnitPrice": 2.55,
                "CustomerID": None,
                "Country": "United Kingdom",
            },
            {
                "InvoiceNo": "536366",
                "StockCode": "84029G",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01 09:00:00",
                "UnitPrice": 3.50,
                "CustomerID": 17850.0,
                "Country": "United Kingdom",
            },
        ]
    )
    out, report = clean(df)
    assert len(out) == 1
    assert report.null_customer_dropped == 1


def test_drops_nonpositive_quantity():
    df = _make_df(
        [
            {
                "InvoiceNo": "1",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 0,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 1.0,
                "CustomerID": 1.0,
                "Country": "UK",
            },
            {
                "InvoiceNo": "2",
                "StockCode": "A",
                "Description": "X",
                "Quantity": -3,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 1.0,
                "CustomerID": 1.0,
                "Country": "UK",
            },
            {
                "InvoiceNo": "3",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 1.0,
                "CustomerID": 1.0,
                "Country": "UK",
            },
        ]
    )
    out, report = clean(df)
    assert len(out) == 1
    assert report.nonpositive_quantity_dropped == 2


def test_drops_nonpositive_unit_price():
    df = _make_df(
        [
            {
                "InvoiceNo": "1",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 0.0,
                "CustomerID": 1.0,
                "Country": "UK",
            },
            {
                "InvoiceNo": "2",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": -1.0,
                "CustomerID": 1.0,
                "Country": "UK",
            },
            {
                "InvoiceNo": "3",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 2.5,
                "CustomerID": 1.0,
                "Country": "UK",
            },
        ]
    )
    out, report = clean(df)
    assert len(out) == 1
    assert report.nonpositive_price_dropped == 2


def test_parses_invoice_date_to_datetime():
    df = _make_df(
        [
            {
                "InvoiceNo": "1",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01 08:26:00",
                "UnitPrice": 2.0,
                "CustomerID": 1.0,
                "Country": "UK",
            }
        ]
    )
    out, _ = clean(df)
    assert pd.api.types.is_datetime64_any_dtype(out["InvoiceDate"])


def test_computes_total_price():
    df = _make_df(
        [
            {
                "InvoiceNo": "1",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 4,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 2.5,
                "CustomerID": 1.0,
                "Country": "UK",
            }
        ]
    )
    out, _ = clean(df)
    assert out.iloc[0]["TotalPrice"] == pytest.approx(10.0)


def test_customer_id_is_int_after_cleaning():
    df = _make_df(
        [
            {
                "InvoiceNo": "1",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 1.0,
                "CustomerID": 17850.0,
                "Country": "UK",
            }
        ]
    )
    out, _ = clean(df)
    assert out["CustomerID"].dtype.kind == "i"


def test_ge_validation_passes_on_clean_data():
    df = _make_df(
        [
            {
                "InvoiceNo": "1",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 2.0,
                "CustomerID": 1.0,
                "Country": "UK",
            },
            {
                "InvoiceNo": "2",
                "StockCode": "B",
                "Description": "Y",
                "Quantity": 5,
                "InvoiceDate": "2010-12-02",
                "UnitPrice": 3.0,
                "CustomerID": 2.0,
                "Country": "UK",
            },
        ]
    )
    out, _ = clean(df)
    assert validate_with_great_expectations(out) is True


def test_clean_report_dataclass_shape():
    df = _make_df(
        [
            {
                "InvoiceNo": "1",
                "StockCode": "A",
                "Description": "X",
                "Quantity": 1,
                "InvoiceDate": "2010-12-01",
                "UnitPrice": 1.0,
                "CustomerID": 1.0,
                "Country": "UK",
            }
        ]
    )
    _, report = clean(df)
    d = report.to_dict()
    assert set(d.keys()) == {
        "rows_in",
        "rows_out",
        "cancelled_dropped",
        "null_customer_dropped",
        "nonpositive_quantity_dropped",
        "nonpositive_price_dropped",
    }
    assert d["rows_in"] == 1
    assert d["rows_out"] == 1
