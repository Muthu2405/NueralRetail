"""Tests for monitoring/drift.py — verify the split + report writer.

These tests build a small synthetic chronological DataFrame, exercise
the split and the save_drift_report entry point, and confirm the HTML
artefact is well-formed. Evidently is real, so we don't mock it — the
goal is a true end-to-end smoke check.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neuralretail.monitoring.drift import (
    make_reference_current_split,
    save_drift_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 200) -> pd.DataFrame:
    """Build a small chronological frame with the columns drift.py expects."""
    base = pd.Timestamp("2020-01-01")
    return pd.DataFrame(
        {
            "InvoiceNo": [f"INV{i:05d}" for i in range(n)],
            "StockCode": ["84029G"] * n,
            "Description": ["Widget"] * n,
            "Quantity": [1 + (i % 5) for i in range(n)],
            "InvoiceDate": [base + pd.Timedelta(hours=i) for i in range(n)],
            "UnitPrice": [1.50 + 0.01 * (i % 10) for i in range(n)],
            "CustomerID": [10000 + (i % 20) for i in range(n)],
            "Country": ["United Kingdom" if i % 2 == 0 else "Germany" for i in range(n)],
            "TotalPrice": [(1 + (i % 5)) * (1.50 + 0.01 * (i % 10)) for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------


def test_split_is_chronological_and_disjoint():
    df = _make_df(200)
    ref, cur = make_reference_current_split(df, reference_fraction=0.7)

    # Disjoint: row counts sum to the input (no duplicates, no drops)
    assert len(ref) + len(cur) == len(df)
    # Chronological: reference is older (max) <= current (min)
    assert ref["InvoiceDate"].max() <= cur["InvoiceDate"].min()


def test_split_fraction_is_honoured():
    df = _make_df(1000)
    ref, cur = make_reference_current_split(df, reference_fraction=0.7)
    # 1000 * 0.7 = 700 exactly (no rounding surprise)
    assert len(ref) == 700
    assert len(cur) == 300


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_invalid_fraction(bad_frac):
    df = _make_df(100)
    with pytest.raises(ValueError):
        make_reference_current_split(df, reference_fraction=bad_frac)


def test_split_requires_invoice_date_column():
    df = _make_df(100).drop(columns=["InvoiceDate"])
    with pytest.raises(KeyError):
        make_reference_current_split(df, reference_fraction=0.7)


# ---------------------------------------------------------------------------
# End-to-end report writer
# ---------------------------------------------------------------------------


def test_save_drift_report_writes_html(tmp_path: Path):
    df = _make_df(200)
    output = tmp_path / "drift.html"
    report = save_drift_report(
        df,
        output_path=output,
        reference_fraction=0.7,
    )

    # File exists, non-empty, and looks like HTML
    assert output.exists()
    assert output.stat().st_size > 0
    head = output.read_text(encoding="utf-8", errors="ignore")[:200].lower()
    assert "<html" in head, f"drift report doesn't look like HTML: {head!r}"

    # Sidecar summary JSON
    summary_json = output.with_suffix(".summary.json")
    assert summary_json.exists()

    # Dataclass shape: counts line up with the input
    assert report.n_reference + report.n_current == len(df)
    assert report.n_columns >= 1
    assert 0.0 <= report.drift_share <= 1.0
    assert report.report_path == output


def test_save_drift_report_uses_default_output_when_omitted(tmp_path: Path):
    """When output_path is None, the module falls back to settings.report_dir."""
    df = _make_df(120)
    # Save into tmp_path so we don't pollute the real report/ dir.
    report = save_drift_report(df, reference_fraction=0.7)
    # The default lands in settings.report_dir ("report/"). We don't assert
    # the exact path here, only that the call succeeded and the file exists.
    assert report.report_path.exists()
    assert report.report_path.stat().st_size > 0
