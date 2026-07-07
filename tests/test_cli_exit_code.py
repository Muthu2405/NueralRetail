"""Test for the CLI's exit-code contract when GE validation fails.

The build prompt requires the data pipeline to fail loudly with a
non-zero exit code when Great Expectations validation fails. This
test exercises the full ``cmd_data`` path with a frame that is
clean (i.e. it would normally pass) but where we monkey-patch
``validate_with_great_expectations`` to return ``False`` — exactly
the GE-failure scenario.
"""

from __future__ import annotations

import pandas as pd

from neuralretail import cli
from neuralretail.data import clean as clean_mod
from neuralretail.data import ingest as ingest_mod


def _good_row() -> dict:
    return {
        "InvoiceNo": "536365",
        "StockCode": "84029G",
        "Description": "X",
        "Quantity": 6,
        "InvoiceDate": pd.Timestamp("2010-12-01 08:26:00"),
        "UnitPrice": 2.55,
        "CustomerID": 17850.0,
        "Country": "United Kingdom",
    }


def test_cmd_data_exits_nonzero_on_ge_failure(monkeypatch, capsys):
    """When ``validate_with_great_expectations`` returns False,
    ``cmd_and_save`` (the helper that wraps the cleaning + GE check)
    raises a RuntimeError; ``cmd_data`` must catch it and return a
    non-zero exit code with an error message on stderr. The
    make-pipeline target depends on this so a broken clean step is
    caught loudly.
    """
    # Make the GE step return False, then have clean_and_save re-raise
    # the way the real implementation does.
    monkeypatch.setattr(
        clean_mod, "validate_with_great_expectations", lambda _df: False
    )

    def _fake_clean_and_save(df, output_path=None, report_path=None):
        if not clean_mod.validate_with_great_expectations(df):
            raise RuntimeError(
                "Great Expectations validation failed on cleaned data — aborting."
            )
        return df.copy(), clean_mod.CleanReport(
            rows_in=len(df), rows_out=len(df),
            cancelled_dropped=0, null_customer_dropped=0,
            nonpositive_quantity_dropped=0, nonpositive_price_dropped=0,
        )

    monkeypatch.setattr(clean_mod, "clean_and_save", _fake_clean_and_save)
    # Bypass the file loader so we don't need a real raw CSV.
    monkeypatch.setattr(ingest_mod, "load_raw", lambda: pd.DataFrame([_good_row() for _ in range(20)]))

    # argparse.Namespace stand-in (cmd_data ignores its args)
    rc = cli.cmd_data(None)

    captured = capsys.readouterr()
    assert rc == 1, f"expected non-zero exit on GE failure, got {rc}"
    assert "ERROR" in captured.err, f"expected error message on stderr, got {captured.err!r}"
    assert "validation" in captured.err.lower()


def test_cmd_data_exits_zero_when_clean_succeeds(monkeypatch, capsys):
    """Sanity check: the same setup with a passing GE validator
    returns 0. Guards against accidentally returning 1 for a
    different reason (e.g. a missing import).
    """
    monkeypatch.setattr(
        clean_mod, "validate_with_great_expectations", lambda _df: True
    )
    monkeypatch.setattr(
        clean_mod, "clean_and_save", lambda df, output_path=None, report_path=None: (df.copy(), clean_mod.CleanReport(
            rows_in=len(df), rows_out=len(df),
            cancelled_dropped=0, null_customer_dropped=0,
            nonpositive_quantity_dropped=0, nonpositive_price_dropped=0,
        ))
    )
    monkeypatch.setattr(ingest_mod, "load_raw", lambda: pd.DataFrame([_good_row() for _ in range(20)]))

    rc = cli.cmd_data(None)
    captured = capsys.readouterr()
    assert rc == 0, f"expected 0 on success, got {rc}; stderr={captured.err!r}"
    assert "ERROR" not in captured.err
