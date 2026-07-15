"""Data cleaning + Great Expectations validation.

Cleaning rules (per spec):
- Drop cancelled invoices (InvoiceNo starting with "C" or "c")
- Drop rows with null CustomerID
- Filter Quantity > 0 and UnitPrice > 0
- Parse InvoiceDate to datetime
- Compute TotalPrice = Quantity * UnitPrice
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

# Silence GE's import-time chatter before we even import it.
for _noisy in (
    "great_expectations",
    "great_expectations._docs_decorators",
    "great_expectations.data_context.types.base",
    "great_expectations.checkpoint",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import great_expectations as gx  # noqa: E402
import great_expectations.expectations as gxe  # noqa: E402
import pandas as pd  # noqa: E402

from neuralretail.config import get_settings  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


@dataclass
class CleanReport:
    """Summary of the cleaning step — rows in, rows out, and per-rule drops."""

    rows_in: int
    rows_out: int
    cancelled_dropped: int
    null_customer_dropped: int
    nonpositive_quantity_dropped: int
    nonpositive_price_dropped: int

    def to_dict(self) -> dict:
        return asdict(self)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    """Apply the standard cleaning rules and return (cleaned_df, report)."""
    rows_in = len(df)
    df = df.copy()

    # 1. Drop cancelled invoices (InvoiceNo starting with C / c)
    inv = df["InvoiceNo"].astype(str)
    cancelled_mask = inv.str.upper().str.startswith("C")
    n_cancelled = int(cancelled_mask.sum())
    df = df.loc[~cancelled_mask]

    # 2. Drop null CustomerID
    null_cust_mask = df["CustomerID"].isna()
    n_null_cust = int(null_cust_mask.sum())
    df = df.loc[~null_cust_mask]

    # 3. Quantity > 0
    bad_qty_mask = df["Quantity"] <= 0
    n_bad_qty = int(bad_qty_mask.sum())
    df = df.loc[~bad_qty_mask]

    # 4. UnitPrice > 0
    bad_price_mask = df["UnitPrice"] <= 0
    n_bad_price = int(bad_price_mask.sum())
    df = df.loc[~bad_price_mask]

    # 5. Parse InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    bad_date_mask = df["InvoiceDate"].isna()
    n_bad_date = int(bad_date_mask.sum())  # noqa: F841
    df = df.loc[~bad_date_mask]

    # 6. TotalPrice
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # CustomerID should be int
    df["CustomerID"] = df["CustomerID"].astype(int)

    report = CleanReport(
        rows_in=rows_in,
        rows_out=len(df),
        cancelled_dropped=n_cancelled,
        null_customer_dropped=n_null_cust,
        nonpositive_quantity_dropped=n_bad_qty,
        nonpositive_price_dropped=n_bad_price,
    )
    logger.info("Clean report: %s", report.to_dict())
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Great Expectations validation (GE 1.x fluent API)
# ---------------------------------------------------------------------------


def _silence_ge_progress_bars() -> None:
    """GE 1.x prints a tqdm progress bar and `_docs_decorators` chatter per run;
    mute both for clean logs.

    GE does ``from tqdm.auto import tqdm`` at module load time and binds the
    class to a local name, so we patch ``tqdm.auto.tqdm`` and the symbol
    inside ``validation_graph`` directly to a silent subclass.
    """
    import os

    import tqdm

    _orig_tqdm = tqdm.tqdm

    class _SilentTqdm(_orig_tqdm):
        def __init__(self, *a, **kw):
            if "file" not in kw:
                kw["file"] = open(os.devnull, "w")
            super().__init__(*a, **kw)
            self.update = lambda *a, **kw: None  # type: ignore[assignment]
            self.refresh = lambda *a, **kw: None  # type: ignore[assignment]

    # Patch the canonical tqdm location and every submodule that may have
    # already imported it.
    tqdm.tqdm = _SilentTqdm
    for sub in ("tqdm.auto", "tqdm.std", "tqdm.notebook", "tqdm.gui"):
        try:
            mod = __import__(sub, fromlist=["tqdm"])
            if hasattr(mod, "tqdm"):
                mod.tqdm = _SilentTqdm  # type: ignore[attr-defined]
        except ImportError:
            pass

    # GE's validation_graph captures the symbol at import time. Walk through
    # every great_expectations module and rewrite the `tqdm` name.
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("great_expectations"):
            continue
        if mod is None or not hasattr(mod, "tqdm"):
            continue
        try:
            mod.tqdm = _SilentTqdm  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

    for noisy in (
        "great_expectations",
        "great_expectations.checkpoint",
        "great_expectations._docs_decorators",
        "great_expectations.data_context.types.base",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _build_suite(suite_name: str) -> gx.ExpectationSuite:
    """Build a fresh ephemeral suite covering the spec rules."""
    suite = gx.ExpectationSuite(name=suite_name)
    suite.add_expectation(
        gxe.ExpectTableRowCountToBeBetween(min_value=1, max_value=10_000_000)
    )
    for col in ("CustomerID", "InvoiceNo", "InvoiceDate", "StockCode", "Quantity", "UnitPrice"):
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="Quantity", min_value=1))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="UnitPrice", min_value=0.01)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="TotalPrice", min_value=0.01)
    )
    return suite


def validate_with_great_expectations(
    df: pd.DataFrame, *, suite_name: str = "neuralretail_cleaned_v1"
) -> bool:
    """Run the lightweight Great Expectations suite.

    Returns True on success, False on failure. Logs a summary.
    """
    try:
        _silence_ge_progress_bars()
        context = gx.get_context(mode="ephemeral")
        ds = context.data_sources.add_pandas(name="neuralretail_ds")
        asset = ds.add_dataframe_asset(name="cleaned_asset")
        bdef = asset.add_batch_definition_whole_dataframe(name="cleaned_bdef")

        suite = context.suites.add(_build_suite(suite_name))
        vdef = context.validation_definitions.add(
            gx.ValidationDefinition(name="cleaned_validation", data=bdef, suite=suite)
        )

        result = vdef.run(batch_parameters={"dataframe": df})
        success = bool(result.success)
        evaluated = len(result.results)
        successful = sum(1 for r in result.results if r.success)
        logger.info(
            "GE suite %s: success=%s evaluated=%d successful=%d",
            suite_name,
            success,
            evaluated,
            successful,
        )
        if not success:
            for r in result.results:
                if not r.success:
                    logger.warning(
                        "  failed: %s — %s",
                        r.expectation_config.type,
                        getattr(r, "result", {}),
                    )
        return success
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Great Expectations validation crashed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def clean_and_save(
    df: pd.DataFrame,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> tuple[pd.DataFrame, CleanReport]:
    """Clean ``df``, validate with GE, write parquet + JSON report."""
    settings = get_settings()
    output_path = output_path or settings.processed_dir / "cleaned.parquet"
    report_path = report_path or settings.processed_dir / "clean_report.json"

    cleaned, report = clean(df)

    if not validate_with_great_expectations(cleaned):
        raise RuntimeError(
            "Great Expectations validation failed on cleaned data — "
            "refusing to write the processed parquet. See logs for details."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info("Wrote cleaned parquet to %s (%d rows)", output_path, len(cleaned))
    logger.info("Wrote clean report to %s", report_path)
    return cleaned, report
