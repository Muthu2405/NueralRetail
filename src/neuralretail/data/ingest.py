"""Data ingest: load Online Retail II from XLSX/CSV with a synthetic fallback.

Behavior
--------
1. If a real Online Retail II file (XLSX or CSV) is present in
   ``NEURALRETAIL_RAW_DIR`` (config), load it.
2. If no real file is present, generate a synthetic sample of the same
   shape (and write it to raw/ so subsequent runs are reproducible) and
   return that.
3. Validate that the spec-mandated columns exist; raise a clear error
   listing any that are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from neuralretail.config import get_settings

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Final[list[str]] = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# Filenames we'll look for in raw_dir, in priority order.
RAW_FILE_CANDIDATES: Final[tuple[str, ...]] = (
    "online_retail_II.xlsx",
    "online_retail_ii.xlsx",
    "online_retail.xlsx",
    "online_retail.csv",
    "online_retail_II.csv",
    "online_retail_ii.csv",
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def generate_synthetic_online_retail(
    n_rows: int = 10_000,
    n_customers: int = 400,
    n_products: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a small synthetic Online Retail II-shaped sample.

    The sample deliberately contains the same edge cases the cleaner must
    handle: a handful of cancellations (InvoiceNo starting with "C"),
    some null CustomerID rows, and a few negative/zero quantities and
    unit prices. The cleaner should drop these and report the drops.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2010-12-01")
    end = pd.Timestamp("2011-12-09")

    # 5% of rows are intentional bad data
    n_bad_cancellations = int(n_rows * 0.02)
    n_bad_null_customer = int(n_rows * 0.02)
    n_bad_neg_qty = int(n_rows * 0.005)
    n_bad_neg_price = int(n_rows * 0.005)
    n_good = n_rows - (
        n_bad_cancellations + n_bad_null_customer + n_bad_neg_qty + n_bad_neg_price
    )

    # --- good rows ---
    seconds = rng.integers(0, int((end - start).total_seconds()), size=n_good)
    invoice_dates = start + pd.to_timedelta(seconds, unit="s")
    invoice_nos = np.array(
        [f"{rng.integers(500_000, 600_000)}" for _ in range(n_good)]
    )
    stock_codes = np.array(
        [f"{rng.integers(10_000, 99_999)}" for _ in range(n_good)]
    )
    descriptions = np.array(
        [
            rng.choice(
                [
                    "WHITE HANGING HEART T-LIGHT HOLDER",
                    "REGENCY CAKESTAND 3 TIER",
                    "JUMBO BAG RED RETROSPOT",
                    "ASSORTED COLOUR BIRD ORNAMENT",
                    "PARTY BUNTING",
                    "LUNCH BAG RED RETROSPOT",
                    "SET OF 3 CAKE TINS PANTRY DESIGN",
                    "POSTAGE",
                    "Manual",
                ]
            )
            for _ in range(n_good)
        ]
    )
    quantities = rng.integers(1, 30, size=n_good)
    unit_prices = np.round(rng.uniform(0.5, 25.0, size=n_good), 2)
    customer_ids = rng.integers(12000, 18000, size=n_good).astype(float)
    countries = rng.choice(
        [
            "United Kingdom",
            "Germany",
            "France",
            "EIRE",
            "Spain",
            "Netherlands",
            "Belgium",
            "Switzerland",
            "Portugal",
            "Australia",
        ],
        size=n_good,
        p=[0.85, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005],
    )

    good = pd.DataFrame(
        {
            "InvoiceNo": invoice_nos,
            "StockCode": stock_codes,
            "Description": descriptions,
            "Quantity": quantities,
            "InvoiceDate": invoice_dates,
            "UnitPrice": unit_prices,
            "CustomerID": customer_ids,
            "Country": countries,
        }
    )

    # --- bad rows: cancellations ---
    bad_cancel = good.iloc[:n_bad_cancellations].copy()
    bad_cancel["InvoiceNo"] = "C" + bad_cancel["InvoiceNo"].astype(str)
    bad_cancel["Quantity"] = -bad_cancel["Quantity"]

    # --- bad rows: null customer ---
    bad_null = good.iloc[
        n_bad_cancellations : n_bad_cancellations + n_bad_null_customer
    ].copy()
    bad_null["CustomerID"] = np.nan

    # --- bad rows: negative quantity ---
    bad_neg_qty = good.iloc[
        n_bad_cancellations + n_bad_null_customer : n_bad_cancellations
        + n_bad_null_customer
        + n_bad_neg_qty
    ].copy()
    bad_neg_qty["Quantity"] = -bad_neg_qty["Quantity"]

    # --- bad rows: zero/negative unit price ---
    start = n_bad_cancellations + n_bad_null_customer + n_bad_neg_qty
    bad_neg_price = good.iloc[start : start + n_bad_neg_price].copy()
    bad_neg_price["UnitPrice"] = 0.0

    df = pd.concat([good, bad_cancel, bad_null, bad_neg_qty, bad_neg_price], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Format detection + loading
# ---------------------------------------------------------------------------


def detect_format(path: Path) -> str:
    """Return 'xlsx' or 'csv' for a given file path."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return "xlsx"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unrecognized file extension for {path!r}: {suffix!r}")


def _read_file(path: Path) -> pd.DataFrame:
    fmt = detect_format(path)
    logger.info("Loading %s file: %s", fmt, path)
    if fmt == "xlsx":
        # The real Online Retail II XLSX has two sheets; concatenate them.
        sheets = pd.read_excel(path, sheet_name=None)
        if len(sheets) == 1:
            return next(iter(sheets.values()))
        return pd.concat(sheets.values(), ignore_index=True)
    return pd.read_csv(path, encoding="latin-1")


def find_raw_file(raw_dir: Path) -> Path | None:
    """Find the first existing raw file from the candidate list."""
    if not raw_dir.exists():
        return None
    for name in RAW_FILE_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    # Fall back: any .xlsx or .csv in the raw_dir
    for pattern in ("*.xlsx", "*.xls", "*.csv"):
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_raw(raw_dir: Path | None = None, *, force_synthetic: bool = False) -> pd.DataFrame:
    """Load raw transaction data, falling back to synthetic if needed.

    Parameters
    ----------
    raw_dir
        Directory to search for the raw file. Defaults to settings.raw_dir.
    force_synthetic
        If True, generate and return synthetic data even if a real file exists.
    """
    settings = get_settings()
    raw_dir = raw_dir or settings.raw_dir

    if not force_synthetic:
        existing = find_raw_file(raw_dir)
        if existing is not None:
            df = _read_file(existing)
            df = _validate_columns(df)
            logger.info(
                "Loaded %d rows from real file %s", len(df), existing.name
            )
            return df

    # Fall back to synthetic
    logger.warning(
        "No real Online Retail II file found in %s — generating synthetic sample.",
        raw_dir,
    )
    df = generate_synthetic_online_retail()
    out_path = raw_dir / "online_retail_synthetic.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote synthetic sample to %s (%d rows)", out_path, len(df))
    return _validate_columns(df)


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw data is missing required columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )
    return df
