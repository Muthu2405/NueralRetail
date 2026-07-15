"""RFM (Recency, Frequency, Monetary) feature engineering.

Inputs: a cleaned transaction DataFrame with the spec columns
(`CustomerID`, `InvoiceNo`, `InvoiceDate`, `TotalPrice`).
Output: one row per customer with three numeric features.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from neuralretail.config import get_settings

logger = logging.getLogger(__name__)

# Output column order — fixed, so downstream code can rely on it.
RFM_COLUMNS: Final[tuple[str, ...]] = (
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
    "FirstPurchase",
    "LastPurchase",
)


def compute_rfm(
    transactions: pd.DataFrame,
    *,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute RFM features per customer.

    Parameters
    ----------
    transactions
        Cleaned transaction frame. Must contain `CustomerID`, `InvoiceNo`,
        `InvoiceDate`, and `TotalPrice`.
    snapshot_date
        The "as-of" date for Recency. Defaults to one day after the latest
        `InvoiceDate` in the data — this is the standard RFM convention when
        computing against a fixed point in time.

    Returns
    -------
    pd.DataFrame
        One row per `CustomerID` with columns
        (`CustomerID`, `Recency`, `Frequency`, `Monetary`,
         `FirstPurchase`, `LastPurchase`).
    """
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions is missing required columns: {missing}")

    df = transactions.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    snapshot_date = pd.Timestamp(snapshot_date)

    grouped = df.groupby("CustomerID", sort=True)

    recency = (snapshot_date - grouped["InvoiceDate"].max()).dt.days.astype(int)
    first_purchase = grouped["InvoiceDate"].min()
    last_purchase = grouped["InvoiceDate"].max()
    frequency = grouped["InvoiceNo"].nunique().astype(int)
    monetary = grouped["TotalPrice"].sum().clip(lower=0.0)

    out = pd.DataFrame(
        {
            "CustomerID": recency.index.astype(int),
            "Recency": recency.values,
            "Frequency": frequency.reindex(recency.index).values,
            "Monetary": monetary.reindex(recency.index).values,
            "FirstPurchase": first_purchase.reindex(recency.index).values,
            "LastPurchase": last_purchase.reindex(recency.index).values,
        }
    )
    out = out.sort_values("Monetary", ascending=False).reset_index(drop=True)
    out = out[list(RFM_COLUMNS)]
    logger.info(
        "RFM: %d customers, Recency mean=%.1f, Frequency mean=%.2f, Monetary mean=%.2f",
        len(out),
        out["Recency"].mean(),
        out["Frequency"].mean(),
        out["Monetary"].mean(),
    )
    return out


def save_rfm(
    rfm: pd.DataFrame,
    output_path: str | Path | None = None,  # noqa: F821
) -> Path:  # noqa: F821
    """Save RFM frame to parquet. Returns the path written."""
    from pathlib import Path

    settings = get_settings()
    path = Path(output_path) if output_path is not None else settings.processed_dir / "rfm.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(path, index=False)
    logger.info("Wrote RFM to %s (%d rows)", path, len(rfm))
    return path
