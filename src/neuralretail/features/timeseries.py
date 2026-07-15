"""Daily time-series feature engineering for the demand forecaster.

Given a cleaned transaction frame, builds:
1. A daily-aggregated frame (Revenue, Orders, ItemsSold, Customers).
2. A feature-augmented frame with lag features, rolling means/std over
   7/14/30 day windows, and calendar flags (day-of-week, is_weekend,
   month, ISO week).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import pandas as pd

from neuralretail.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_LAGS: Final[tuple[int, ...]] = (1, 7, 14)
DEFAULT_WINDOWS: Final[tuple[int, ...]] = (7, 14, 30)


def build_daily_revenue(
    transactions: pd.DataFrame,
    *,
    fill_missing_days: bool = True,
) -> pd.DataFrame:
    """Aggregate transactions to one row per calendar day.

    Columns:
        - InvoiceDate (date, index)
        - Revenue   (sum of TotalPrice)
        - Orders    (nunique InvoiceNo)
        - ItemsSold (sum of Quantity)
        - Customers (nunique CustomerID)
    """
    required = {"InvoiceDate", "InvoiceNo", "TotalPrice", "Quantity", "CustomerID"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions is missing required columns: {missing}")

    df = transactions.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["InvoiceDate"] = df["InvoiceDate"].dt.normalize()  # truncate to date

    grouped = df.groupby("InvoiceDate", sort=True)
    daily = pd.DataFrame(
        {
            "Revenue": grouped["TotalPrice"].sum(),
            "Orders": grouped["InvoiceNo"].nunique(),
            "ItemsSold": grouped["Quantity"].sum(),
            "Customers": grouped["CustomerID"].nunique(),
        }
    )

    if fill_missing_days and len(daily) > 0:
        full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_idx, fill_value=0)
        daily.index.name = "InvoiceDate"

    logger.info(
        "Daily revenue: %d days from %s to %s, total revenue=%.2f",
        len(daily),
        daily.index.min().date() if len(daily) else "n/a",
        daily.index.max().date() if len(daily) else "n/a",
        daily["Revenue"].sum(),
    )
    return daily


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, weekend flag, month, ISO week."""
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise ValueError("daily must have a DatetimeIndex")
    out = daily.copy()
    out["day_of_week"] = out.index.dayofweek.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["day_of_month"] = out.index.day.astype("int8")
    out["month"] = out.index.month.astype("int8")
    out["iso_week"] = out.index.isocalendar().week.astype("int16")
    out["year"] = out.index.year.astype("int16")
    return out


def add_lag_rolling_features(
    daily: pd.DataFrame,
    *,
    target: str = "Revenue",
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Add lag features and rolling-window stats for the target column."""
    if target not in daily.columns:
        raise ValueError(f"target column {target!r} not in daily frame")
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise ValueError("daily must have a DatetimeIndex")

    out = daily.copy()
    for lag in lags:
        if lag <= 0:
            raise ValueError("lags must be positive integers")
        out[f"lag_{lag}"] = out[target].shift(lag)
    for w in windows:
        if w <= 0:
            raise ValueError("windows must be positive integers")
        rolling = out[target].shift(1).rolling(window=w, min_periods=max(1, w // 2))
        out[f"rolling_mean_{w}"] = rolling.mean()
        out[f"rolling_std_{w}"] = rolling.std()
    return out


def build_timeseries_features(
    transactions: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    target: str = "Revenue",
) -> pd.DataFrame:
    """End-to-end: daily revenue -> calendar features -> lag/rolling features."""
    daily = build_daily_revenue(transactions)
    daily = add_calendar_features(daily)
    daily = add_lag_rolling_features(daily, target=target, lags=lags, windows=windows)
    logger.info("Timeseries features: %d rows, %d columns", *daily.shape)
    return daily


def save_timeseries(
    frame: pd.DataFrame,
    name: str,
    output_path: str | Path | None = None,  # noqa: F821
) -> Path:  # noqa: F821
    """Save a timeseries frame to parquet. Returns the path written."""
    from pathlib import Path

    settings = get_settings()
    path = Path(output_path) if output_path is not None else settings.processed_dir / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    logger.info("Wrote %s to %s (%d rows)", name, path, len(frame))
    return path
