"""Data-drift monitoring with Evidently AI.

Generates an HTML report comparing a "reference" slice of the cleaned
data (oldest 70 % by default) to a "current" slice (newest 30 %).
This is a portfolio-grade drift check, not a live alerting pipeline —
the artefact is the report, saved to ``report/drift_report.html``.

Per the spec, we report *data* drift only. Per-column drift summary
is logged back as a dataclass for the CLI / docs to consume.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from neuralretail.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Summary of a data-drift comparison between reference and current slices."""

    n_reference: int
    n_current: int
    reference_start: pd.Timestamp
    reference_end: pd.Timestamp
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    n_columns: int
    n_drifted_columns: int
    drift_share: float
    drifted_columns: list[str]
    report_path: Path

    def to_dict(self) -> dict:
        d = asdict(self)
        # Timestamps aren't JSON-native; stringify.
        d["reference_start"] = str(self.reference_start)
        d["reference_end"] = str(self.reference_end)
        d["current_start"] = str(self.current_start)
        d["current_end"] = str(self.current_end)
        d["report_path"] = str(self.report_path)
        return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Columns fed into the drift detector. Numeric and categorical gives
# Evidently enough variety to be informative; we deliberately exclude
# high-cardinality IDs (InvoiceNo, StockCode, Description, CustomerID).
_DRIFT_COLUMNS: list[str] = ["Quantity", "UnitPrice", "TotalPrice", "Country"]


def make_reference_current_split(
    df: pd.DataFrame, reference_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` chronologically into (reference, current) windows.

    The split is on the ``InvoiceDate`` column; rows with the smallest
    timestamps form the reference window, the rest form current. The
    two slices are disjoint and ordered (reference_end <= current_start).
    """
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError(
            f"reference_fraction must be in (0, 1), got {reference_fraction!r}"
        )
    if "InvoiceDate" not in df.columns:
        raise KeyError("df must have an 'InvoiceDate' column to split chronologically")

    ordered = df.sort_values("InvoiceDate", kind="mergesort").reset_index(drop=True)
    cutoff = int(len(ordered) * reference_fraction)
    if cutoff == 0 or cutoff == len(ordered):
        raise ValueError(
            f"reference_fraction={reference_fraction} gives an empty slice "
            f"(n={len(ordered)}); use a fraction strictly between 0 and 1."
        )
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()


def _prepare_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Project to the drift columns, plus derived calendar features."""
    missing = [c for c in _DRIFT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"df is missing expected columns: {missing}")

    out = df[_DRIFT_COLUMNS].copy()
    if "InvoiceDate" in df.columns:
        ts = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        out["Hour"] = ts.dt.hour
        out["DayOfWeek"] = ts.dt.dayofweek
    return out


def build_drift_summary(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[dict, dict]:
    """Run an Evidently ``DataDriftPreset`` and return (per_column, summary).

    ``per_column`` maps column name -> ``{"score": float, "method": str,
    "threshold": float, "drift_detected": bool}``. The ``score`` is whatever
    distance the test produced (e.g. Wasserstein / Jensen-Shannon / p-value);
    ``drift_detected`` follows Evidently's own rule of
    ``score > threshold``.

    ``summary`` is the aggregate ``{n_columns, n_drifted_columns,
    drift_share, drifted_columns}`` taken from the
    ``DriftedColumnsCount`` metric emitted by the preset.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ref = _prepare_for_drift(reference)
    cur = _prepare_for_drift(current)

    snapshot = Report([DataDriftPreset()]).run(
        reference_data=ref, current_data=cur
    )
    raw = snapshot.dict()

    # Default to the per-column rule, but the aggregate metric is the
    # canonical "how many columns drifted" number from Evidently.
    import re

    per_column: dict[str, dict] = {}
    drifted_columns: list[str] = []
    n_cols = 0
    # ValueDrift names look like:
    #   "ValueDrift(column=Quantity,method=Wasserstein distance (normed),threshold=0.1)"
    # The first comma lands inside the ValueDrift(prefix, so we use a regex
    # to pull out column=, method=, threshold= reliably.
    value_drift_re = re.compile(
        r"ValueDrift\(column=(?P<column>[^,]+),method=(?P<method>[^,]+),threshold=(?P<threshold>[\d.]+)\)"
    )
    for m in raw.get("metrics", []):
        name = m.get("metric_name", "") or ""
        match = value_drift_re.match(name)
        if not match:
            continue
        col = match.group("column")
        method = match.group("method")
        try:
            threshold = float(match.group("threshold"))
        except ValueError:
            threshold = float("nan")
        n_cols += 1
        try:
            score = float(m.get("value"))
        except (TypeError, ValueError):
            score = float("nan")
        # For Wasserstein / Jensen-Shannon: drift when score > threshold.
        # For p-value tests: drift when score < threshold.
        is_pvalue = "p_value" in method
        drifted = (score < threshold) if is_pvalue else (score > threshold)
        per_column[col] = {
            "score": score,
            "method": method,
            "threshold": threshold,
            "drift_detected": drifted,
        }
        if drifted:
            drifted_columns.append(col)

    # Override with the aggregate DriftedColumnsCount when present (the
    # canonical aggregate — `drift_share` from the preset takes
    # precedence over our hand-counted count).
    drift_share: float | None = None
    n_drifted_aggregate: int | None = None
    for m in raw.get("metrics", []):
        name = m.get("metric_name", "") or ""
        if not name.startswith("DriftedColumnsCount"):
            continue
        val = m.get("value") or {}
        if isinstance(val, dict):
            try:
                n_drifted_aggregate = int(val.get("count", 0))
                drift_share = float(val.get("share", 0.0))
            except (TypeError, ValueError):
                pass
        break

    summary: dict = {
        "n_columns": n_cols,
        "n_drifted_columns": n_drifted_aggregate
        if n_drifted_aggregate is not None
        else len(drifted_columns),
        "drift_share": drift_share
        if drift_share is not None
        else ((len(drifted_columns) / n_cols) if n_cols else 0.0),
        "drifted_columns": drifted_columns,
    }
    return per_column, summary


def save_drift_report(
    df: pd.DataFrame,
    *,
    output_path: Path | None = None,
    reference_fraction: float = 0.7,
) -> DriftReport:
    """Build a chronological reference/current split, run Evidently, write HTML.

    Returns a :class:`DriftReport` dataclass and also writes a JSON
    summary next to the HTML for machine consumption.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    settings = get_settings()
    output_path = output_path or settings.report_dir / "drift_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference, current = make_reference_current_split(
        df, reference_fraction=reference_fraction
    )

    ref_prepared = _prepare_for_drift(reference)
    cur_prepared = _prepare_for_drift(current)

    logger.info(
        "Drift split: reference=%d rows (%s → %s), current=%d rows (%s → %s)",
        len(reference),
        reference["InvoiceDate"].min(),
        reference["InvoiceDate"].max(),
        len(current),
        current["InvoiceDate"].min(),
        current["InvoiceDate"].max(),
    )

    snapshot = Report([DataDriftPreset()]).run(
        reference_data=ref_prepared, current_data=cur_prepared
    )
    snapshot.save_html(str(output_path))

    per_column, summary = build_drift_summary(reference, current)

    # Sidecar JSON for machine readers (and the CLI summary print).
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "reference": {
                    "n_rows": len(reference),
                    "start": str(reference["InvoiceDate"].min()),
                    "end": str(reference["InvoiceDate"].max()),
                },
                "current": {
                    "n_rows": len(current),
                    "start": str(current["InvoiceDate"].min()),
                    "end": str(current["InvoiceDate"].max()),
                },
                "summary": summary,
                "per_column": per_column,
            },
            indent=2,
            default=str,
        )
    )

    report = DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        reference_start=reference["InvoiceDate"].min(),
        reference_end=reference["InvoiceDate"].max(),
        current_start=current["InvoiceDate"].min(),
        current_end=current["InvoiceDate"].max(),
        n_columns=summary["n_columns"],
        n_drifted_columns=summary["n_drifted_columns"],
        drift_share=summary["drift_share"],
        drifted_columns=summary["drifted_columns"],
        report_path=output_path,
    )
    logger.info(
        "Drift report: %d/%d columns drifted (share=%.2f) → %s",
        report.n_drifted_columns,
        report.n_columns,
        report.drift_share,
        report.report_path,
    )
    return report
