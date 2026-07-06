"""Command-line entry point for neuralretail.

This is intentionally thin — each subcommand just imports and calls the
appropriate module function. Keeping dispatch in one place means tests
can call module functions directly without spawning a subprocess.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from neuralretail.config import get_settings


def _setup_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


def cmd_data(_args: argparse.Namespace) -> int:
    """Phase 1: ingest raw -> clean + GE validate -> processed parquet."""
    from neuralretail.data.ingest import load_raw
    from neuralretail.data.clean import clean_and_save

    settings = get_settings()
    raw = load_raw()
    cleaned, report = clean_and_save(raw)
    print(
        f"Cleaned: {report.rows_in} -> {report.rows_out} rows "
        f"(cancelled={report.cancelled_dropped}, "
        f"null_customer={report.null_customer_dropped}, "
        f"bad_qty={report.nonpositive_quantity_dropped}, "
        f"bad_price={report.nonpositive_price_dropped})"
    )
    print(f"Processed parquet: {settings.processed_dir / 'cleaned.parquet'}")
    return 0


def cmd_features(_args: argparse.Namespace) -> int:
    """Phase 2: build RFM + daily-revenue / lag-rolling features."""
    from neuralretail.config import get_settings
    from neuralretail.features.rfm import compute_rfm, save_rfm
    from neuralretail.features.timeseries import (
        build_daily_revenue,
        build_timeseries_features,
        save_timeseries,
    )

    settings = get_settings()
    cleaned_path = settings.processed_dir / "cleaned.parquet"
    if not cleaned_path.exists():
        print(
            f"ERROR: {cleaned_path} not found. Run `python -m neuralretail.cli data` first.",
            file=sys.stderr,
        )
        return 1

    transactions = pd.read_parquet(cleaned_path)

    # RFM
    rfm = compute_rfm(transactions)
    rfm_path = save_rfm(rfm)
    print(
        f"RFM: {len(rfm):,} customers, "
        f"Recency mean={rfm['Recency'].mean():.1f}d, "
        f"Frequency mean={rfm['Frequency'].mean():.2f}, "
        f"Monetary mean=${rfm['Monetary'].mean():.2f}"
    )
    print(f"  -> {rfm_path}")

    # Daily revenue
    daily = build_daily_revenue(transactions)
    daily_path = save_timeseries(daily, "daily_revenue")
    print(
        f"Daily revenue: {len(daily):,} days, "
        f"{daily.index.min().date()} to {daily.index.max().date()}, "
        f"total ${daily['Revenue'].sum():,.2f}"
    )
    print(f"  -> {daily_path}")

    # Lag/rolling + calendar features
    ts = build_timeseries_features(transactions)
    ts_path = save_timeseries(ts, "timeseries_features")
    print(f"Timeseries features: {ts.shape[0]:,} days x {ts.shape[1]} cols")
    print(f"  -> {ts_path}")
    return 0


def cmd_train(_args: argparse.Namespace) -> int:
    """Phase 3: train forecasting, churn, segmentation, inventory; log to MLflow."""
    from neuralretail.models import forecasting, churn, segmentation, inventory
    from neuralretail.models._mlflow_utils import setup_mlflow

    settings = get_settings()
    cleaned_path = settings.processed_dir / "cleaned.parquet"
    rfm_path = settings.processed_dir / "rfm.parquet"
    daily_path = settings.processed_dir / "daily_revenue.parquet"

    for label, p in [
        ("cleaned transactions", cleaned_path),
        ("RFM features", rfm_path),
        ("daily revenue", daily_path),
    ]:
        if not p.exists():
            print(
                f"ERROR: {label} parquet not found at {p}. "
                "Run `python -m neuralretail.cli data` and `... features` first.",
                file=sys.stderr,
            )
            return 1

    setup_mlflow()

    transactions = pd.read_parquet(cleaned_path)
    rfm = pd.read_parquet(rfm_path)
    daily = pd.read_parquet(daily_path)

    # --- Forecasting ---
    print("Training demand forecaster (Prophet)...")
    fc = forecasting.train(daily, horizon_days=30, holdout_days=30)
    print(f"  -> MAPE={fc.metrics['mape']:.4f}, RMSE={fc.metrics['rmse']:.2f}")
    forecasting.save(fc.model)

    # --- Churn ---
    print("Training churn classifier (XGBoost)...")
    ch = churn.train(transactions, rfm)
    print(
        f"  -> AUC-ROC={ch.metrics['auc_roc']:.4f}, "
        f"precision={ch.metrics['precision']:.4f}, "
        f"recall={ch.metrics['recall']:.4f}, F1={ch.metrics['f1']:.4f}"
    )
    churn.save(ch.model)

    # --- Segmentation ---
    print("Training customer segmentation (KMeans)...")
    sg = segmentation.train(rfm)
    print(f"  -> k={sg.k}, silhouette={sg.metrics['silhouette']:.4f}")
    print(f"     personas: {dict(sg.persona_map)}")
    segmentation.save(sg.pipeline)

    # --- Inventory ---
    print("Building inventory table (ABC + EOQ + dead-stock)...")
    inv = inventory.train(transactions)
    print(
        f"  -> {int(inv.metrics['n_skus'])} SKUs, "
        f"A={int(inv.metrics['n_class_a'])} "
        f"B={int(inv.metrics['n_class_b'])} "
        f"C={int(inv.metrics['n_class_c'])}, "
        f"dead-stock={int(inv.metrics['n_dead_stock'])}"
    )
    inventory.save(inv.table)
    return 0


def cmd_promote(_args: argparse.Namespace) -> int:
    """Phase 4: promote the best run of each registered model to 'Production' alias."""
    from neuralretail.models._mlflow_utils import promote_best

    promoted = promote_best()
    if not promoted:
        print("No models promoted (no runs found).")
        return 0
    for name, version in promoted.items():
        print(f"  {name} -> Production (v{version})")
    return 0


def cmd_monitor(_args: argparse.Namespace) -> int:
    """Phase 7: generate Evidently data-drift HTML report (reference vs current)."""
    from neuralretail.monitoring.drift import save_drift_report

    settings = get_settings()
    cleaned_path = settings.processed_dir / "cleaned.parquet"
    if not cleaned_path.exists():
        print(
            f"ERROR: {cleaned_path} not found. Run `python -m neuralretail.cli data` first.",
            file=sys.stderr,
        )
        return 1

    transactions = pd.read_parquet(cleaned_path)
    report = save_drift_report(
        transactions,
        output_path=settings.report_dir / "drift_report.html",
        reference_fraction=settings.drift_reference_fraction,
    )
    print(
        f"Drift: {report.n_drifted_columns}/{report.n_columns} columns drifted "
        f"(share={report.drift_share:.2f}); "
        f"ref={report.n_reference:,} rows [{report.reference_start.date()} -> "
        f"{report.reference_end.date()}], current={report.n_current:,} rows "
        f"[{report.current_start.date()} -> {report.current_end.date()}]"
    )
    if report.drifted_columns:
        print(f"  drifted: {', '.join(report.drifted_columns)}")
    print(f"  -> {report.report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralretail",
        description="NeuralRetail — AI-powered retail sales intelligence.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Ingest + clean raw data")
    p_data.set_defaults(func=cmd_data)

    p_features = sub.add_parser("features", help="Build feature parquets")
    p_features.set_defaults(func=cmd_features)

    p_train = sub.add_parser("train", help="Train all models")
    p_train.set_defaults(func=cmd_train)

    p_promote = sub.add_parser("promote", help="Promote best run per model to Production")
    p_promote.set_defaults(func=cmd_promote)

    p_monitor = sub.add_parser(
        "monitor",
        help="Phase 7: generate Evidently data-drift HTML report (reference vs current)",
    )
    p_monitor.set_defaults(func=cmd_monitor)

    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
