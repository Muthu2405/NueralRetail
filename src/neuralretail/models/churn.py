"""Churn classifier.

A customer is "churned" if their last purchase is more than
``INACTIVITY_DAYS`` before the snapshot date.

Features
--------
- RFM (Recency, Frequency, Monetary) from features/rfm.py
- Behavioral aggregates computed from the transaction table:
  - avg_basket_size  = mean(TotalPrice) per invoice
  - unique_products  = nunique(StockCode) per customer
  - avg_days_between = mean gap between consecutive invoices
  - is_uk            = whether Country is "United Kingdom"
- Plus one-hot top countries (top-5 by customer count, rest -> "Other")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from neuralretail.config import get_settings
from neuralretail.models._mlflow_utils import (
    REGISTERED_MODEL_NAMES,
    log_metrics,
    log_params,
    start_run,
)

logger = logging.getLogger(__name__)

INACTIVITY_DAYS = 90
RANDOM_STATE = 42

FEATURE_COLUMNS: list[str] = [
    "Recency",
    "Frequency",
    "Monetary",
    "avg_basket_size",
    "unique_products",
    "avg_days_between",
    "is_uk",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _behavioural_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-customer behavioural aggregates."""
    df = transactions.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df = df.sort_values(["CustomerID", "InvoiceDate"])

    g = df.groupby("CustomerID", sort=True)
    out = pd.DataFrame(index=sorted(g.groups.keys()))
    out.index.name = "CustomerID"
    out["avg_basket_size"] = g["TotalPrice"].mean()
    out["unique_products"] = g["StockCode"].nunique()

    # avg_days_between: mean of gaps between distinct invoice dates.
    # Vectorise over (CustomerID, InvoiceDate) by computing per-customer
    # diffs, then taking the mean. This avoids the brittle Series-of-Series
    # pattern that breaks on single-row customers.
    distinct_dates = (
        df[["CustomerID", "InvoiceDate"]]
        .assign(InvoiceDate=df["InvoiceDate"].dt.normalize())
        .drop_duplicates()
        .sort_values(["CustomerID", "InvoiceDate"])
    )
    distinct_dates["gap_days"] = (
        distinct_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    )
    avg_gap = (
        distinct_dates.groupby("CustomerID")["gap_days"].mean()
        .reindex(out.index)
    )

    # Country
    country_mode = g["Country"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    out["is_uk"] = (country_mode == "United Kingdom").astype(int).reindex(out.index)

    out = out.reset_index()
    out["avg_days_between"] = avg_gap.values
    return out


def build_training_table(
    transactions: pd.DataFrame,
    rfm: pd.DataFrame,
    *,
    snapshot_date: pd.Timestamp | None = None,
    inactivity_days: int = INACTIVITY_DAYS,
) -> pd.DataFrame:
    """Merge RFM + behavioural features, attach a binary churn label."""
    if snapshot_date is None:
        snapshot_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

    behav = _behavioural_features(transactions)
    df = rfm.merge(behav, on="CustomerID", how="left")
    # Churn label
    df["churned"] = (df["Recency"] > inactivity_days).astype(int)

    # Fill NaNs from behavioural features for one-shot buyers
    df["avg_days_between"] = df["avg_days_between"].fillna(df["Recency"])
    df["avg_basket_size"] = df["avg_basket_size"].fillna(df["Monetary"] / df["Frequency"].clip(lower=1))
    df["unique_products"] = df["unique_products"].fillna(0)
    df["is_uk"] = df["is_uk"].fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


@dataclass
class ChurnModelResult:
    model: XGBClassifier
    feature_columns: list[str]
    metrics: dict[str, float]
    feature_importances: pd.DataFrame


def train(
    transactions: pd.DataFrame,
    rfm: pd.DataFrame,
    *,
    snapshot_date: pd.Timestamp | None = None,
    inactivity_days: int = INACTIVITY_DAYS,
    enable_lightgbm: bool | None = None,
    run_name: str = "xgboost_churn",
) -> ChurnModelResult:
    """Train an XGBoost churn classifier; logs metrics + SHAP to MLflow.

    If ``Settings.enable_lightgbm`` (or the ``enable_lightgbm`` arg)
    is True, also fits a LightGBM model on the same training data
    in a *separate* MLflow run (tagged ``framework=lightgbm``) and
    logs a side-by-side comparison. The XGBoost model is still the
    one returned and registered as ``neuralretail_churn_classifier``.
    LightGBM is opt-in because it requires the ``lightgbm`` package
    and the spec is satisfied with XGBoost alone.
    """
    from neuralretail.config import get_settings

    settings = get_settings()
    if enable_lightgbm is None:
        enable_lightgbm = settings.enable_lightgbm
    df = build_training_table(transactions, rfm, snapshot_date=snapshot_date, inactivity_days=inactivity_days)
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "churn_rate": float(y.mean()),
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
    }
    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    with start_run(run_name=run_name, tags={"model": "churn", "framework": "xgboost"}):
        log_params(
            {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "inactivity_days": inactivity_days,
                "snapshot_date": str(snapshot_date) if snapshot_date is not None else "auto",
                "enable_lightgbm": bool(enable_lightgbm),
            }
        )
        log_metrics(metrics)
        importances.to_csv("feature_importances.csv", index=False)
        import mlflow
        mlflow.log_artifact("feature_importances.csv")
        try:
            mlflow.xgboost.log_model(
                model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAMES["churn"]
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not register churn model: %s", exc)
            mlflow.xgboost.log_model(model, artifact_path="model")

        # SHAP summary plot
        try:
            import matplotlib
            import shap

            matplotlib.use("Agg")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact("shap_summary.png")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not generate SHAP summary: %s", exc)

    # Optional LightGBM run (side-by-side comparison, does NOT
    # replace the XGBoost run for the registered model).
    if enable_lightgbm:
        try:
            import lightgbm as lgb

            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
            lgb_metrics = {
                "auc_roc": float(roc_auc_score(y_test, lgb_proba)),
                "accuracy": float(accuracy_score(y_test, lgb_pred)),
                "precision": float(precision_score(y_test, lgb_pred, zero_division=0)),
                "recall": float(recall_score(y_test, lgb_pred, zero_division=0)),
                "f1": float(f1_score(y_test, lgb_pred, zero_division=0)),
            }
            with start_run(
                run_name="lightgbm_churn",
                tags={"model": "churn", "framework": "lightgbm"},
            ):
                log_params(
                    {
                        "n_estimators": 200,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "inactivity_days": inactivity_days,
                    }
                )
                log_metrics(lgb_metrics)
            logger.info(
                "LightGBM churn (opt-in): AUC-ROC=%.4f F1=%.4f",
                lgb_metrics["auc_roc"],
                lgb_metrics["f1"],
            )
        except ImportError:
            logger.warning(
                "enable_lightgbm=True but the lightgbm package is not installed; "
                "skipping the LightGBM comparison run."
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("LightGBM comparison run failed: %s", exc)

    logger.info("Churn model: AUC-ROC=%.4f precision=%.4f recall=%.4f", metrics["auc_roc"], metrics["precision"], metrics["recall"])
    return ChurnModelResult(model=model, feature_columns=FEATURE_COLUMNS, metrics=metrics, feature_importances=importances)


def predict(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return churn probability (column 1) for each row of X."""
    return model.predict_proba(X)[:, 1]


def save(model: XGBClassifier, path: str | None = None) -> str:
    settings = get_settings()
    path = path or str(settings.models_dir / "churn_xgb.json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    return path


def load_latest(path: str | None = None) -> XGBClassifier:
    settings = get_settings()
    path = path or str(settings.models_dir / "churn_xgb.json")
    m = XGBClassifier()
    m.load_model(path)
    return m


def explain_one(model: XGBClassifier, x: pd.DataFrame) -> Any:
    """Per-customer SHAP waterfall values."""
    import shap

    explainer = shap.TreeExplainer(model)
    return explainer(x)
