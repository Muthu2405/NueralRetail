"""Demand forecasting with Prophet.

Input: a daily-revenue frame (DatetimeIndex + ``Revenue`` column).
Output: a fitted Prophet model + forecast (with confidence intervals) +
metrics (MAPE, RMSE) computed on a held-out tail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from neuralretail.models._mlflow_utils import (
    REGISTERED_MODEL_NAMES,
    log_metrics,
    log_params,
    start_run,
)
from neuralretail.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Holds the trained model, in-sample fit, and out-of-sample forecast."""

    model: Prophet
    history: pd.DataFrame  # original revenue history (ds, y)
    fitted: pd.DataFrame  # Prophet's in-sample prediction
    forecast: pd.DataFrame  # Prophet's full forecast (history + horizon)
    metrics: dict[str, float]  # MAPE, RMSE on holdout
    horizon_days: int


def _prepare(daily: pd.DataFrame) -> pd.DataFrame:
    """Reshape a daily-revenue frame into Prophet's expected (ds, y) layout."""
    if isinstance(daily.index, pd.DatetimeIndex):
        df = daily.reset_index()
    else:
        df = daily.copy()
    rename = {df.columns[0]: "ds"}
    df = df.rename(columns=rename)
    if "Revenue" in df.columns:
        df = df.rename(columns={"Revenue": "y"})
    elif "y" not in df.columns:
        raise ValueError("daily frame must contain a 'Revenue' column")
    return df[["ds", "y"]]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE, ignoring zero actuals to avoid division-by-zero."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))


def train(
    daily: pd.DataFrame,
    *,
    horizon_days: int = 30,
    holdout_days: int = 30,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    run_name: str = "prophet_demand",
) -> ForecastResult:
    """Train a Prophet forecaster with a tail holdout for honest evaluation."""
    df = _prepare(daily)
    if len(df) <= holdout_days + 14:
        raise ValueError(
            f"need at least {holdout_days + 14} days of history; got {len(df)}"
        )

    train_df = df.iloc[:-holdout_days].copy()
    test_df = df.iloc[-holdout_days:].copy()

    model = Prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
    )
    model.fit(train_df)

    # Evaluate on holdout
    future_test = test_df[["ds"]].copy()
    pred_test = model.predict(future_test)
    y_true = test_df["y"].values
    y_pred = pred_test["yhat"].values
    mape = _mape(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Refit on the full history and forecast the horizon
    full_model = Prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
    )
    full_model.fit(df)
    future = full_model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = full_model.predict(future)

    result = ForecastResult(
        model=full_model,
        history=df,
        fitted=pred_test,
        forecast=forecast,
        metrics={"mape": mape, "rmse": rmse, "holdout_days": float(holdout_days)},
        horizon_days=horizon_days,
    )

    # Log to MLflow
    with start_run(run_name=run_name, tags={"model": "forecasting", "framework": "prophet"}):
        log_params(
            {
                "horizon_days": horizon_days,
                "holdout_days": holdout_days,
                "weekly_seasonality": weekly_seasonality,
                "yearly_seasonality": yearly_seasonality,
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        )
        log_metrics(result.metrics)
        # Save the forecast as a CSV artifact
        forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)
        forecast_out.to_csv("forecast.csv", index=False)
        # Save the model (Prophet is a pyfunc; serialize to JSON)
        import mlflow

        model_json = model_to_json(full_model)
        with open("model.json", "w") as f:
            f.write(model_json)
        mlflow.log_artifact("model.json")
        mlflow.log_artifact("forecast.csv")
    logger.info("Prophet trained: MAPE=%.4f RMSE=%.2f horizon=%dd", mape, rmse, horizon_days)
    return result


def predict(model: Prophet, periods: int) -> pd.DataFrame:
    """Forecast ``periods`` days into the future from the model's last fit point."""
    future = model.make_future_dataframe(periods=periods, freq="D")
    return model.predict(future)


def save(model: Prophet, path: str | None = None) -> str:
    """Serialize a Prophet model to disk (JSON)."""
    import json
    from pathlib import Path

    settings = get_settings()
    path = path or str(settings.models_dir / "prophet_demand.json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(model_to_json(model))
    return path


def load_latest(path: str | None = None) -> Prophet:
    """Load a Prophet model from a JSON file."""
    from pathlib import Path

    settings = get_settings()
    path = path or str(settings.models_dir / "prophet_demand.json")
    return model_from_json(Path(path).read_text())
