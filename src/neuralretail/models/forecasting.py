"""Demand forecasting with Prophet.

Input: a daily-revenue frame (DatetimeIndex + ``Revenue`` column).
Output: a fitted Prophet model + forecast (with confidence intervals) +
metrics (MAPE, RMSE) computed on a held-out tail.

Note on LSTM: the build-prompt spec mentioned an optional LSTM ensemble
behind a flag. Prophet alone ships as the default forecaster; LSTM
remains a documented future step (no flag plumbed through the codebase
to avoid a dead-code surface). The natural next step is to fit a
small LSTM on the residuals after Prophet, add it as a second
registered model, and blend the two for the final forecast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import mlflow.pyfunc
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


def _holiday_dataframe(country: str) -> pd.DataFrame | None:
    """Return a Prophet-compatible holidays DataFrame for ``country``.

    Prophet's ``make_holidays_df`` is not exposed in every version; this
    helper falls back to a small built-in list for the most common
    countries (UK / US / DE) so the pipeline doesn't depend on
    ``prophet.hdays`` being importable.
    """
    if not country:
        return None
    key = country.strip().upper()
    try:
        from prophet.make_holidays import make_holidays_df

        years = (2010, 2011, 2012)
        return make_holidays_df(year_list=list(years), country=key)
    except Exception:
        pass
    # Built-in fallback (just the major Q4 / Q1 holidays)
    builtin = {
        "UK": [
            ("2010-12-25", "ChristmasDay"),
            ("2010-12-27", "BoxingDay"),
            ("2011-01-01", "NewYearsDay"),
            ("2011-04-22", "GoodFriday"),
            ("2011-04-25", "EasterMonday"),
            ("2011-12-25", "ChristmasDay"),
            ("2011-12-26", "BoxingDay"),
        ],
        "US": [
            ("2010-11-25", "Thanksgiving"),
            ("2010-12-25", "ChristmasDay"),
            ("2011-01-01", "NewYearsDay"),
            ("2011-07-04", "IndependenceDay"),
            ("2011-11-24", "Thanksgiving"),
            ("2011-12-25", "ChristmasDay"),
        ],
        "DE": [
            ("2010-12-25", "ChristmasDay"),
            ("2010-10-03", "GermanUnity"),
            ("2011-01-01", "NewYearsDay"),
            ("2011-05-01", "LabourDay"),
            ("2011-10-03", "GermanUnity"),
            ("2011-12-25", "ChristmasDay"),
        ],
    }
    rows = builtin.get(key)
    if rows is None:
        logger.warning("Unknown Prophet holiday country %r; disabling holidays", country)
        return None
    return pd.DataFrame(rows, columns=["ds", "holiday"])


def _build_prophet(
    *,
    weekly_seasonality: bool,
    yearly_seasonality: bool,
    weekly_fourier_order: int,
    yearly_fourier_order: int,
    seasonality_mode: str,
    changepoint_prior_scale: float,
    holidays: pd.DataFrame | None,
) -> Prophet:
    """Construct a Prophet model with the configured hyper-parameters."""
    model = Prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        holidays=holidays,
    )
    # Add explicit Fourier orders so the trained model is reproducible
    # across runs and settings. Without this, Prophet auto-picks based
    # on the data window.
    if weekly_seasonality:
        model.add_seasonality(
            name="weekly",
            period=7,
            fourier_order=weekly_fourier_order,
            mode=seasonality_mode,
        )
    if yearly_seasonality:
        model.add_seasonality(
            name="yearly",
            period=365.25,
            fourier_order=yearly_fourier_order,
            mode=seasonality_mode,
        )
    return model


def train(
    daily: pd.DataFrame,
    *,
    horizon_days: int = 30,
    holdout_days: int = 30,
    weekly_seasonality: bool | None = None,
    yearly_seasonality: bool | None = None,
    run_name: str = "prophet_demand",
) -> ForecastResult:
    """Train a Prophet forecaster with a tail holdout for honest evaluation.

    Hyper-parameters (changepoint_prior_scale, seasonality_mode,
    weekly/yearly Fourier orders, holiday calendar) are pulled from
    :class:`Settings` so the trained model is reproducible from the
    environment. The seasonality flags default to None, which means
    "use the value from Settings" — set them explicitly to override
    for a single run.
    """
    from neuralretail.config import get_settings

    settings = get_settings()
    if weekly_seasonality is None:
        weekly_seasonality = True
    if yearly_seasonality is None:
        yearly_seasonality = settings.prophet_yearly_seasonality
    df = _prepare(daily)
    if len(df) <= holdout_days + 14:
        raise ValueError(
            f"need at least {holdout_days + 14} days of history; got {len(df)}"
        )

    train_df = df.iloc[:-holdout_days].copy()
    test_df = df.iloc[-holdout_days:].copy()

    holidays = _holiday_dataframe(settings.prophet_holidays_country)

    model = _build_prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        weekly_fourier_order=settings.prophet_weekly_fourier_order,
        yearly_fourier_order=settings.prophet_yearly_fourier_order,
        seasonality_mode=settings.prophet_seasonality_mode,
        changepoint_prior_scale=settings.prophet_changepoint_prior_scale,
        holidays=holidays,
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
    full_model = _build_prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        weekly_fourier_order=settings.prophet_weekly_fourier_order,
        yearly_fourier_order=settings.prophet_yearly_fourier_order,
        seasonality_mode=settings.prophet_seasonality_mode,
        changepoint_prior_scale=settings.prophet_changepoint_prior_scale,
        holidays=holidays,
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

    # Log to MLflow (params, metrics, JSON artifact, plus pyfunc registration)
    with start_run(run_name=run_name, tags={"model": "forecasting", "framework": "prophet"}):
        log_params(
            {
                "horizon_days": horizon_days,
                "holdout_days": holdout_days,
                "weekly_seasonality": weekly_seasonality,
                "yearly_seasonality": yearly_seasonality,
                "weekly_fourier_order": settings.prophet_weekly_fourier_order,
                "yearly_fourier_order": settings.prophet_yearly_fourier_order,
                "seasonality_mode": settings.prophet_seasonality_mode,
                "changepoint_prior_scale": settings.prophet_changepoint_prior_scale,
                "holidays_country": settings.prophet_holidays_country or "(none)",
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        )
        log_metrics(result.metrics)
        forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)
        forecast_out.to_csv("forecast.csv", index=False)
        import mlflow

        model_json = model_to_json(full_model)
        with open("model.json", "w") as f:
            f.write(model_json)
        mlflow.log_artifact("model.json")
        mlflow.log_artifact("forecast.csv")

        # Register the Prophet model as a pyfunc so the API can load
        # it from the MLflow registry (Production alias) and not just
        # from the on-disk JSON.
        try:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=_ProphetPyFunc(full_model),
                registered_model_name=REGISTERED_MODEL_NAMES["forecasting"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not register Prophet pyfunc: %s", exc)
    logger.info(
        "Prophet trained: MAPE=%.4f RMSE=%.2f horizon=%dd mode=%s",
        mape,
        rmse,
        horizon_days,
        settings.prophet_seasonality_mode,
    )
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


# ---------------------------------------------------------------------------
# MLflow pyfunc wrapper — registers Prophet under a registered model name
# so the API can load it from the registry at startup.
# ---------------------------------------------------------------------------


class _ProphetPyFunc(mlflow.pyfunc.PythonModel):
    """Trivial MLflow pyfunc wrapper that exposes a Prophet forecast.

    The wrapper holds the fitted Prophet model and a small
    ``predict()`` that takes a list of horizon days and returns a
    DataFrame of (ds, yhat, yhat_lower, yhat_upper).

    This is the contract the FastAPI ``/predict/demand`` endpoint uses
    when loading from the MLflow registry. The on-disk JSON path is
    still the primary loader for the dashboard (faster, no registry
    needed) and is used as a fallback if the registry is empty.
    """

    def __init__(self, prophet_model: Prophet) -> None:
        super().__init__()
        self._model = prophet_model

    def predict(self, context, model_input):  # noqa: D401
        import pandas as pd

        # Accept a DataFrame (or list of dicts) with a 'horizon_days'
        # column or a single int. Returns a DataFrame of forecast
        # rows.
        if isinstance(model_input, pd.DataFrame):
            if "horizon_days" in model_input.columns:
                horizon = int(model_input["horizon_days"].iloc[0])
            elif "horizon" in model_input.columns:
                horizon = int(model_input["horizon"].iloc[0])
            else:
                horizon = 30
        elif isinstance(model_input, (list, tuple)) and model_input:
            horizon = int(model_input[0])
        else:
            horizon = 30
        future = self._model.make_future_dataframe(periods=horizon, freq="D")
        fc = self._model.predict(future)
        return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
