"""Debug Prophet forecast quality — different settings."""
import importlib
import neuralretail.data.ingest as ig
importlib.reload(ig)
import pandas as pd
import numpy as np
from neuralretail.data.clean import clean
from neuralretail.models.forecasting import _prepare, _build_prophet, _mape
from neuralretail.config import get_settings
from prophet import Prophet
from sklearn.metrics import mean_squared_error

raw = ig.load_raw()
df, _ = clean(raw)
daily = df.assign(Revenue=df["Quantity"] * df["UnitPrice"]).set_index("InvoiceDate").resample("D")["Revenue"].sum()
ds = _prepare(daily)
train_df = ds.iloc[:-30].copy()
test_df = ds.iloc[-30:].copy()

settings = get_settings()
for yearly, fourier, mode in [
    (False, 10, "multiplicative"),
    (True, 10, "additive"),
    (True, 5, "multiplicative"),
    (False, 0, "additive"),
]:
    m = _build_prophet(
        weekly_seasonality=True,
        yearly_seasonality=yearly,
        weekly_fourier_order=settings.prophet_weekly_fourier_order,
        yearly_fourier_order=fourier,
        seasonality_mode=mode,
        changepoint_prior_scale=settings.prophet_changepoint_prior_scale,
        holidays=None,
    )
    m.fit(train_df)
    pred = m.predict(test_df[["ds"]])
    y_true = test_df["y"].values
    y_pred = pred["yhat"].values
    mape = _mape(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"yearly={yearly} fourier={fourier} mode={mode}: MAPE={mape:.4f} RMSE={rmse:.2f} mean_pred={y_pred.mean():.0f} mean_actual={y_true.mean():.0f}")
