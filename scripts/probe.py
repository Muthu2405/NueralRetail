"""One-off probe to validate synthetic generator + Prophet MAPE."""
import importlib
import neuralretail.data.ingest as ig
importlib.reload(ig)
import pandas as pd
import numpy as np
from neuralretail.data.clean import clean
from neuralretail.models.forecasting import train as fc_train

raw = ig.load_raw()
df, report = clean(raw)
print("cleaned rows", len(df))
print(report)
print("cleaned date range", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())
days = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days + 1
print("days", days)
daily = df.assign(Revenue=df["Quantity"] * df["UnitPrice"]).set_index("InvoiceDate").resample("D")["Revenue"].sum()
print("daily describe")
print(daily.describe())
print("daily head 10")
print(daily.head(10).to_string())
print("daily tail 10")
print(daily.tail(10).to_string())
print("CV daily", daily.std() / daily.mean())

# Forecast check
import logging
logging.basicConfig(level=logging.WARNING)
res = fc_train(daily)
print("MAPE", res.metrics["mape"], "RMSE", res.metrics["rmse"])
