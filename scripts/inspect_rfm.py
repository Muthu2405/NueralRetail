"""Inspect RFM distribution after fresh data generation."""
import pandas as pd
import numpy as np
from neuralretail.features.rfm import compute_rfm
from neuralretail.data.clean import clean
from neuralretail.data.ingest import load_raw

raw = load_raw()
df, _ = clean(raw)
rfm = compute_rfm(df)
print("RFM shape", rfm.shape)
print(rfm.describe())
print()
print("Histogram-ish counts per (R-bucket, F-bucket)")
# Buckets: Recency <10, 10-30, 30-60, 60-120, 120+
r_buckets = pd.cut(rfm["Recency"], bins=[-1, 10, 30, 60, 120, 1000], labels=["<10", "10-30", "30-60", "60-120", "120+"])
f_buckets = pd.cut(rfm["Frequency"], bins=[0, 2, 5, 10, 20, 1000], labels=["1-2", "3-5", "6-10", "11-20", "20+"])
m_buckets = pd.cut(rfm["Monetary"], bins=[0, 300, 800, 1500, 3000, 1e6], labels=["<300", "300-800", "800-1500", "1500-3000", "3000+"])
ct = pd.crosstab([r_buckets, f_buckets], m_buckets, margins=True)
print(ct.to_string())
