import pandas as pd
import numpy as np
start = pd.Timestamp("2010-12-01")
end = pd.Timestamp("2011-12-09")
n_days = (end - start).days + 1
weekday = np.array([(start + pd.Timedelta(days=i)).weekday() for i in range(n_days)])
weekly = 1.0 + 0.30 * np.sin(2.0 * np.pi * weekday / 7.0)
yearly = 1.0 + 0.20 * np.cos(2.0 * np.pi * np.arange(n_days) / 365.0)
trend = 1.0 + 0.05 * np.linspace(0.0, 1.0, n_days)
for d in [0, 30, 100, 180, 200, 300, 360, 371]:
    print(
        d, start + pd.Timedelta(days=d),
        "weekday", weekday[d], "weekly", round(weekly[d], 3),
        "yearly", round(yearly[d], 3), "trend", round(trend[d], 3),
        "product", round(weekly[d] * yearly[d] * trend[d], 3),
    )
print("min/mean/max weekly", weekly.min(), round(weekly.mean(), 3), weekly.max())
