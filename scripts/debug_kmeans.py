"""Debug KMeans silhouette per k."""
import importlib
import neuralretail.data.ingest as ig
importlib.reload(ig)
import pandas as pd
import numpy as np
from neuralretail.data.clean import clean
from neuralretail.features.rfm import compute_rfm
from neuralretail.models.segmentation import _fit_pipeline, _select_k
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

raw = ig.load_raw()
df, _ = clean(raw)
rfm = compute_rfm(df)
X = rfm[["Recency", "Frequency", "Monetary"]].fillna(0).to_numpy(dtype=float)
print("RFM shape", rfm.shape)

for k in range(3, 9):
    pipe = _fit_pipeline(X, k)
    X_scaled = pipe.named_steps["scaler"].transform(X)
    labels = pipe.named_steps["kmeans"].labels_
    sil = silhouette_score(X_scaled, labels)
    # Centroids (in scaled space)
    centroids_scaled = pipe.named_steps["kmeans"].cluster_centers_
    # Centroid distances
    from scipy.spatial.distance import pdist
    if k > 1:
        d = pdist(centroids_scaled)
        print(f"k={k}: silhouette={sil:.4f} min_centroid_dist={d.min():.2f} mean_centroid_dist={d.mean():.2f}")
    else:
        print(f"k={k}: silhouette={sil:.4f}")
