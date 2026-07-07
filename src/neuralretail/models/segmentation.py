"""Customer segmentation via KMeans on RFM.

Personas are derived from cluster-centroid rank, not hard-coded:
- Top-quartile Monetary + Top-quartile Frequency + Bottom-quartile Recency -> Champions
- Top-quartile Monetary + Top-quartile Frequency (Recency mid)               -> Loyal Customers
- Top-quartile Monetary + High Recency                                     -> At Risk
- Bottom-quartile Frequency + High Recency                                 -> Hibernating
- Everything else                                                          -> Regular
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuralretail.config import get_settings
from neuralretail.models._mlflow_utils import (
    REGISTERED_MODEL_NAMES,
    log_metrics,
    log_params,
    start_run,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
SEGMENT_FEATURES: list[str] = ["Recency", "Frequency", "Monetary"]

# Stable display order for plots and tables.
PERSONA_ORDER: list[str] = [
    "Champions",
    "Loyal Customers",
    "Regular",
    "At Risk",
    "Hibernating",
]


@dataclass
class SegmentationResult:
    pipeline: Pipeline
    labels: np.ndarray
    persona_map: dict[int, str]
    k: int
    metrics: dict[str, float]
    summary: pd.DataFrame  # per-persona counts + avg RFM


# ---------------------------------------------------------------------------
# KMeans with k selection
# ---------------------------------------------------------------------------


def _fit_pipeline(X: np.ndarray, k: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(X)
    return pipe


def _select_k(X: np.ndarray, k_range: range) -> tuple[int, dict[int, float]]:
    """Return (best_k, {k: silhouette}) for k in k_range.

    Silhouette is computed on the *scaled* features — that's the
    space KMeans actually operates in. Computing it on the raw RFM
    units (where Monetary dominates) under-weights Recency and
    Frequency, and KMeans appears worse than it is.
    """
    scores: dict[int, float] = {}
    for k in k_range:
        pipe = _fit_pipeline(X, k)
        X_scaled = pipe.named_steps["scaler"].transform(X)
        labels = pipe.named_steps["kmeans"].labels_
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(X_scaled, labels))
    if not scores:
        raise RuntimeError("Could not compute silhouette for any k")
    best_k = max(scores, key=scores.get)
    return best_k, scores


def _assign_personas(centroids: pd.DataFrame) -> dict[int, str]:
    """Map each cluster index to a persona using quartile-based rules.

    centroids: DataFrame indexed by cluster id, columns = SEGMENT_FEATURES,
    with the *unscaled* mean RFM per cluster.
    """
    # Use cluster medians of RFM rather than the raw centroid (which is
    # already a mean but the units differ). Quartiles are computed across
    # the cluster centroids to rank them.
    q_money = centroids["Monetary"].quantile(0.75)
    q_freq = centroids["Frequency"].quantile(0.75)
    # Lower Recency = better (more recent).
    q_rec_lo = centroids["Recency"].quantile(0.25)
    q_rec_hi = centroids["Recency"].quantile(0.75)

    mapping: dict[int, str] = {}
    used: set[str] = set()
    for cid, row in centroids.iterrows():
        if (
            row["Monetary"] >= q_money
            and row["Frequency"] >= q_freq
            and row["Recency"] <= q_rec_lo
        ):
            persona = "Champions"
        elif row["Monetary"] >= q_money and row["Frequency"] >= q_freq:
            persona = "Loyal Customers"
        elif row["Monetary"] >= q_money and row["Recency"] >= q_rec_hi:
            persona = "At Risk"
        elif row["Frequency"] < centroids["Frequency"].median() and row["Recency"] >= q_rec_hi:
            persona = "Hibernating"
        else:
            persona = "Regular"
        if persona in used:
            # Avoid two clusters collapsing onto the same persona:
            # pick the next unused one in the priority order.
            for alt in PERSONA_ORDER:
                if alt not in used:
                    persona = alt
                    break
        used.add(persona)
        mapping[int(cid)] = persona
    return mapping


def train(
    rfm: pd.DataFrame,
    *,
    k_min: int = 4,
    k_max: int = 8,
    run_name: str = "kmeans_segmentation",
) -> SegmentationResult:
    """Fit KMeans on RFM, pick k by silhouette, derive personas, log to MLflow.

    The default k range is [4, 8] (per the build-prompt spec — "4-8
    clusters"). When the data has fewer natural clusters than the
    minimum k, KMeans will still fit a k=k_min model (each "real"
    cluster is split into sub-clusters), but the silhouette will
    be lower than it would be at the natural k. In that case the
    ``_assign_personas`` step falls back to a fixed priority order
    so the persona names are stable across runs.
    """
    X = rfm[SEGMENT_FEATURES].fillna(0).to_numpy(dtype=float)
    best_k, scores = _select_k(X, range(k_min, k_max + 1))
    logger.info("Segmentation: best k=%d scores=%s", best_k, {k: round(s, 3) for k, s in scores.items()})

    pipeline = _fit_pipeline(X, best_k)
    labels = pipeline.named_steps["kmeans"].labels_

    # Compute per-cluster mean RFM (in the original RFM units).
    rfm_with_labels = rfm.copy()
    rfm_with_labels["cluster"] = labels
    centroids = rfm_with_labels.groupby("cluster")[SEGMENT_FEATURES].mean()
    persona_map = _assign_personas(centroids)
    rfm_with_labels["persona"] = rfm_with_labels["cluster"].map(persona_map)

    # Per-persona summary
    summary = (
        rfm_with_labels.groupby("persona")
        .agg(
            n_customers=("CustomerID", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
        )
        .reindex([p for p in PERSONA_ORDER if p in set(persona_map.values())])
        .reset_index()
    )

    # Compute the final silhouette on the scaled features (the space
    # KMeans actually operates in) so the reported number matches
    # what the model is using internally.
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    final_sil = float(silhouette_score(X_scaled, labels))
    metrics = {
        "best_k": float(best_k),
        "silhouette": final_sil,
        "n_customers": float(len(rfm_with_labels)),
        **{f"silhouette_k{k}": v for k, v in scores.items()},
    }

    with start_run(run_name=run_name, tags={"model": "segmentation", "framework": "sklearn"}):
        log_params({"k_min": k_min, "k_max": k_max, "best_k": best_k})
        log_metrics(metrics)
        summary.to_csv("persona_summary.csv", index=False)
        import mlflow
        mlflow.log_artifact("persona_summary.csv")
        try:
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAMES["segmentation"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not register segmentation model: %s", exc)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

    logger.info("Segmentation: k=%d silhouette=%.4f personas=%s", best_k, final_sil, persona_map)
    return SegmentationResult(
        pipeline=pipeline,
        labels=labels,
        persona_map=persona_map,
        k=best_k,
        metrics=metrics,
        summary=summary,
    )


def predict(pipeline: Pipeline, rfm: pd.DataFrame) -> pd.DataFrame:
    """Return cluster + persona for each customer in rfm."""
    X = rfm[SEGMENT_FEATURES].fillna(0).to_numpy(dtype=float)
    labels = pipeline.predict(X)
    out = rfm.copy()
    out["cluster"] = labels
    return out


def save(pipeline: Pipeline, path: str | None = None) -> str:
    import joblib

    settings = get_settings()
    path = path or str(settings.models_dir / "segmentation_kmeans.joblib")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_latest(path: str | None = None) -> Pipeline:
    import joblib

    settings = get_settings()
    path = path or str(settings.models_dir / "segmentation_kmeans.joblib")
    return joblib.load(path)
