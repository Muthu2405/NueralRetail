"""Tests for the customer segmentation module (KMeans on RFM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuralretail.models.segmentation import (
    PERSONA_ORDER,
    _assign_personas,
    _select_k,
    train,
)


def _rfm_row(cid, recency, freq, monetary):
    return {
        "CustomerID": cid,
        "Recency": recency,
        "Frequency": freq,
        "Monetary": monetary,
        "FirstPurchase": pd.Timestamp("2011-01-01"),
        "LastPurchase": pd.Timestamp("2011-12-01") - pd.Timedelta(days=recency),
    }


def test_select_k_picks_3_for_three_obvious_clusters():
    """A fixture with three well-separated clusters should pick k=3."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(50):
        rows.append(_rfm_row(len(rows), rng.normal(5, 1), rng.normal(20, 2), rng.normal(5000, 200)))
    for _ in range(50):
        rows.append(_rfm_row(len(rows), rng.normal(45, 5), rng.normal(5, 1), rng.normal(500, 100)))
    for _ in range(50):
        rows.append(_rfm_row(len(rows), rng.normal(150, 10), rng.normal(1, 0.2), rng.normal(100, 30)))
    rfm = pd.DataFrame(rows)
    X = rfm[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float)
    best_k, scores = _select_k(X, range(3, 8))
    assert best_k == 3
    # All silhouettes are in (-1, 1).
    for s in scores.values():
        assert -1.0 <= s <= 1.0


def test_assign_personas_uses_priority_order():
    """When centroid ordering is ambiguous (e.g. 2 clusters but 5
    personas defined), the mapper should fall back to PERSONA_ORDER
    so the names are stable.
    """
    centroids = pd.DataFrame(
        {
            "Recency": [10.0, 100.0],
            "Frequency": [15.0, 3.0],
            "Monetary": [4000.0, 200.0],
        },
        index=[0, 1],
    )
    mapping = _assign_personas(centroids)
    # Both personas should be in PERSONA_ORDER
    for v in mapping.values():
        assert v in PERSONA_ORDER
    # No two clusters share a persona
    assert len(set(mapping.values())) == len(mapping)


def test_assign_personas_prefers_champions_for_low_r_high_f_high_m():
    centroids = pd.DataFrame(
        {
            "Recency": [3.0],
            "Frequency": [25.0],
            "Monetary": [6000.0],
        },
        index=[0],
    )
    mapping = _assign_personas(centroids)
    assert mapping[0] == "Champions"


def test_end_to_end_train_small_synthetic():
    """Train on a small fixture with 3 well-separated clusters and
    assert the silhouette is reasonable."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(50):
        rows.append(_rfm_row(len(rows), rng.normal(5, 1), rng.normal(20, 2), rng.normal(5000, 200)))
    for _ in range(50):
        rows.append(_rfm_row(len(rows), rng.normal(45, 5), rng.normal(5, 1), rng.normal(500, 100)))
    for _ in range(50):
        rows.append(_rfm_row(len(rows), rng.normal(150, 10), rng.normal(1, 0.2), rng.normal(100, 30)))
    rfm = pd.DataFrame(rows)
    res = train(rfm, k_min=3, k_max=4, run_name="test_seg")
    assert res.k in (3, 4)
    assert res.metrics["silhouette"] > 0.3
    assert len(res.persona_map) == res.k
    # Summary has one row per persona
    assert len(res.summary) == res.k
