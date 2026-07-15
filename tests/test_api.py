"""Smoke tests for the FastAPI scoring service.

These tests use FastAPI's TestClient. They require the models to have
been trained (i.e. ``models/*.json``, ``models/*.joblib``,
``models/inventory_table.csv`` to exist). If they're missing, the
``/health`` endpoint will report ``degraded`` and the per-endpoint
tests will skip.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Force a known API key for the test session before importing the app
os.environ.setdefault("NEURALRETAIL_API_KEY", "test-key")
os.environ.setdefault("NEURALRETAIL_LOG_LEVEL", "WARNING")

from neuralretail.api.main import app  # noqa: E402
from neuralretail.config import get_settings  # noqa: E402

API_KEY = "test-key"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def models_available() -> bool:
    s = get_settings()
    return all(
        (s.models_dir / name).exists()
        for name in (
            "prophet_demand.json",
            "churn_xgb.json",
            "segmentation_kmeans.joblib",
            "inventory_table.csv",
        )
    )


def test_health_no_auth_required(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "version" in body
    assert "models_loaded" in body


def test_churn_endpoint_requires_api_key(client: TestClient) -> None:
    r = client.post(
        "/predict/churn",
        json={"customers": [{"recency": 10, "frequency": 2, "monetary": 100.0}]},
    )
    assert r.status_code == 401


def test_churn_endpoint_rejects_bad_payload(client: TestClient) -> None:
    r = client.post(
        "/predict/churn",
        headers=HEADERS,
        json={"customers": [{"recency": -1, "frequency": 2, "monetary": 100.0}]},
    )
    assert r.status_code == 422


def test_churn_endpoint_happy_path(client: TestClient, models_available: bool) -> None:
    if not models_available:
        pytest.skip("Models not trained yet; skipping churn happy path")
    r = client.post(
        "/predict/churn",
        headers=HEADERS,
        json={
            "customers": [
                {
                    "recency": 10,
                    "frequency": 5,
                    "monetary": 1000.0,
                    "avg_basket_size": 50.0,
                    "unique_products": 10,
                    "avg_days_between": 14.0,
                    "is_uk": 1,
                },
                {
                    "recency": 200,
                    "frequency": 1,
                    "monetary": 50.0,
                    "avg_basket_size": 50.0,
                    "unique_products": 1,
                    "avg_days_between": 200.0,
                    "is_uk": 1,
                },
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert 0.0 <= p["churn_probability"] <= 1.0
        assert isinstance(p["churned"], bool)


def test_segment_endpoint_happy_path(client: TestClient, models_available: bool) -> None:
    if not models_available:
        pytest.skip("Models not trained yet; skipping segment happy path")
    r = client.post(
        "/segment/score",
        headers=HEADERS,
        json={
            "customers": [
                {"recency": 5, "frequency": 10, "monetary": 2000.0},
                {"recency": 200, "frequency": 1, "monetary": 30.0},
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "persona_map" in body
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert p["persona"]


def test_inventory_endpoint_happy_path(client: TestClient, models_available: bool) -> None:
    if not models_available:
        pytest.skip("Models not trained yet; skipping inventory happy path")
    r = client.post(
        "/inventory/reorder",
        headers=HEADERS,
        json={"top_n": 5, "abc_filter": "A", "dead_stock_only": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "rows" in body
    assert "summary" in body
    assert len(body["rows"]) <= 5
    for row in body["rows"]:
        assert row["abc"] in {"A", "B", "C"}
        assert row["eoq"] >= 0


def test_demand_endpoint_happy_path(client: TestClient, models_available: bool) -> None:
    if not models_available:
        pytest.skip("Models not trained yet; skipping demand happy path")
    r = client.post(
        "/predict/demand",
        headers=HEADERS,
        json={"horizon_days": 7, "include_history": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["horizon_days"] == 7
    assert len(body["points"]) == 7
    for p in body["points"]:
        assert "ds" in p
        assert "yhat" in p
