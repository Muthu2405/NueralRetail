"""FastAPI scoring service for the neuralretail platform.

Endpoints
---------
- GET  /health                 (no auth)  — liveness + model-load status
- POST /predict/demand         (auth)     — Prophet forecast
- POST /predict/churn          (auth)     — XGBoost churn probability
- POST /segment/score          (auth)     — KMeans cluster + persona
- POST /inventory/reorder      (auth)     — ABC + EOQ reorder list

Models are loaded once at startup from the on-disk artifacts
(``models/``). The MLflow registry is the source of truth in production;
for local dev, the on-disk loaders are faster and don't need a running
MLflow server.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from prophet import Prophet

from neuralretail import __version__
from neuralretail.api.schemas import (
    ChurnPrediction,
    ChurnRequest,
    ChurnResponse,
    DemandPoint,
    DemandRequest,
    DemandResponse,
    HealthResponse,
    InventoryRequest,
    InventoryResponse,
    InventoryRow,
    SegmentPrediction,
    SegmentRequest,
    SegmentResponse,
)
from neuralretail.config import get_settings
from neuralretail.features.timeseries import build_daily_revenue
from neuralretail.models import churn as churn_mod
from neuralretail.models import forecasting as fc_mod
from neuralretail.models import inventory as inv_mod
from neuralretail.models import segmentation as seg_mod

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# App + lifespan (model loading)
# ---------------------------------------------------------------------------


# Module-level state populated by the lifespan handler.
class _State:
    prophet: Prophet | None = None
    churn_model: Any = None
    seg_pipeline: Any = None
    inventory_table: pd.DataFrame | None = None
    loaded: dict[str, bool] = {}


def _safe_load_prophet() -> Prophet | None:
    path = get_settings().models_dir / "prophet_demand.json"
    if not path.exists():
        logger.warning("Prophet model not found at %s", path)
        return None
    try:
        return fc_mod.load_latest(str(path))
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load Prophet: %s", exc)
        return None


def _safe_load_churn() -> Any:
    path = get_settings().models_dir / "churn_xgb.json"
    if not path.exists():
        logger.warning("Churn model not found at %s", path)
        return None
    try:
        return churn_mod.load_latest(str(path))
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load churn model: %s", exc)
        return None


def _safe_load_segmentation() -> Any:
    path = get_settings().models_dir / "segmentation_kmeans.joblib"
    if not path.exists():
        logger.warning("Segmentation model not found at %s", path)
        return None
    try:
        return seg_mod.load_latest(str(path))
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load segmentation: %s", exc)
        return None


def _safe_load_inventory() -> pd.DataFrame | None:
    path = get_settings().models_dir / "inventory_table.csv"
    if not path.exists():
        logger.warning("Inventory table not found at %s", path)
        return None
    try:
        return inv_mod.load_latest(str(path))
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load inventory: %s", exc)
        return None


def _load_models_into_state() -> None:
    """Populate ``_State`` from on-disk artifacts. Idempotent."""
    t0 = time.time()
    _State.prophet = _safe_load_prophet()
    _State.churn_model = _safe_load_churn()
    _State.seg_pipeline = _safe_load_segmentation()
    _State.inventory_table = _safe_load_inventory()
    _State.loaded = {
        "forecasting": _State.prophet is not None,
        "churn": _State.churn_model is not None,
        "segmentation": _State.seg_pipeline is not None,
        "inventory": _State.inventory_table is not None,
    }
    logger.info("Models loaded in %.2fs: %s", time.time() - t0, _State.loaded)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup. No teardown needed (artifacts are on disk)."""
    _load_models_into_state()
    yield


app = FastAPI(
    title="NeuralRetail API",
    version=__version__,
    description="AI-powered retail sales intelligence — forecasting, churn, segmentation, inventory.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _require_api_key(api_key: str | None = Depends(API_KEY_HEADER)) -> None:
    settings = get_settings()
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    all_loaded = all(_State.loaded.values()) if _State.loaded else False
    return HealthResponse(
        status="ok" if all_loaded else "degraded",
        version=__version__,
        models_loaded=dict(_State.loaded),
    )


# ---------------------------------------------------------------------------
# /predict/demand
# ---------------------------------------------------------------------------


@app.post("/predict/demand", response_model=DemandResponse, dependencies=[Depends(_require_api_key)])
def predict_demand(req: DemandRequest) -> DemandResponse:
    if _State.prophet is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not loaded. Run `make train` first.",
        )

    future = _State.prophet.make_future_dataframe(periods=req.horizon_days, freq="D")
    forecast = _State.prophet.predict(future)
    tail = forecast.tail(req.horizon_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    points = [
        DemandPoint(
            ds=row["ds"].date() if hasattr(row["ds"], "date") else row["ds"],
            yhat=float(row["yhat"]),
            yhat_lower=float(row["yhat_lower"]),
            yhat_upper=float(row["yhat_upper"]),
        )
        for _, row in tail.iterrows()
    ]
    return DemandResponse(horizon_days=req.horizon_days, points=points, metrics={})


# ---------------------------------------------------------------------------
# /predict/churn
# ---------------------------------------------------------------------------


@app.post("/predict/churn", response_model=ChurnResponse, dependencies=[Depends(_require_api_key)])
def predict_churn(req: ChurnRequest) -> ChurnResponse:
    if _State.churn_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn model is not loaded. Run `make train` first.",
        )

    feature_rows = [
        {
            "Recency": c.recency,
            "Frequency": c.frequency,
            "Monetary": c.monetary,
            "avg_basket_size": c.avg_basket_size,
            "unique_products": c.unique_products,
            "avg_days_between": c.avg_days_between,
            "is_uk": c.is_uk,
        }
        for c in req.customers
    ]
    X = pd.DataFrame(feature_rows)[churn_mod.FEATURE_COLUMNS].fillna(0)
    probas = _State.churn_model.predict_proba(X)[:, 1]
    predictions = [
        ChurnPrediction(churn_probability=float(p), churned=bool(p >= 0.5))
        for p in probas
    ]
    return ChurnResponse(predictions=predictions, metrics={})


# ---------------------------------------------------------------------------
# /segment/score
# ---------------------------------------------------------------------------


@app.post("/segment/score", response_model=SegmentResponse, dependencies=[Depends(_require_api_key)])
def segment_score(req: SegmentRequest) -> SegmentResponse:
    if _State.seg_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Segmentation model is not loaded. Run `make train` first.",
        )

    rfm = pd.DataFrame(
        [
            {
                "Recency": c.recency,
                "Frequency": c.frequency,
                "Monetary": c.monetary,
            }
            for c in req.customers
        ]
    )
    labels = _State.seg_pipeline.predict(rfm[seg_mod.SEGMENT_FEATURES].fillna(0).to_numpy(dtype=float))
    k = _State.seg_pipeline.named_steps["kmeans"].n_clusters

    # Re-derive persona map from training data centroids. We persisted
    # only the pipeline, so recompute against this batch.
    centroids = pd.DataFrame(_State.seg_pipeline.named_steps["scaler"].inverse_transform(
        _State.seg_pipeline.named_steps["kmeans"].cluster_centers_
    ), columns=seg_mod.SEGMENT_FEATURES)
    persona_map = seg_mod._assign_personas(centroids)

    predictions = [
        SegmentPrediction(cluster=int(lab), persona=persona_map.get(int(lab), f"cluster_{lab}"))
        for lab in labels
    ]
    return SegmentResponse(
        k=k,
        silhouette=-1.0,  # not re-evaluated at request time
        persona_map=persona_map,
        predictions=predictions,
        metrics={},
    )


# ---------------------------------------------------------------------------
# /inventory/reorder
# ---------------------------------------------------------------------------


@app.post(
    "/inventory/reorder", response_model=InventoryResponse, dependencies=[Depends(_require_api_key)]
)
def inventory_reorder(req: InventoryRequest) -> InventoryResponse:
    if _State.inventory_table is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inventory table is not loaded. Run `make train` first.",
        )

    table = _State.inventory_table
    df = table.copy()
    if req.abc_filter != "ALL":
        df = df[df["ABC"] == req.abc_filter]
    if req.dead_stock_only:
        df = df[df["IsDeadStock"] == 1]
    df = df.sort_values("Revenue", ascending=False).head(req.top_n)

    rows = [
        InventoryRow(
            stock_code=str(r["StockCode"]),
            description=str(r["Description"]),
            abc=str(r["ABC"]),
            units_sold=float(r["UnitsSold"]),
            revenue=float(r["Revenue"]),
            annual_demand=float(r["AnnualDemand"]),
            eoq=float(r["EOQ"]),
            days_since_last_sale=int(r["DaysSinceLastSale"]),
            is_dead_stock=bool(int(r["IsDeadStock"])),
        )
        for _, r in df.iterrows()
    ]

    summary: dict[str, float] = {}
    for col in (
        "n_skus",
        "n_class_a",
        "n_class_b",
        "n_class_c",
        "n_dead_stock",
        "dead_stock_pct",
        "total_revenue",
        "span_years",
    ):
        if col in table.columns:
            summary[col] = float(table[col].iloc[0])

    return InventoryResponse(rows=rows, summary=summary, generated_at=datetime.now(timezone.utc))
