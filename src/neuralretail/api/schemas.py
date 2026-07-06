"""Pydantic v2 request/response schemas for the neuralretail API.

Each endpoint has a paired Request/Response model. Field validators
catch obvious shape errors (negative numbers, non-monotonic ranges) at
the API boundary instead of letting them bubble up as 500s from the
model layer.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared config: forbid extras, allow pandas-friendly coercion
# ---------------------------------------------------------------------------

_API_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    str_strip_whitespace=True,
    populate_by_name=True,
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    model_config = _API_MODEL_CONFIG

    status: Literal["ok", "degraded"]
    version: str
    models_loaded: dict[str, bool] = Field(
        default_factory=dict,
        description="Map of logical model name -> whether it is loaded.",
    )


# ---------------------------------------------------------------------------
# /predict/demand
# ---------------------------------------------------------------------------


class DemandRequest(BaseModel):
    model_config = _API_MODEL_CONFIG

    horizon_days: int = Field(default=30, ge=1, le=365, description="Days to forecast forward")
    include_history: bool = Field(default=False, description="Include historical fit in response")

    @field_validator("horizon_days")
    @classmethod
    def _horizon_reasonable(cls, v: int) -> int:
        if v < 1 or v > 365:
            raise ValueError("horizon_days must be in [1, 365]")
        return v


class DemandPoint(BaseModel):
    model_config = _API_MODEL_CONFIG

    ds: date
    yhat: float
    yhat_lower: float
    yhat_upper: float


class DemandResponse(BaseModel):
    model_config = _API_MODEL_CONFIG

    horizon_days: int
    points: list[DemandPoint]
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Holdout metrics from the most recent training run (MAPE, RMSE).",
    )


# ---------------------------------------------------------------------------
# /predict/churn
# ---------------------------------------------------------------------------


class ChurnFeatures(BaseModel):
    model_config = _API_MODEL_CONFIG

    recency: float = Field(ge=0, description="Days since last purchase")
    frequency: int = Field(ge=0, description="Distinct invoice count")
    monetary: float = Field(ge=0, description="Total spend")
    avg_basket_size: float = Field(ge=0, default=0.0)
    unique_products: int = Field(ge=0, default=0)
    avg_days_between: float = Field(ge=0, default=0.0)
    is_uk: int = Field(ge=0, le=1, default=0)


class ChurnRequest(BaseModel):
    model_config = _API_MODEL_CONFIG

    customers: list[ChurnFeatures] = Field(min_length=1, max_length=10_000)


class ChurnPrediction(BaseModel):
    model_config = _API_MODEL_CONFIG

    churn_probability: float = Field(ge=0.0, le=1.0)
    churned: bool


class ChurnResponse(BaseModel):
    model_config = _API_MODEL_CONFIG

    predictions: list[ChurnPrediction]
    metrics: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# /segment/score
# ---------------------------------------------------------------------------


class SegmentRequest(BaseModel):
    model_config = _API_MODEL_CONFIG

    customers: list[ChurnFeatures] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _rfm_only(self) -> "SegmentRequest":
        # Segmentation uses only the RFM columns of ChurnFeatures; the
        # other fields are ignored but allowed for client convenience.
        return self


class SegmentPrediction(BaseModel):
    model_config = _API_MODEL_CONFIG

    cluster: int = Field(ge=0)
    persona: str


class SegmentResponse(BaseModel):
    model_config = _API_MODEL_CONFIG

    k: int
    silhouette: float
    persona_map: dict[int, str]
    predictions: list[SegmentPrediction]
    metrics: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# /inventory/reorder
# ---------------------------------------------------------------------------


class InventoryRequest(BaseModel):
    model_config = _API_MODEL_CONFIG

    top_n: int = Field(default=20, ge=1, le=500, description="How many SKUs to return")
    abc_filter: Literal["A", "B", "C", "ALL"] = Field(default="A")
    dead_stock_only: bool = Field(default=False)


class InventoryRow(BaseModel):
    model_config = _API_MODEL_CONFIG

    stock_code: str
    description: str
    abc: Literal["A", "B", "C"]
    units_sold: float
    revenue: float
    annual_demand: float
    eoq: float
    days_since_last_sale: int
    is_dead_stock: bool


class InventoryResponse(BaseModel):
    model_config = _API_MODEL_CONFIG

    rows: list[InventoryRow]
    summary: dict[str, float] = Field(
        default_factory=dict,
        description="Inventory metrics from the most recent training run.",
    )
    generated_at: datetime
