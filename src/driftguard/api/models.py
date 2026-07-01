"""Typed request/response schemas. These contracts are stable — changing a field
name or type is a breaking change to the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw news text to classify.")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted AG News topic.")
    label_id: int
    confidence: float
    served_by: str = Field(..., description="'primary' or 'baseline'.")
    latency_ms: float
    scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str


class ReadyResponse(BaseModel):
    ready: bool
    servable_tiers: list[str]


class ModelInfo(BaseModel):
    active_tier: str
    primary_version: str | None
    primary_available: bool
    baseline_version: str
    primary_source: str | None = None
