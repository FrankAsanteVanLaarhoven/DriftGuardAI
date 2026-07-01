"""FastAPI serving layer implementing the DriftGuard fallback contract.

Guarantees:
* The baseline loads at startup or the process exits (never start with no model).
* ``/ready`` is model-agnostic: 200 as long as *any* model can serve.
* ``/predict`` tries the primary; on any error or latency-budget breach it serves
  the baseline, increments ``driftguard_fallback_total``, and tags the response
  ``served_by: "baseline"``. A bad primary never yields a 5xx.
* ``driftguard_model_tier{tier="baseline"}`` = 1 while running degraded.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from driftguard import registry
from driftguard.api.models import (
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)
from driftguard.config import Settings, get_settings

log = logging.getLogger("driftguard.api")

# Module-level metrics: registered once against the default registry and shared
# across app instances (so building a fresh app in tests never double-registers).
MODEL_TIER = Gauge("driftguard_model_tier", "Active serving tier (1=active).", ["tier"])
FALLBACK_TOTAL = Counter("driftguard_fallback_total", "Requests served by the baseline fallback.")
PREDICTIONS_TOTAL = Counter("driftguard_predictions_total", "Predictions served.", ["served_by"])
BUDGET_BREACH_TOTAL = Counter(
    "driftguard_primary_latency_breach_total", "Primary predictions over the latency budget."
)


class LatencyBudgetExceeded(Exception):
    """Raised when the primary exceeds its per-request latency budget."""


def refresh_models(app: FastAPI) -> None:
    """(Re)load baseline + primary into app state and update the tier gauge.

    The baseline is mandatory; failure here is fatal. The primary is best-effort.
    """
    settings: Settings = app.state.settings
    app.state.baseline = registry.load_baseline(settings)  # raises if unusable
    primary, source = registry.load_primary(settings)
    app.state.primary = primary
    app.state.primary_source = source
    app.state.active_tier = "primary" if primary is not None else "baseline"

    MODEL_TIER.labels(tier="primary").set(1 if primary is not None else 0)
    MODEL_TIER.labels(tier="baseline").set(0 if primary is not None else 1)
    if primary is None:
        log.warning("Running DEGRADED: primary unavailable, serving baseline only.")
    else:
        log.info("Primary available (%s); baseline standing by.", source)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "settings"):
        app.state.settings = get_settings()
    logging.basicConfig(level=app.state.settings.log_level)
    refresh_models(app)
    yield


def _degrade(app: FastAPI) -> None:
    """Drop the primary and mark the service degraded (baseline-only)."""
    app.state.primary = None
    app.state.primary_source = None
    app.state.active_tier = "baseline"
    MODEL_TIER.labels(tier="primary").set(0)
    MODEL_TIER.labels(tier="baseline").set(1)


def _primary_backed(app: FastAPI) -> bool:
    """True unless a pointer-sourced primary's backing artifact has vanished.

    Lets an operator rotate/remove the primary out from under a running service:
    the next request detects the missing file and degrades to baseline gracefully.
    """
    source = getattr(app.state, "primary_source", None) or ""
    if source.startswith("pointer:"):
        from pathlib import Path

        target = Path(source.split("pointer:", 1)[1])
        pointer = app.state.settings.primary_pointer_path
        return target.exists() and pointer.exists()
    return True


def _serve(app: FastAPI, text: str) -> tuple[dict, str]:
    """Return (prediction, served_by). Never raises for a model problem."""
    settings: Settings = app.state.settings
    primary = app.state.primary
    if primary is not None and not _primary_backed(app):
        log.warning("Primary backing artifact vanished; degrading to baseline.")
        _degrade(app)
        primary = None
    if primary is not None:
        try:
            t0 = time.perf_counter()
            result = registry.predict(primary, [text])[0]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > settings.primary_latency_budget_ms:
                BUDGET_BREACH_TOTAL.inc()
                raise LatencyBudgetExceeded(f"{elapsed_ms:.1f}ms > "
                                            f"{settings.primary_latency_budget_ms}ms")
            return result, "primary"
        except Exception as exc:  # noqa: BLE001 - any primary failure => graceful fallback
            log.warning("Primary failed (%s); falling back to baseline.", exc)
            FALLBACK_TOTAL.inc()

    result = registry.predict(app.state.baseline, [text])[0]
    return result, "baseline"


def build_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="DriftGuard", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or get_settings()

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # Liveness: the process is up. Independent of model health.
        return HealthResponse(status="ok", app=app.state.settings.app_name)

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        # Model-agnostic readiness: baseline alone is enough to stay in rotation.
        tiers = []
        if getattr(app.state, "primary", None) is not None:
            tiers.append("primary")
        if getattr(app.state, "baseline", None) is not None:
            tiers.append("baseline")
        return ReadyResponse(ready=bool(tiers), servable_tiers=tiers)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        t0 = time.perf_counter()
        result, served_by = _serve(app, req.text)
        PREDICTIONS_TOTAL.labels(served_by=served_by).inc()
        return PredictResponse(
            label=result["label"],
            label_id=result["label_id"],
            confidence=result["confidence"],
            served_by=served_by,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            scores=result["scores"],
        )

    @app.get("/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        primary = getattr(app.state, "primary", None)
        baseline = app.state.baseline
        primary_version = str(primary["version"]) if primary else None
        return ModelInfo(
            active_tier=getattr(app.state, "active_tier", "baseline"),
            primary_version=primary_version,
            primary_available=primary is not None,
            baseline_version=str(baseline.get("version", "unknown")),
            primary_source=getattr(app.state, "primary_source", None),
        )

    return app


# Module-level app for `uvicorn driftguard.api.main:app`.
app = build_app()
