"""Model construction, evaluation, the baseline gate, and the fallback loaders.

A *model bundle* is a plain dict persisted with joblib so it stays portable across
library versions::

    {"pipeline": <sklearn Pipeline>, "labels": (...), "kind": "primary"|"baseline",
     "version": "<str>", "trained_at": "<iso8601>", "metrics": {...}}

The serving layer only needs :func:`load_baseline`, :func:`load_primary`,
:func:`canary_selftest`, and :func:`predict`. Training uses the builders, the
evaluator, the gate, and :func:`log_and_register`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from driftguard.config import AG_NEWS_LABELS, Settings, get_settings

log = logging.getLogger("driftguard.registry")

# A known sample whose label is unambiguous; used for the startup canary self-test.
CANARY_TEXT = "The government held elections and world leaders met to discuss the treaty."
CANARY_LABEL = "World"


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def build_baseline_pipeline() -> Pipeline:
    """Tiny, dependency-light, always-loadable fallback model."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 1), min_df=5,
                                      sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=300, C=1.0)),
        ]
    )


def build_primary_pipeline() -> Pipeline:
    """Larger, higher-capacity primary model (still fast, CPU-friendly)."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=4.0)),
        ]
    )


# --------------------------------------------------------------------------- #
# Evaluation + the baseline gate
# --------------------------------------------------------------------------- #
def evaluate(pipeline: Pipeline, texts: list[str], labels: list[int]) -> dict[str, float]:
    preds = pipeline.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


@dataclass(frozen=True)
class GateResult:
    passed: bool
    candidate_macro_f1: float
    baseline_macro_f1: float
    margin: float
    reason: str


def baseline_gate(candidate_macro_f1: float, baseline_macro_f1: float,
                  margin: float = 0.0) -> GateResult:
    """Fail-closed promotion gate: candidate must beat baseline by >= margin."""
    threshold = baseline_macro_f1 + margin
    passed = candidate_macro_f1 >= threshold
    reason = (
        f"candidate macro-F1 {candidate_macro_f1:.4f} "
        f"{'>=' if passed else '<'} baseline {baseline_macro_f1:.4f} + margin {margin:.4f} "
        f"(= {threshold:.4f})"
    )
    return GateResult(passed, candidate_macro_f1, baseline_macro_f1, margin, reason)


def effective_promotion_bar(baseline_macro_f1: float,
                            incumbent_macro_f1: float | None = None) -> tuple[float, str]:
    """The score a candidate must clear to be promoted: never below the committed
    baseline *and* never below the model currently in production.

    Returns ``(bar, source)`` where ``source`` names which model set the bar.
    """
    if incumbent_macro_f1 is not None and incumbent_macro_f1 > baseline_macro_f1:
        return incumbent_macro_f1, "incumbent primary"
    return baseline_macro_f1, "baseline"


def incumbent_gate(candidate_macro_f1: float, baseline_macro_f1: float,
                   incumbent_macro_f1: float | None = None,
                   margin: float = 0.0) -> GateResult:
    """No-worse-than-incumbent promotion gate.

    Extends :func:`baseline_gate` so a candidate is promoted only if it clears
    ``max(baseline, incumbent_primary) + margin``. This closes the downgrade gap where a
    candidate beats the tiny baseline but is *worse* than the model already serving —
    e.g. a slow transformer that scores below the incumbent linear primary. With no
    incumbent (fresh deploy) it degrades exactly to :func:`baseline_gate`.

    Model-agnostic: the arguments are any scalar holdout metric (macro-F1 in this
    reference implementation). Re-exported by the :mod:`driftguard.governance` framework.
    """
    bar, source = effective_promotion_bar(baseline_macro_f1, incumbent_macro_f1)
    threshold = bar + margin
    passed = candidate_macro_f1 >= threshold
    inc_txt = f"{incumbent_macro_f1:.4f}" if incumbent_macro_f1 is not None else "n/a"
    reason = (
        f"candidate macro-F1 {candidate_macro_f1:.4f} {'>=' if passed else '<'} "
        f"max(baseline {baseline_macro_f1:.4f}, incumbent {inc_txt}) + margin {margin:.4f} "
        f"(= {threshold:.4f}; bar set by {source})"
    )
    return GateResult(passed, candidate_macro_f1, bar, margin, reason)


def current_primary_macro_f1(settings: Settings | None = None) -> float | None:
    """Macro-F1 of the primary currently in production, or ``None`` if none is recorded.

    Reads ``artifacts/metrics.json`` (written by the last promoted primary) — cheap and
    avoids deserialising a possibly-heavy served model just to read its score.
    """
    settings = settings or get_settings()
    path = settings.metrics_path
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["macro_f1"])
    except Exception:  # noqa: BLE001 - a missing/garbled metrics file just means "no incumbent"
        return None


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    mode: str
    reason: str


def promotion_gate(candidate_fixed_f1: float, baseline_fixed_f1: float,
                   candidate_refreshed_f1: float | None = None,
                   baseline_refreshed_f1: float | None = None,
                   margin: float = 0.0, mode: str = "fixed",
                   regression_floor: float = 0.05) -> PromotionDecision:
    """Drift-aware promotion decision.

    * ``fixed``     — the classic gate on the frozen holdout (unchanged behaviour).
    * ``refreshed`` — beat the baseline on a current-distribution holdout.
    * ``dual``      — adapt to the new distribution (beat baseline on the refreshed
                      holdout by ``margin``) **and** avoid catastrophic forgetting (drop
                      no more than ``regression_floor`` on the fixed holdout).

    ``dual`` is the safe resolution to the concept-drift recovery block: it promotes
    genuine recovery without letting the model forget the old distribution wholesale.
    """
    if mode == "fixed":
        g = baseline_gate(candidate_fixed_f1, baseline_fixed_f1, margin)
        return PromotionDecision(g.passed, mode, g.reason)

    if candidate_refreshed_f1 is None or baseline_refreshed_f1 is None:
        raise ValueError(f"mode={mode!r} needs candidate_refreshed_f1 and baseline_refreshed_f1")

    if mode == "refreshed":
        g = baseline_gate(candidate_refreshed_f1, baseline_refreshed_f1, margin)
        return PromotionDecision(g.passed, mode, f"refreshed holdout — {g.reason}")

    if mode == "dual":
        adapts = candidate_refreshed_f1 >= baseline_refreshed_f1 + margin
        floor_threshold = baseline_fixed_f1 - regression_floor
        no_forget = candidate_fixed_f1 >= floor_threshold
        passed = adapts and no_forget
        reason = (
            f"refreshed {candidate_refreshed_f1:.4f} {'>=' if adapts else '<'} "
            f"{baseline_refreshed_f1:.4f}+{margin:.4f}; "
            f"fixed-floor {candidate_fixed_f1:.4f} {'>=' if no_forget else '<'} "
            f"{floor_threshold:.4f} (baseline {baseline_fixed_f1:.4f} - "
            f"floor {regression_floor:.4f})"
        )
        return PromotionDecision(passed, mode, reason)

    raise ValueError(f"unknown gate mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def make_bundle(pipeline: Pipeline, kind: str, metrics: dict[str, float],
                version: str) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "labels": AG_NEWS_LABELS,
        "kind": kind,
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Inference helper
# --------------------------------------------------------------------------- #
def predict(bundle: dict[str, Any], texts: list[str]) -> list[dict[str, Any]]:
    pipe: Pipeline = bundle["pipeline"]
    labels = bundle["labels"]
    proba = pipe.predict_proba(texts)
    out = []
    for row in proba:
        idx = int(np.argmax(row))
        out.append({
            "label": labels[idx],
            "label_id": idx,
            "confidence": float(row[idx]),
            "scores": {labels[i]: float(p) for i, p in enumerate(row)},
        })
    return out


def canary_selftest(bundle: dict[str, Any]) -> bool:
    """Predict a known sample; return True only if the model produces a valid label."""
    try:
        result = predict(bundle, [CANARY_TEXT])[0]
        return result["label"] in bundle["labels"]
    except Exception as exc:  # noqa: BLE001 - self-test must never raise
        log.warning("Canary self-test failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Fallback loaders (used by the serving layer)
# --------------------------------------------------------------------------- #
def load_baseline(settings: Settings | None = None) -> dict[str, Any]:
    """Load the committed baseline. Raises on failure — the process must not start
    without at least one serving model."""
    settings = settings or get_settings()
    bundle = load_bundle(settings.baseline_path)
    if not canary_selftest(bundle):
        raise RuntimeError("Baseline model failed its canary self-test.")
    return bundle


def load_primary(settings: Settings | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort primary load. Returns (bundle, source) or (None, None).

    Order: MLflow registry (if a ``models:/`` URI is configured and reachable),
    otherwise the local pointer file (``models/primary_pointer`` → a joblib path).
    Any failure is swallowed and reported so the service can degrade to baseline.
    """
    settings = settings or get_settings()
    uri = settings.primary_model_uri.strip()

    if uri.startswith("models:/"):
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            pipe = mlflow.sklearn.load_model(uri)
            bundle = make_bundle(pipe, "primary", {}, version=uri)
            if canary_selftest(bundle):
                return bundle, f"mlflow:{uri}"
            log.warning("Primary from %s failed canary self-test.", uri)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load primary from MLflow (%s): %s", uri, exc)

    pointer = settings.primary_pointer_path
    if pointer.exists():
        try:
            target = Path(pointer.read_text().strip())
            if not target.is_absolute():
                target = pointer.parent.parent / target
            bundle = load_bundle(target)
            if canary_selftest(bundle):
                return bundle, f"pointer:{target}"
            log.warning("Primary from pointer %s failed canary self-test.", target)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load primary from pointer %s: %s", pointer, exc)

    return None, None


# --------------------------------------------------------------------------- #
# MLflow tracking + registry
# --------------------------------------------------------------------------- #
def log_and_register(pipeline: Pipeline, params: dict[str, Any], metrics: dict[str, float],
                     settings: Settings | None = None, promote: bool = False) -> dict[str, str]:
    """Log a run to MLflow, register the model, and (if promoted) set the
    ``production`` alias. Returns identifiers for the report."""
    settings = settings or get_settings()
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=settings.registered_model_name,
        )

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = _latest_version(client, settings.registered_model_name)
    result = {"run_id": run.info.run_id, "model_version": version,
              "model_uri": info.model_uri, "promoted": "false"}

    if promote and version:
        client.set_registered_model_alias(settings.registered_model_name, "production", version)
        # Best-effort classic stage for tooling that still reads stages.
        try:
            client.transition_model_version_stage(
                settings.registered_model_name, version, "Production",
                archive_existing_versions=True,
            )
        except Exception as exc:  # noqa: BLE001
            log.info("Stage transition skipped (alias set): %s", exc)
        result["promoted"] = "true"
    return result


def _latest_version(client, name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return ""
    return max(versions, key=lambda v: int(v.version)).version


def production_version(settings: Settings | None = None) -> str | None:
    """Return the model version currently behind the ``production`` alias, if any."""
    settings = settings or get_settings()
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    try:
        mv = client.get_model_version_by_alias(settings.registered_model_name, "production")
        return mv.version
    except Exception:  # noqa: BLE001
        return None


def promote_version(version: str, settings: Settings | None = None) -> None:
    """Human-gated promotion: point the ``production`` alias at ``version``."""
    settings = settings or get_settings()
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    client.set_registered_model_alias(settings.registered_model_name, "production", version)
    try:
        client.transition_model_version_stage(
            settings.registered_model_name, version, "Production",
            archive_existing_versions=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.info("Stage transition skipped (alias set): %s", exc)


def latest_version(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    return _latest_version(client, settings.registered_model_name)
