"""The drift-detector interface — the pluggable half of the framework.

A ``DriftDetector`` learns a reference distribution (``fit``) and scores a current window
(``detect``), returning a ``DetectionResult``. Detectors are **modality-agnostic by
composition**: each is configured with a small extractor (a ``values_fn`` for PSI, an
sklearn ``estimator`` for the domain classifier) that adapts it to text, tabular rows,
embeddings, etc. The governance layer is deliberately *not* coupled to detectors —
detection triggers a retrain; governance decides promotion.

``reference`` / ``current`` are any indexable batch: a ``list`` (text), a numpy array
(embeddings), or a pandas frame (tabular). The helpers below index them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class DetectionResult:
    detector: str
    statistic: float          # higher ⇒ more drift
    threshold: float
    drift: bool
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DriftDetector(Protocol):
    """Any object with a ``name``, ``fit(reference)`` and ``detect(current)``."""

    name: str

    def fit(self, reference: Any) -> DriftDetector: ...

    def detect(self, current: Any) -> DetectionResult: ...


# --- uniform indexing over list / ndarray / DataFrame ----------------------- #
def batch_len(x: Any) -> int:
    return int(x.shape[0]) if hasattr(x, "shape") else len(x)


def batch_take(x: Any, idx: list[int]) -> Any:
    if hasattr(x, "iloc"):          # pandas DataFrame / Series
        return x.iloc[idx]
    if hasattr(x, "shape"):         # numpy array
        return x[idx]
    return [x[i] for i in idx]      # list


def batch_concat(a: Any, b: Any) -> Any:
    import numpy as np
    if isinstance(a, list):
        return a + b
    if hasattr(a, "iloc"):
        import pandas as pd
        return pd.concat([a, b], ignore_index=True)
    return np.vstack([a, b])


class CompositeDetector:
    """Combine detectors with an ``any`` (safety-first) or ``all`` rule."""

    def __init__(self, detectors: list[DriftDetector], rule: str = "any",
                 name: str = "composite"):
        self.detectors = detectors
        self.rule = rule
        self.name = name

    def fit(self, reference: Any) -> CompositeDetector:
        for d in self.detectors:
            d.fit(reference)
        return self

    def detect(self, current: Any) -> DetectionResult:
        results = [d.detect(current) for d in self.detectors]
        flags = [r.drift for r in results]
        drift = (bool(flags) and all(flags)) if self.rule == "all" else any(flags)
        return DetectionResult(
            detector=self.name,
            statistic=max((r.statistic for r in results), default=0.0),
            threshold=0.0,
            drift=drift,
            extra={
                "rule": self.rule,
                "triggered_by": [r.detector for r in results if r.drift],
                "signals": {r.detector: {"statistic": r.statistic,
                                         "threshold": r.threshold, "drift": r.drift}
                            for r in results},
            },
        )
