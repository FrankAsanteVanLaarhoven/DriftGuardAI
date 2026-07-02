"""Population Stability Index detector — a covariate-shift proxy on any scalar signal.

Configured with a ``values_fn`` that maps a batch to a 1-D array of numbers, so the same
detector serves text token-counts, a tabular feature column, an embedding norm, etc.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from driftguard.detectors.base import DetectionResult


class PSIDetector:
    def __init__(self, values_fn: Callable[[Any], Any], threshold: float = 0.2,
                 bins: int = 10, name: str = "psi"):
        self.values_fn = values_fn
        self.threshold = threshold
        self.bins = bins
        self.name = name
        self._edges: np.ndarray | None = None
        self._ref: np.ndarray | None = None

    @classmethod
    def from_reference(cls, reference: dict[str, Any], values_fn: Callable[[Any], Any],
                       threshold: float = 0.2, name: str = "psi") -> PSIDetector:
        """Build from a *frozen* reference (``bin_edges`` + ``reference_proportions``), e.g.
        ``driftguard.drift.build_reference`` output — a training-time distribution rather
        than a fit-on-the-current-sample one. Reproduces ``drift.compute_psi`` exactly."""
        det = cls(values_fn=values_fn, threshold=threshold, name=name)
        det._edges = np.asarray(reference["bin_edges"], dtype=float)
        det._ref = np.asarray(reference["reference_proportions"], dtype=float)
        return det

    def fit(self, reference: Any) -> PSIDetector:
        v = np.asarray(self.values_fn(reference), dtype=float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, self.bins + 1))) if v.size \
            else np.array([0.0, 1.0])
        if edges.size < 2:
            edges = np.array([v.min() - 1.0, v.max() + 1.0]) if v.size else np.array([0.0, 1.0])
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        self._edges = edges
        self._ref = np.histogram(v, edges)[0] / max(len(v), 1)
        return self

    def score(self, current: Any) -> float:
        if self._edges is None:
            raise RuntimeError("PSIDetector.detect called before fit().")
        eps = 1e-6
        v = np.asarray(self.values_fn(current), dtype=float)
        cur = np.histogram(v, self._edges)[0] / max(len(v), 1)
        # clip both sides (matches drift.compute_psi exactly for frozen-reference parity)
        exp = np.clip(self._ref, eps, None)
        cur = np.clip(cur, eps, None)
        return float(np.sum((cur - exp) * np.log(cur / exp)))

    def detect(self, current: Any) -> DetectionResult:
        s = self.score(current)
        return DetectionResult(self.name, s, self.threshold, s > self.threshold)
