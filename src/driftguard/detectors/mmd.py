"""Maximum Mean Discrepancy drift detector on embedding batches (modality-agnostic).

Compares the distribution of reference vs current **embedding vectors**. Linear kernel
(default): ``MMD^2 = ||mean(ref) - mean(cur)||^2`` — exact and cheap for L2-normalized
embeddings. An RBF kernel is available for a full nonlinear two-sample test. Operates on
dense vectors, so it serves any embedding modality (sentence, image, audio); the caller
supplies the encoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from driftguard.detectors.base import DetectionResult


class MMDDetector:
    def __init__(self, threshold: float = 0.01, kernel: str = "linear",
                 gamma: float | None = None, name: str = "embedding_mmd"):
        self.threshold = threshold
        self.kernel = kernel
        self.gamma = gamma
        self.name = name
        self._reference: np.ndarray | None = None

    def fit(self, reference: Any) -> MMDDetector:
        self._reference = np.asarray(reference, dtype=float)
        return self

    def _rbf_gram_mean(self, a: np.ndarray, b: np.ndarray, gamma: float) -> float:
        sq = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
        return float(np.exp(-gamma * np.clip(sq, 0.0, None)).mean())

    def score(self, current: Any) -> float:
        if self._reference is None:
            raise RuntimeError("MMDDetector.detect called before fit().")
        x = self._reference
        y = np.asarray(current, dtype=float)
        if self.kernel == "linear":
            diff = x.mean(axis=0) - y.mean(axis=0)
            return float(diff @ diff)
        gamma = self.gamma if self.gamma is not None else 1.0 / x.shape[1]
        return (self._rbf_gram_mean(x, x, gamma) + self._rbf_gram_mean(y, y, gamma)
                - 2.0 * self._rbf_gram_mean(x, y, gamma))

    def detect(self, current: Any) -> DetectionResult:
        s = self.score(current)
        return DetectionResult(self.name, s, self.threshold, s > self.threshold,
                               extra={"kernel": self.kernel})
