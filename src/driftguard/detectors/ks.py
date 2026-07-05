"""Per-feature two-sample Kolmogorov–Smirnov detector with Bonferroni correction.

The head-to-head benchmark (``benchmarks/head_to_head.py``) showed a corrected
classical K-S over a handful of good descriptor columns is the strongest window-level
detector on descriptor-visible drift — beating the learned composite at zero
false-positive cost. This detector absorbs that finding into the framework.

Configured with a ``features_fn`` mapping a batch to a 2-D feature frame (columns =
descriptors), so the same detector serves text descriptors, raw tabular columns, or
embedding summary stats. Drift ⇔ any column's K-S p-value clears ``alpha / n_columns``
(Bonferroni — the same scheme Alibi Detect's ``KSDrift`` applies).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from driftguard.detectors.base import DetectionResult


class DescriptorKSDetector:
    def __init__(self, features_fn: Callable[[Any], Any] | None = None,
                 alpha: float = 0.05, name: str = "descriptor_ks"):
        self.features_fn = features_fn or (lambda batch: batch)
        self.alpha = alpha
        self.name = name
        self._ref: np.ndarray | None = None
        self._columns: list[str] | None = None

    def _frame(self, batch: Any) -> tuple[np.ndarray, list[str]]:
        feats = self.features_fn(batch)
        if hasattr(feats, "columns"):        # pandas DataFrame
            return feats.to_numpy(dtype=float), [str(c) for c in feats.columns]
        arr = np.asarray(feats, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr, [f"f{i}" for i in range(arr.shape[1])]

    def fit(self, reference: Any) -> DescriptorKSDetector:
        self._ref, self._columns = self._frame(reference)
        return self

    def detect(self, current: Any) -> DetectionResult:
        if self._ref is None or self._columns is None:
            raise RuntimeError("DescriptorKSDetector.detect called before fit().")
        from scipy.stats import ks_2samp

        cur, _ = self._frame(current)
        pvals = {col: float(ks_2samp(self._ref[:, i], cur[:, i]).pvalue)
                 for i, col in enumerate(self._columns)}
        corrected_alpha = self.alpha / max(len(pvals), 1)
        p_min = min(pvals.values(), default=1.0)
        # DetectionResult convention: higher statistic ⇒ more drift.
        return DetectionResult(
            detector=self.name,
            statistic=1.0 - p_min,
            threshold=1.0 - corrected_alpha,
            drift=p_min < corrected_alpha,
            extra={"p_values": pvals, "alpha": self.alpha,
                   "corrected_alpha": corrected_alpha,
                   "n_reference": int(self._ref.shape[0]),
                   "n_current": int(cur.shape[0])},
        )
