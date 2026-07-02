"""Domain-classifier drift detector (Rabanser et al. 2019), modality-agnostic.

Train a classifier to tell reference (0) from current (1); cross-validated ROC-AUC ≈ 0.5
means indistinguishable (no drift), → 1.0 means easily separable (drift). The ``estimator``
carries all modality knowledge: a ``Pipeline(TfidfVectorizer, LogReg)`` for text, a plain
gradient booster for a tabular numeric matrix or an embedding array.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from driftguard.detectors.base import DetectionResult, batch_concat, batch_len, batch_take


class DomainClassifierDetector:
    def __init__(self, estimator: Any, threshold: float = 0.75, seed: int = 42,
                 splits: int = 3, name: str = "domain_classifier"):
        self.estimator = estimator
        self.threshold = threshold
        self.seed = seed
        self.splits = splits
        self.name = name
        self._reference: Any = None

    def fit(self, reference: Any) -> DomainClassifierDetector:
        self._reference = reference
        return self

    def _subsample(self, batch: Any, n: int, rng) -> Any:
        # Subsample only when larger than n, preserving the draw order — this reproduces
        # the text detector's balancing exactly (for benchmark parity).
        if batch_len(batch) <= n:
            return batch
        idx = rng.choice(batch_len(batch), n, replace=False).tolist()
        return batch_take(batch, idx)

    def _balanced(self, current: Any):
        rng = np.random.default_rng(self.seed)
        n = min(batch_len(self._reference), batch_len(current))
        return self._subsample(self._reference, n, rng), self._subsample(current, n, rng), n

    def _auc_and_n(self, current: Any) -> tuple[float, int]:
        if self._reference is None:
            raise RuntimeError("DomainClassifierDetector.detect called before fit().")
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        ref, cur, n = self._balanced(current)
        x = batch_concat(ref, cur)
        y = np.array([0] * n + [1] * n)
        splits = max(2, min(self.splits, n))
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.seed)
        auc = float(np.mean(cross_val_score(clone(self.estimator), x, y, cv=cv,
                                            scoring="roc_auc")))
        return auc, n

    def score(self, current: Any) -> float:
        return self._auc_and_n(current)[0]

    def detect(self, current: Any) -> DetectionResult:
        auc, n = self._auc_and_n(current)
        return DetectionResult(self.name, auc, self.threshold, auc >= self.threshold,
                               extra={"n_reference": n, "n_current": n})
