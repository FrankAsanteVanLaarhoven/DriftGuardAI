"""The pluggable detector interface: protocol conformance, text/tabular reuse, and a
third modality (dense embeddings) working with zero new detector code."""

import numpy as np
import pandas as pd

from driftguard.detectors import (
    CompositeDetector,
    DetectionResult,
    DomainClassifierDetector,
    DriftDetector,
    PSIDetector,
)


def _texts(n_tokens: int, count: int, rng) -> list[str]:
    return [" ".join(["w"] * int(rng.integers(max(1, n_tokens - 2), n_tokens + 3)))
            for _ in range(count)]


def test_detectors_satisfy_the_protocol():
    from sklearn.linear_model import LogisticRegression

    assert isinstance(PSIDetector(values_fn=lambda x: [0.0]), DriftDetector)
    assert isinstance(DomainClassifierDetector(LogisticRegression()), DriftDetector)
    assert isinstance(CompositeDetector([]), DriftDetector)


def test_psi_detector_reproduces_text_token_count_behaviour():
    rng = np.random.default_rng(0)
    ref, same, longer = _texts(6, 300, rng), _texts(6, 200, rng), _texts(30, 200, rng)
    det = PSIDetector(values_fn=lambda xs: [len(t.split()) for t in xs], threshold=0.2).fit(ref)
    assert det.detect(same).drift is False
    result = det.detect(longer)
    assert result.drift is True
    assert isinstance(result, DetectionResult) and result.statistic > 0.2


def test_domain_classifier_on_embeddings_is_a_free_third_modality():
    # Dense vectors (an embedding modality) — same detector, no new code.
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, (200, 8))
    same = rng.normal(0.0, 1.0, (200, 8))
    shifted = rng.normal(3.0, 1.0, (200, 8))
    det = DomainClassifierDetector(LogisticRegression(max_iter=200), threshold=0.75).fit(ref)
    assert det.detect(same).drift is False        # AUC ≈ 0.5, indistinguishable
    assert det.detect(shifted).drift is True       # separable ⇒ drift


def test_composite_any_vs_all_rules():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0.0, 1.0, 300)})
    cur = pd.DataFrame({"x": rng.normal(2.0, 1.0, 300)})   # a clear shift on x
    fires = PSIDetector(values_fn=lambda df: df["x"].to_numpy(), threshold=0.1, name="a")
    quiet = PSIDetector(values_fn=lambda df: df["x"].to_numpy(), threshold=1e9, name="b")

    any_c = CompositeDetector([fires, quiet], rule="any").fit(ref)
    all_c = CompositeDetector([fires, quiet], rule="all").fit(ref)
    assert any_c.detect(cur).drift is True
    assert all_c.detect(cur).drift is False
    assert any_c.detect(cur).extra["triggered_by"] == ["a"]
