"""The embedding reference instance must reuse the governance layer + shared detectors.

Offline: a deterministic HashingVectorizer stands in for the sentence-transformer encoder,
so the wiring (encode -> detect -> govern) is exercised with no torch and no network. The
real measured numbers come from running examples/embedding_20news.py on a GPU host.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

pytest.importorskip("sklearn")
import embedding_20news as emb  # noqa: E402

from driftguard import governance  # noqa: E402


def _hash_encode(texts):
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=64, alternate_sign=False, norm="l2")
    return hv.transform(list(texts)).toarray().astype(float)


def _synthetic_corpus():
    themes = {
        0: "space rocket orbit satellite launch nasa mars probe",
        1: "baseball pitch inning bat home run league bases",
        2: "election policy government senate vote law budget",
        3: "graphics pixel render shader gpu image texture mesh",
    }
    rng = np.random.default_rng(0)
    texts, labels = [], []
    for label, vocab in themes.items():
        words = vocab.split()
        for _ in range(40):
            texts.append(" ".join(rng.choice(words, size=12)))
            labels.append(label)
    return texts, np.asarray(labels)


def test_embedding_instance_reuses_the_governance_layer():
    # Identical framework objects, not re-implementations — the whole generalizability claim.
    assert emb.incumbent_gate is governance.incumbent_gate
    assert emb.promotion_gate is governance.promotion_gate
    assert emb.recovery_ratio is governance.recovery_ratio
    assert emb.retention_ratio is governance.retention_ratio


def test_offline_embedding_evaluate_detects_drift_and_governs():
    texts, y = _synthetic_corpus()
    r = emb.evaluate(texts, y, texts, y, encode=_hash_encode, severity=0.9, baseline_dim=8)

    assert r["instance"].startswith("embedding")
    assert r["detection"]["detected"] is True          # heavy contamination is caught
    g = r["governance"]
    assert 0.0 <= g["retention_ratio"] <= 1.5          # a well-formed ratio
    assert isinstance(g["dual_gate_passed"], bool)
