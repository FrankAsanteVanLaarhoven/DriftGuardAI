"""Smoke test for the drift-injection benchmark harness (tiny config for CI)."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

pytest.importorskip("pandas")
import drift_generators as gen  # noqa: E402

from driftguard import drift, textdrift  # noqa: E402
from driftguard.config import get_settings  # noqa: E402


def _pool():
    try:
        from driftguard.data import load_split
        return load_split("test", get_settings())
    except FileNotFoundError:
        pytest.skip("processed data not built (run `make data`)")


def test_generators_produce_windows_of_requested_size():
    pool = _pool()
    rng = random.Random(0)
    for fn in gen.GENERATORS.values():
        window = fn(pool, 50, rng)
        assert len(window) == 50
        assert all(isinstance(t, str) and t for t in window)


def test_no_false_positive_and_semantic_is_caught():
    settings = get_settings()
    pool = _pool()
    reference = textdrift.load_reference_texts(settings)
    ref_dist = drift.load_reference(settings)

    clean = gen.no_drift(pool, 200, random.Random(1))
    assert textdrift.composite_drift(clean, reference, ref_dist, settings)["drift"] is False

    shifted = gen.semantic_replace(pool, 200, random.Random(1), severity=0.7)
    assert textdrift.composite_drift(shifted, reference, ref_dist, settings)["drift"] is True


def test_severity_sweep_is_monotone_in_separability():
    _pool()  # ensure data exists / skip otherwise
    from eval_harness import sweep

    result = sweep("gradual_topic", severities=[0.1, 0.9], seeds=1, window=200)
    aucs = [r["mean_domain_auc"] for r in result["rows"]]
    # Heavier topic injection must be at least as separable as light injection.
    assert aucs[-1] > aucs[0]


def test_vocab_concept_drift_transform_is_deterministic():
    from closed_loop import vocab_drift

    text = "the central bank raised interest rates today"
    a = vocab_drift(text, p=0.7)
    b = vocab_drift(text, p=0.7)
    assert a == b                      # deterministic (hash-based, no per-run salt)
    assert len(a.split()) == len(text.split())
    assert "_v2" in a                  # some tokens acquired the new surface form
    assert vocab_drift(text, p=0.0) == text   # p=0 is a no-op
