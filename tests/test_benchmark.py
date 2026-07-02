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


def test_new_families_are_detected():
    settings = get_settings()
    pool = _pool()
    reference = textdrift.load_reference_texts(settings)
    ref_dist = drift.load_reference(settings)

    noisy = gen.char_noise(pool, 200, random.Random(2), severity=0.2)
    assert textdrift.composite_drift(noisy, reference, ref_dist, settings)["drift"] is True

    dropped = gen.token_dropout(pool, 200, random.Random(2), severity=0.5)
    assert textdrift.composite_drift(dropped, reference, ref_dist, settings)["drift"] is True


def test_detector_scorecard_composite_is_a_superset_of_psi():
    _pool()  # ensure data / skip otherwise
    from eval_harness import run

    summary = run(seeds=1, window=150)
    sc = summary["detector_scorecard"]
    assert set(sc) == {"psi", "domain_classifier", "composite"}
    # Composite = PSI OR domain, so it never recalls less than either single detector.
    assert sc["composite"]["recall"] >= sc["psi"]["recall"]
    assert sc["composite"]["recall"] >= sc["domain_classifier"]["recall"]
    # The domain classifier carries real signal (catches semantic kinds PSI misses).
    assert sc["domain_classifier"]["recall"] > 0.0


def test_recovery_and_retention_ratio_formulas():
    from closed_loop import recovery_ratio, retention_ratio

    # Regains 0.0826 of a 0.0853 drift-induced loss -> ~0.968 recovered.
    assert abs(recovery_ratio(0.9170, 0.8344, 0.9197) - 0.9683) < 1e-3
    assert recovery_ratio(0.8344, 0.8344, 0.9197) == 0.0        # no recovery
    assert recovery_ratio(0.9, 0.9, 0.9) == 0.0                 # degenerate denom, no crash
    # Retention: candidate keeps 0.8519/0.9197 ~ 0.926 of the old-distribution score.
    assert retention_ratio(0.9197, 0.9197) == 1.0
    assert abs(retention_ratio(0.8519, 0.9197) - 0.9263) < 1e-3


def test_streaming_abrupt_change_is_detected_quickly():
    _pool()
    from streaming import run

    result = run(kind="semantic_replace", n_windows=6, change_point=2, window=150,
                 seeds=1, band=3, patterns=("abrupt",))
    row = result["patterns"][0]
    assert row["pattern"] == "abrupt"
    assert row["missed_detection_rate"] == 0.0
    assert row["detection_delay_windows"] is not None
    assert row["detection_delay_windows"] <= 2  # abrupt full-severity => fast alarm
