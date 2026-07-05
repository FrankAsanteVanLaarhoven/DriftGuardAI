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


def test_semantic_rotation_preserves_every_descriptor_by_construction():
    pool = _pool()
    rng = random.Random(7)
    mapping = gen._rotation_mapping(pool, rng)
    assert mapping and all(k != v and len(k) == len(v) for k, v in mapping.items())

    # Apply at severity 1.0 and compare each text with its own rotation: token count
    # and every per-word character length must be identical — which pins token_count,
    # char_count, and mean_word_len exactly; alpha->alpha pins non_alpha_rate; the
    # frequency floor pins oov_rate.
    texts = pool["text"].tolist()[:100]
    rot_rng = random.Random(8)
    changed = 0
    for t in texts:
        words = t.split()
        rotated = [mapping[w.lower()] if w.isalpha() and w.lower() in mapping else w
                   for w in words]
        assert len(rotated) == len(words)
        assert [len(w) for w in rotated] == [len(w) for w in words]
        changed += sum(1 for a, b in zip(words, rotated, strict=True) if a != b)
    assert changed > 100  # the rotation genuinely rewrites a lot of content
    del rot_rng


def test_semantic_rotation_is_caught_by_reading_words_not_descriptors():
    settings = get_settings()
    pool = _pool()
    reference = textdrift.load_reference_texts(settings)
    ref_dist = drift.load_reference(settings)

    window = gen.semantic_rotation(pool, 300, random.Random(3), severity=0.7)
    result = textdrift.composite_drift(window, reference, ref_dist, settings)
    # The whole point: descriptors are preserved, so PSI and descriptor-KS abstain...
    assert result["signals"]["psi"]["drift"] is False
    assert result["signals"]["descriptor_ks"]["drift"] is False
    # ...and only the detector that reads the words catches it.
    assert result["signals"]["domain_classifier"]["drift"] is True
    assert result["drift"] is True


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
    assert set(sc) == {"psi", "domain_classifier", "descriptor_ks", "composite"}
    # Composite = any-rule union, so it never recalls less than any single detector.
    assert sc["composite"]["recall"] >= sc["psi"]["recall"]
    assert sc["composite"]["recall"] >= sc["domain_classifier"]["recall"]
    assert sc["composite"]["recall"] >= sc["descriptor_ks"]["recall"]
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


def test_head_to_head_descriptors_are_reference_safe():
    from head_to_head import DESCRIPTOR_COLUMNS, build_descriptors, reference_vocab

    vocab = reference_vocab(["the market rallied today", "central bank rates"])
    df = build_descriptors(["the market rallied", "unknownword bank"], vocab)
    assert list(df.columns) == list(DESCRIPTOR_COLUMNS)
    assert len(df) == 2
    # First text: every token in vocab -> oov 0; second: one of two tokens unseen.
    assert df["oov_rate"].iloc[0] == 0.0
    assert abs(df["oov_rate"].iloc[1] - 0.5) < 1e-9
    assert (df["token_count"] == [3, 2]).all()


def test_head_to_head_smoke_all_tools_score():
    pytest.importorskip("evidently")
    pytest.importorskip("nannyml")
    _pool()  # ensure data / skip otherwise
    from head_to_head import TOOLS, run

    summary = run(seeds=1, window=150, chunk_size=150, ref_size=600)
    assert set(summary["scorecard"]) == set(TOOLS)
    for tool in TOOLS:
        card = summary["scorecard"][tool]
        assert 0.0 <= card["f1"] <= 1.0
        assert summary["mean_latency_s"][tool] > 0.0
    # DriftGuard's composite must keep its zero-false-alarm property on no_drift.
    no_drift_row = next(r for r in summary["rows"] if r["kind"] == "no_drift")
    assert no_drift_row["driftguard"] == 0.0


def test_safe_promotion_oracle_labels():
    from driftguard.governance import safe_promotion_oracle

    # The measured p=0.7 full-data run: recovers on drift, retention 0.926 >= 0.90 floor.
    assert safe_promotion_oracle(0.9170, 0.8344, 0.8519, 0.9197) is True
    # Catastrophic forgetting (retention ~0.787, the p=0.9 regime) is unsafe even though
    # the candidate wins on the new distribution.
    assert safe_promotion_oracle(0.9300, 0.8000, 0.7238, 0.9197) is False
    # Regressing the incumbent on live (new-distribution) traffic is unsafe outright.
    assert safe_promotion_oracle(0.8000, 0.8344, 0.9100, 0.9197) is False
    # The floor is a parameter: relaxing it flips the forgetting verdict.
    assert safe_promotion_oracle(0.9300, 0.8000, 0.7238, 0.9197,
                                 retention_floor=0.75) is True


def test_promotion_decision_quality_scoring():
    from driftguard.governance import promotion_decision_quality

    q = promotion_decision_quality([
        (True, True), (True, True), (True, False),   # 3 promotions, 1 unsafe
        (False, True),                                # a safe candidate blocked
        (False, False),                               # an unsafe candidate blocked
    ])
    assert q["trials"] == 5 and q["promotions"] == 3
    assert q["unsafe_promotions"] == 1 and q["safe_candidates"] == 3
    assert abs(q["promotion_precision"] - 2 / 3) < 1e-9
    assert abs(q["promotion_recall"] - 2 / 3) < 1e-9
    assert abs(q["unsafe_promotion_rate"] - 1 / 5) < 1e-9

    # A gate that never promotes: precision undefined, zero unsafe rate, zero recall.
    q0 = promotion_decision_quality([(False, True), (False, False)])
    assert q0["promotion_precision"] is None
    assert q0["unsafe_promotion_rate"] == 0.0
    assert q0["promotion_recall"] == 0.0

    # No safe candidate exists: recall undefined, every promotion is unsafe.
    q1 = promotion_decision_quality([(True, False), (False, False)])
    assert q1["promotion_recall"] is None
    assert q1["promotion_precision"] == 0.0
    assert q1["unsafe_promotion_rate"] == 0.5


def test_slice_gate_fails_closed_when_aggregate_win_masks_slice_collapse():
    from driftguard.governance import slice_gate

    incumbent = {"World": 0.90, "Sports": 0.95, "Business": 0.88, "Sci/Tech": 0.91}
    # Macro average improves (+0.02 overall) but Business collapses by 0.12.
    candidate = {"World": 0.95, "Sports": 0.98, "Business": 0.76, "Sci/Tech": 0.97}
    result = slice_gate(candidate, incumbent, regression_floor=0.05)
    assert result.passed is False
    assert result.worst_slice == "Business"
    assert abs(result.worst_delta - (-0.12)) < 1e-9
    assert result.report["Business"]["retention"] < 0.9

    # Within-floor wobble passes.
    ok = slice_gate({k: v - 0.01 for k, v in incumbent.items()}, incumbent,
                    regression_floor=0.05)
    assert ok.passed is True

    # A slice missing from the candidate fails closed.
    partial = {k: v for k, v in candidate.items() if k != "World"}
    assert slice_gate(partial, incumbent).passed is False
    # No slices at all fails closed.
    assert slice_gate({}, {}).passed is False


def test_expected_calibration_error_and_gate():
    from driftguard.governance import calibration_gate, expected_calibration_error

    # Perfectly calibrated: 70% confidence, 70% correct in that bin.
    conf = [0.7] * 10
    corr = [True] * 7 + [False] * 3
    assert abs(expected_calibration_error(conf, corr)) < 1e-9

    # Overconfident: 90% confidence, 60% correct -> ECE = 0.3.
    conf = [0.9] * 10
    corr = [True] * 6 + [False] * 4
    assert abs(expected_calibration_error(conf, corr) - 0.3) < 1e-9

    # Empty input is defined (0.0), and the gate is fail-closed beyond tolerance.
    assert expected_calibration_error([], []) == 0.0
    assert calibration_gate(0.10, 0.05, tolerance=0.02).passed is False
    assert calibration_gate(0.06, 0.05, tolerance=0.02).passed is True


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
