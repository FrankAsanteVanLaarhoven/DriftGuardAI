"""Unit tests for configuration and pure helpers.

Expanded in later phases with the baseline gate and PSI tests.
"""

from driftguard import __version__, drift, registry
from driftguard.config import AG_NEWS_LABELS, get_settings


def test_version_is_set():
    assert __version__


def test_settings_defaults_are_sane():
    s = get_settings()
    assert s.app_name == "driftguard"
    assert 0.0 < s.val_fraction < 1.0
    assert s.psi_threshold > 0
    assert s.random_seed == 42


def test_ag_news_label_order_is_fixed():
    # This order is a hard contract with the dataset and the served label ids.
    assert AG_NEWS_LABELS == ("World", "Sports", "Business", "Sci/Tech")
    assert len(AG_NEWS_LABELS) == 4


def test_baseline_gate_promotes_a_better_candidate():
    gate = registry.baseline_gate(candidate_macro_f1=0.92, baseline_macro_f1=0.90, margin=0.0)
    assert gate.passed


def test_baseline_gate_rejects_a_worse_candidate():
    # Fail-closed: a regression must never pass, even by a hair.
    gate = registry.baseline_gate(candidate_macro_f1=0.89, baseline_macro_f1=0.90, margin=0.0)
    assert not gate.passed


def test_baseline_gate_respects_margin():
    # Beating the baseline is not enough if it does not clear the margin.
    gate = registry.baseline_gate(candidate_macro_f1=0.905, baseline_macro_f1=0.90, margin=0.01)
    assert not gate.passed


def test_psi_flags_a_shifted_distribution():
    import random

    rng = random.Random(0)
    reference_texts = ["w " * rng.randint(20, 60) for _ in range(1000)]
    stable = ["w " * rng.randint(20, 60) for _ in range(300)]
    shifted = ["w " * rng.randint(1, 5) for _ in range(300)]  # much shorter -> large shift

    ref = drift.build_reference(reference_texts, bins=10)
    assert drift.compute_psi(stable, ref)["psi"] < 0.1
    assert drift.compute_psi(shifted, ref)["psi"] > 0.2
