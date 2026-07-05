"""Text-aware drift tests — the multi-layer sensitivity proof.

The domain-classifier detector must catch a *semantic* shift (same length
distribution, disjoint vocabulary) that ``token_count`` PSI is blind to, and must
stay quiet on in-distribution data.
"""

import random

from driftguard import drift, textdrift
from driftguard.config import Settings


def _gen(vocab, n, rng):
    return [" ".join(rng.choice(vocab) for _ in range(rng.randint(20, 60))) for _ in range(n)]


def _corpora():
    rng = random.Random(0)
    vocab_a = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
    vocab_b = "one two three four five six seven eight nine ten".split()  # disjoint
    reference = _gen(vocab_a, 400, rng)
    in_dist = _gen(vocab_a, 250, rng)     # same vocab + lengths
    semantic = _gen(vocab_b, 250, rng)    # same lengths, disjoint vocabulary
    return reference, in_dist, semantic


def test_domain_classifier_quiet_on_in_distribution():
    reference, in_dist, _ = _corpora()
    dom = textdrift.domain_classifier_drift(reference, in_dist, seed=0, threshold=0.75)
    assert dom["auc"] < 0.75            # indistinguishable
    assert dom["drift"] is False


def test_domain_classifier_catches_semantic_shift_that_psi_misses():
    reference, _, semantic = _corpora()

    # PSI on token_count is blind: same length distribution => no drift.
    ref_dist = drift.build_reference(reference, bins=10)
    psi = drift.compute_psi(semantic, ref_dist)["psi"]
    assert drift.classify_psi(psi, 0.2) != "drift"

    # The domain classifier reads the words and catches it.
    dom = textdrift.domain_classifier_drift(reference, semantic, seed=0, threshold=0.75)
    assert dom["auc"] > 0.9
    assert dom["drift"] is True


def test_composite_fires_when_either_signal_fires():
    reference, in_dist, semantic = _corpora()
    ref_dist = drift.build_reference(reference, bins=10)

    quiet = textdrift.composite_drift(in_dist, reference, ref_dist)
    assert quiet["drift"] is False

    caught = textdrift.composite_drift(semantic, reference, ref_dist)
    assert caught["drift"] is True
    assert "domain_classifier" in caught["triggered_by"]
    assert "psi" not in caught["triggered_by"]   # PSI alone would have missed it


def test_descriptor_ks_quiet_in_distribution_and_catches_shift():
    reference, in_dist, semantic = _corpora()

    quiet = textdrift.descriptor_ks_drift(reference, in_dist)
    assert quiet["drift"] is False

    # Disjoint vocabulary moves oov_rate and word-length descriptors decisively.
    caught = textdrift.descriptor_ks_drift(reference, semantic)
    assert caught["drift"] is True
    assert caught["p_min"] < caught["corrected_alpha"]


def test_descriptor_ks_vocab_split_prevents_self_vocab_false_alarm():
    # The reference sample scored by K-S (odd half) must carry a *nonzero* oov_rate
    # against the vocab half — the self-vocab degenerate case measured in the
    # head-to-head protocol. Same-vocab windows then stay in-family.
    rng = random.Random(1)
    wide_vocab = [f"w{i}" for i in range(500)]
    reference = _gen(wide_vocab, 400, rng)   # each half sees ~different token subsets
    in_dist = _gen(wide_vocab, 250, rng)
    result = textdrift.descriptor_ks_drift(reference, in_dist)
    assert result["drift"] is False


def test_composite_includes_descriptor_ks_signal():
    reference, _, semantic = _corpora()
    ref_dist = drift.build_reference(reference, bins=10)
    caught = textdrift.composite_drift(semantic, reference, ref_dist)
    assert "descriptor_ks" in caught["signals"]
    assert caught["signals"]["descriptor_ks"]["drift"] is True
    assert "descriptor_ks" in caught["triggered_by"]


def test_composite_rule_all_requires_every_signal():
    # With rule="all", the classifier-only semantic catch is NOT declared drift,
    # because PSI abstains. Confirms the combination rule is configurable.
    reference, _, semantic = _corpora()
    ref_dist = drift.build_reference(reference, bins=10)
    strict = Settings(drift_composite_rule="all")

    result = textdrift.composite_drift(semantic, reference, ref_dist, settings=strict)
    assert result["rule"] == "all"
    assert result["signals"]["domain_classifier"]["drift"] is True
    assert result["signals"]["psi"]["drift"] is False
    assert result["drift"] is False   # "all" gate not satisfied
