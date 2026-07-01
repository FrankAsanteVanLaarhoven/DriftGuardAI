"""Text-aware drift tests — the multi-layer sensitivity proof.

The domain-classifier detector must catch a *semantic* shift (same length
distribution, disjoint vocabulary) that ``token_count`` PSI is blind to, and must
stay quiet on in-distribution data.
"""

import random

from driftguard import drift, textdrift


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
