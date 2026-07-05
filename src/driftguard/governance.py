"""Model-agnostic governance layer: promotion gates + adaptation-safety metrics.

This is the *framework* core of DriftGuard, separate from the text-classification
*reference implementation*. Every primitive here operates purely on **scalar quality
scores** (a model's metric on a holdout) and returns a promotion decision or a safety
ratio. None of them know the model type, the task, or the features — macro-F1 on
AG News text is simply the reference instance wired up in this repo. The same gate and
the same metrics govern a tabular classifier, a ranker, or an LLM-eval score unchanged.

The contract:

* **Promotion gates** decide *whether to replace the production model* with a candidate,
  given their holdout scores:
    - ``baseline_gate``   — never ship a model worse than a committed floor (the CI gate).
    - ``incumbent_gate``  — never *promote* a model worse than the one already serving
                            (``max(baseline, incumbent) + margin``); the headline rule.
    - ``promotion_gate``  — drift-aware, with a ``dual`` mode that requires the candidate
                            to adapt to the new distribution *and* not catastrophically
                            forget the old one.
* **Adaptation-safety metrics** quantify *how well and how safely* a retrained model
  adapted:
    - ``recovery_ratio``  — fraction of the drift-induced loss regained on the new data.
    - ``retention_ratio`` — share of original-distribution performance kept.

See ``docs/GOVERNANCE.md`` for the full write-up and how the text reference implementation
instantiates this contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from driftguard.registry import (
    GateResult,
    PromotionDecision,
    baseline_gate,
    effective_promotion_bar,
    incumbent_gate,
    promotion_gate,
)

__all__ = [
    "GateResult",
    "PromotionDecision",
    "baseline_gate",
    "incumbent_gate",
    "promotion_gate",
    "effective_promotion_bar",
    "recovery_ratio",
    "retention_ratio",
    "safe_promotion_oracle",
    "promotion_decision_quality",
]


def recovery_ratio(candidate_new_score: float, stale_new_score: float,
                   original_score: float) -> float:
    """Fraction of the drift-induced loss a retrained candidate regains on the *new*
    distribution: ``(candidate_new - stale_new) / (original - stale_new)``.

    1.0 = fully restored to the pre-drift level; 0.0 = no recovery. Model-agnostic —
    each ``*_score`` is any scalar holdout metric (macro-F1 in the reference impl).
    """
    denom = original_score - stale_new_score
    return (candidate_new_score - stale_new_score) / denom if denom > 1e-9 else 0.0


def retention_ratio(candidate_original_score: float, stale_original_score: float) -> float:
    """Share of the *original*-distribution performance kept after adapting:
    ``candidate_original / stale_original``. 1.0 = no forgetting; lower = more given up.
    """
    if stale_original_score > 1e-9:
        return candidate_original_score / stale_original_score
    return 0.0


def safe_promotion_oracle(candidate_new_score: float, incumbent_new_score: float,
                          candidate_original_score: float, incumbent_original_score: float,
                          retention_floor: float = 0.90) -> bool:
    """Ground-truth safety label for a promotion decision (benchmark scoring only).

    A promotion is *safe* iff the candidate (a) is at least as good as the incumbent on
    the **new** (live) distribution and (b) retains at least ``retention_floor`` of the
    incumbent's **original**-distribution score. This oracle needs both models scored on
    both distributions, so it is only computable in a controlled benchmark; the
    production gates (``incumbent_gate``, ``promotion_gate``) approximate it from
    committed baselines. Scoring a gate's decisions against this oracle is what
    ``promotion_decision_quality`` does.
    """
    return (candidate_new_score >= incumbent_new_score
            and retention_ratio(candidate_original_score, incumbent_original_score)
            >= retention_floor)


def promotion_decision_quality(
        decisions: Sequence[tuple[bool, bool]]) -> dict[str, float | int | None]:
    """Score a gate's promote/block decisions against ground-truth safety labels.

    ``decisions`` holds one ``(promoted, safe)`` pair per trial, where ``safe`` comes
    from ``safe_promotion_oracle``. Returns:

    - ``promotion_precision`` — of the promotions, the fraction that were safe
      (``None`` if the gate never promoted);
    - ``promotion_recall`` — of the genuinely safe candidates, the fraction promoted
      (``None`` if no candidate was safe). A gate that blocks everything has perfect
      precision and zero recall — report both or the number is meaningless;
    - ``unsafe_promotion_rate`` — unsafe promotions over **all** trials ("how often did
      it ship a regressive model");
    - raw counts (``trials``, ``promotions``, ``unsafe_promotions``, ``safe_candidates``).
    """
    trials = len(decisions)
    promotions = sum(1 for promoted, _ in decisions if promoted)
    safe_promotions = sum(1 for promoted, safe in decisions if promoted and safe)
    unsafe_promotions = promotions - safe_promotions
    safe_candidates = sum(1 for _, safe in decisions if safe)
    return {
        "trials": trials,
        "promotions": promotions,
        "unsafe_promotions": unsafe_promotions,
        "safe_candidates": safe_candidates,
        "promotion_precision": safe_promotions / promotions if promotions else None,
        "promotion_recall": safe_promotions / safe_candidates if safe_candidates else None,
        "unsafe_promotion_rate": unsafe_promotions / trials if trials else 0.0,
    }
