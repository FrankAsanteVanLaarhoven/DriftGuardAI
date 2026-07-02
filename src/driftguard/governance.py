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
