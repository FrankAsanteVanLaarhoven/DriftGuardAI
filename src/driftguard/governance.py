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
* **Slice-level safety** extends no-worse-than-incumbent below the aggregate:
    - ``slice_gate``  — fail-closed if ANY slice regresses more than the floor; an
                        aggregate win must never mask a slice collapse.
    - ``slice_retention_report`` — per-slice deltas and retention ratios.
* **Calibration safety** — a candidate that recovers accuracy but becomes overconfident
  is unsafe for any downstream consumer of its probabilities:
    - ``expected_calibration_error`` — top-label ECE from confidences + correctness.
    - ``calibration_gate`` — candidate ECE may not exceed incumbent ECE + tolerance.

See ``docs/GOVERNANCE.md`` for the full write-up and how the text reference implementation
instantiates this contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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
    "SliceGateResult",
    "slice_gate",
    "slice_retention_report",
    "CalibrationGateResult",
    "expected_calibration_error",
    "calibration_gate",
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


@dataclass(frozen=True)
class SliceGateResult:
    passed: bool
    worst_slice: str | None
    worst_delta: float
    regression_floor: float
    reason: str
    report: dict[str, dict[str, float]] = field(default_factory=dict)


def slice_retention_report(candidate_slices: Mapping[str, float],
                           incumbent_slices: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """Per-slice comparison: candidate vs incumbent score, delta, and retention ratio.

    Slices are any named partitions with a scalar score each — per-class F1 in the
    text reference instance, but equally per-segment AUC, per-region accuracy, or a
    fairness cohort metric. Model-agnostic like everything in this module.
    """
    return {
        name: {
            "candidate": float(candidate_slices.get(name, 0.0)),
            "incumbent": float(inc),
            "delta": float(candidate_slices.get(name, 0.0)) - float(inc),
            "retention": retention_ratio(float(candidate_slices.get(name, 0.0)), float(inc)),
        }
        for name, inc in incumbent_slices.items()
    }


def slice_gate(candidate_slices: Mapping[str, float],
               incumbent_slices: Mapping[str, float],
               regression_floor: float = 0.05) -> SliceGateResult:
    """Fail-closed slice-level no-worse-than-incumbent: EVERY incumbent slice must hold
    ``candidate >= incumbent - regression_floor``. An aggregate improvement must never
    mask a slice collapse. A slice missing from the candidate scores fails closed.
    """
    report = slice_retention_report(candidate_slices, incumbent_slices)
    if not report:
        return SliceGateResult(False, None, 0.0, regression_floor,
                               "no incumbent slices supplied — fail closed", {})
    worst_slice = min(report, key=lambda name: report[name]["delta"])
    worst_delta = report[worst_slice]["delta"]
    missing = [name for name in incumbent_slices if name not in candidate_slices]
    passed = not missing and worst_delta >= -regression_floor
    if missing:
        reason = f"candidate missing slice scores for {missing} — fail closed"
    else:
        reason = (f"worst slice '{worst_slice}' delta {worst_delta:+.4f} "
                  f"{'>=' if passed else '<'} -floor ({-regression_floor:.4f})")
    return SliceGateResult(passed, worst_slice, float(worst_delta), regression_floor,
                           reason, report)


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               bins: int = 10) -> float:
    """Top-label ECE: bin predictions by confidence; ECE = Σ (n_b/N)·|acc_b − conf_b|.

    ``confidences`` are the winning-class probabilities, ``correct`` whether the
    prediction matched the label. 0 = perfectly calibrated. Model-agnostic — any
    classifier that emits a winning-class probability can be scored.
    """
    import numpy as np

    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece)


@dataclass(frozen=True)
class CalibrationGateResult:
    passed: bool
    candidate_ece: float
    incumbent_ece: float
    tolerance: float
    reason: str


def calibration_gate(candidate_ece: float, incumbent_ece: float,
                     tolerance: float = 0.02) -> CalibrationGateResult:
    """Fail-closed calibration check: the candidate's ECE may exceed the incumbent's by
    at most ``tolerance``. Accuracy recovery bought with overconfidence is a regression
    for every consumer of the model's probabilities (thresholds, triage, review queues).
    """
    passed = candidate_ece <= incumbent_ece + tolerance
    reason = (f"candidate ECE {candidate_ece:.4f} "
              f"{'<=' if passed else '>'} incumbent {incumbent_ece:.4f} "
              f"+ tolerance {tolerance:.4f}")
    return CalibrationGateResult(passed, float(candidate_ece), float(incumbent_ece),
                                 tolerance, reason)


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
