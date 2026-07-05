# The governance framework

DriftGuard is two things: a **model-agnostic framework** for *governed model adaptation
under distribution shift*, and a **text-classification reference implementation** that
validates it end to end. This document describes the framework — the reusable core that
does not depend on text, TF-IDF, or any particular model.

The framework lives in [`src/driftguard/governance.py`](../src/driftguard/governance.py)
and operates purely on **scalar quality scores** (a model's metric on a holdout). Macro-F1
on AG News is the reference instance; the same primitives govern the tabular and embedding
instances below — or a ranker or an LLM-eval score — unchanged.

## 1. Promotion gates — *should we replace what's in production?*

Given holdout scores, decide whether a candidate may be promoted. Fail-closed by default.

| Gate | Rule | Purpose |
|------|------|---------|
| `baseline_gate(candidate, baseline, margin)` | `candidate ≥ baseline + margin` | The floor: never ship a model worse than a committed baseline (the CI gate). |
| `incumbent_gate(candidate, baseline, incumbent, margin)` | `candidate ≥ max(baseline, incumbent) + margin` | Never *promote* a model worse than the one already serving — closes the "beats the baseline but downgrades production" gap. |
| `promotion_gate(..., mode="dual", regression_floor)` | adapt on the refreshed holdout **and** drop ≤ `regression_floor` on the fixed holdout | Drift-aware promotion: adapt to the new distribution without catastrophically forgetting the old one. |

The gates return a `GateResult` / `PromotionDecision` with a human-readable `reason`, so an
audit trail records *why* each promotion was allowed or refused.

## 2. Adaptation-safety metrics — *how well, and how safely, did it adapt?*

After a drift-triggered retrain, two ratios quantify the trade-off:

- **`recovery_ratio(candidate_new, stale_new, original)`** =
  `(candidate_new − stale_new) / (original − stale_new)` — the fraction of the
  drift-induced loss regained on the **new** distribution. 1.0 = fully restored.
- **`retention_ratio(candidate_original, stale_original)`** =
  `candidate_original / stale_original` — the share of **original**-distribution
  performance kept after adapting. 1.0 = no forgetting.

Recovery without retention is catastrophic forgetting; retention without recovery is
failure to adapt. The `dual` gate exists to require *both*.

### Below the aggregate: slices and calibration

Aggregate ratios can hide where the damage lands. Two further primitive families,
still operating only on scalars, expose it:

- **`slice_gate(candidate_slices, incumbent_slices, regression_floor)`** — fail-closed
  no-worse-than-incumbent applied to **every named slice** (per-class F1 in the text
  instance; per-segment AUC, per-cohort accuracy, a fairness partition — anything that
  yields one scalar per slice). An aggregate win must never mask a slice collapse; a
  missing slice fails closed. `slice_retention_report` gives the per-slice deltas and
  retention ratios behind the decision.
- **`expected_calibration_error(confidences, correct)`** + **`calibration_gate`** —
  top-label ECE, and a fail-closed check that the candidate's ECE does not exceed the
  incumbent's by more than a tolerance. Accuracy recovery bought with overconfidence is
  a regression for every consumer of the model's probabilities (thresholds, triage,
  review queues).

Measured on the reference loop (`make recovery`, vocab drift p=0.7): the aggregate
`dual` gate **passes** the retrained candidate — and the slice + calibration layer shows
exactly what that pass accepts: forgetting concentrates in two classes (Sci/Tech
−0.085, Business −0.081 on the fixed holdout, vs Sports −0.045), and the candidate's
old-distribution ECE is ~4× the incumbent's (0.019 → 0.070) while being *better*
calibrated than the stale model on the new distribution (0.020 vs 0.036). The
aggregate gate answers "may it ship?"; the slice/calibration report is the risk
statement of what shipping it means.

Every decision can be exported as a **versioned, tamper-evident
`PromotionDecisionRecord`** — required gates drive a fail-closed derived decision,
advisory gates ride along as the risk report, and the human gate is a first-class
outcome. See [`PROMOTION_DECISION.md`](PROMOTION_DECISION.md).

## 3. The measured trade-off

The reference implementation exercises the framework under controlled concept drift
(`make recovery-sweep`). As drift severity rises, recovery improves but retention falls —
and the `dual` gate promotes only while adaptation stays safe, failing closed once
retention collapses (see [`../benchmarks/README.md`](../benchmarks/README.md) and
[`../CASE_STUDY.md`](../CASE_STUDY.md) for the measured curve).

## 4. Instantiating the framework for another domain

To govern a different model/task you supply three things; the gates and metrics are reused
verbatim:

1. **A drift detector** — from the pluggable `driftguard.detectors` package
   ([`docs/DETECTORS.md`](DETECTORS.md)): configure `PSIDetector` / `DomainClassifierDetector`
   with a `values_fn` or an `estimator` for your modality (text, tabular, embeddings) —
   no new detector code.
2. **A holdout scorer** — a function returning a scalar quality metric on fixed and
   drift-refreshed holdouts.
3. **A retrain step** — produces a candidate from freshly labelled data.

Feed the resulting scores to `incumbent_gate` / `promotion_gate` and report
`recovery_ratio` / `retention_ratio`. Nothing in the governance layer changes.

## 5. Proven: two more instances (tabular + embeddings)

This is not aspirational — two further instances import `incumbent_gate`, `promotion_gate`,
`recovery_ratio`, and `retention_ratio` **unchanged** (tests assert they are the *same
objects* the text service uses).

**Tabular** — [`examples/tabular_adult.py`](../examples/tabular_adult.py)
(`make example-tabular`): **OpenML Adult** + **HistGradientBoosting**, PSI on features + a
domain classifier. As covariate drift deepens, retention falls and the `dual` gate flips
PASS → fail-closed, exactly as on text:

| severity | detected | recovery ratio | retention ratio | dual gate |
|----------|----------|----------------|-----------------|-----------|
| 0.10     | True     | 0.780          | 0.936           | PASS      |
| 0.20     | True     | 0.765          | 0.861           | FAIL      |
| 0.40     | True     | 0.779          | 0.728           | FAIL      |

**Embeddings** — [`examples/embedding_20news.py`](../examples/embedding_20news.py)
(`make example-embedding`): **20 Newsgroups** + logistic regression on **MiniLM** sentence
embeddings, with the shared detectors on the dense vectors. The drift is an
information-preserving rotation, so retraining *fully* recovers (recovery ≈ 1.0) — yet
retention still falls as the candidate forgets the clean distribution, and the gate flips:

| severity | detected | recovery ratio | retention ratio | dual gate |
|----------|----------|----------------|-----------------|-----------|
| 0.10     | True     | 1.000          | 0.993           | PASS      |
| 0.50     | True     | 1.000          | 0.937           | PASS      |
| 0.75     | True     | 1.000          | 0.606           | FAIL      |

The embedding case makes the point sharpest: **recovery alone is not safety**. A model can
perfectly relearn the drifted task and still be unpromotable because it has forgotten
production — which is exactly what the forgetting-aware gate catches. Same gates, same
metrics, three model families and data types.
