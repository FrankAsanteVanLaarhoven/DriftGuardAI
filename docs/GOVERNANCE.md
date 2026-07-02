# The governance framework

DriftGuard is two things: a **model-agnostic framework** for *governed model adaptation
under distribution shift*, and a **text-classification reference implementation** that
validates it end to end. This document describes the framework — the reusable core that
does not depend on text, TF-IDF, or any particular model.

The framework lives in [`src/driftguard/governance.py`](../src/driftguard/governance.py)
and operates purely on **scalar quality scores** (a model's metric on a holdout). Macro-F1
on AG News is the reference instance; the same primitives govern a tabular classifier, a
ranker, or an LLM-eval score unchanged.

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

## 5. Proven: a second instance (tabular)

This is not aspirational — [`examples/tabular_adult.py`](../examples/tabular_adult.py)
(`make example-tabular`) is a second instance on **OpenML Adult** with a
**HistGradientBoosting** model and a tabular drift detector (PSI on features + a domain
classifier). It imports `incumbent_gate`, `promotion_gate`, `recovery_ratio`, and
`retention_ratio` **unchanged** — a test asserts they are the *same objects* the text
service uses. As covariate drift deepens, retention falls and the `dual` gate flips from
PASS to fail-closed, exactly as on text:

| severity | detected | recovery ratio | retention ratio | dual gate |
|----------|----------|----------------|-----------------|-----------|
| 0.10     | True     | 0.780          | 0.936           | PASS      |
| 0.20     | True     | 0.765          | 0.861           | FAIL      |
| 0.40     | True     | 0.779          | 0.728           | FAIL      |

Same gates, same metrics, same safety behaviour — a different model family and data type.
