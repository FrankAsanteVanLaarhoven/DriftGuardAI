# The drift-detector interface

Drift **detection** is the pluggable half of the framework (the other half, promotion
**governance**, is in [`GOVERNANCE.md`](GOVERNANCE.md) and stays decoupled — detection
triggers a retrain; governance decides promotion). Detectors live in
[`src/driftguard/detectors/`](../src/driftguard/detectors/).

## The contract

```python
@runtime_checkable
class DriftDetector(Protocol):
    name: str
    def fit(self, reference) -> "DriftDetector": ...   # learn the reference distribution
    def detect(self, current) -> DetectionResult: ...   # statistic + threshold + drift bool
```

`reference` / `current` are any indexable batch — a `list` (text), a numpy array
(embeddings), or a pandas frame (tabular). Detectors are **modality-agnostic by
composition**: each is adapted with a small extractor, not new detector code.

| Detector | Adapter it takes | Signal |
|----------|------------------|--------|
| `PSIDetector(values_fn, threshold, bins)` | `values_fn`: batch → 1-D numbers | Population Stability Index over any scalar (token count, a feature column, an embedding norm). |
| `DomainClassifierDetector(estimator, threshold)` | an sklearn `estimator` accepting the raw items | Cross-validated reference-vs-current ROC-AUC (Rabanser et al. 2019). |
| `MMDDetector(threshold, kernel)` | embedding vectors (linear or RBF kernel) | Maximum Mean Discrepancy between reference and current embedding batches. |
| `DescriptorKSDetector(features_fn, alpha)` | `features_fn`: batch → 2-D feature frame | Per-column two-sample K-S, Bonferroni-corrected to family-wise `alpha` (the KSDrift scheme). Absorbed from the head-to-head benchmark, where exactly this test beat the learned composite on descriptor-visible drift. |
| `CompositeDetector(detectors, rule)` | — | Combines detectors with `any` (safety-first) or `all`. |

## Same detector, three modalities

```python
from driftguard.detectors import PSIDetector, DomainClassifierDetector

# Text — PSI on token count
PSIDetector(values_fn=lambda xs: [len(t.split()) for t in xs])

# Tabular — PSI on a feature column + a gradient booster domain classifier
PSIDetector(values_fn=lambda df: df["hours-per-week"].to_numpy())
DomainClassifierDetector(HistGradientBoostingClassifier())      # see examples/tabular_adult.py

# Embeddings — the same domain classifier on dense vectors, zero new code
DomainClassifierDetector(LogisticRegression())                  # see examples/embedding_20news.py
```

Adding a modality is: supply a `values_fn` and/or an `estimator`. No detector subclass, no
duplicated PSI/AUC math. `tests/test_detectors.py` asserts protocol conformance and runs
all three modalities; `examples/tabular_adult.py` (tabular) and `examples/embedding_20news.py`
(MiniLM sentence embeddings) use these detectors in full, measured instances.

## The text service runs on these detectors

The production text path (`src/driftguard/textdrift.py`) is **not** a separate
implementation — `domain_classifier_drift` delegates to `DomainClassifierDetector` (TF-IDF +
logistic regression), `descriptor_ks_drift` delegates to `DescriptorKSDetector` over the
five shared text descriptors (token_count, char_count, mean_word_len, oov_rate,
non_alpha_rate — the same frame the head-to-head benchmark hands every comparator), and
`composite_drift` reads the frozen training reference through `PSIDetector.from_reference`,
which reproduces `drift.compute_psi` to the last decimal (guarded by
`test_psi_from_reference_matches_compute_psi_exactly`). One subtlety the K-S layer carries
from the benchmark protocol: the reference sample is split deterministically — half builds
the OOV vocabulary, the other half is the K-S reference — because a self-vocabulary makes
the reference's `oov_rate` identically 0 and every live window then trivially separates (a
guaranteed false alarm). The optional embedding-MMD signal (`embedding_mmd_drift`) likewise
delegates to `MMDDetector` — so `textdrift.py` carries **no local detector logic at all**.
There is one detector code path across text, tabular, and embeddings; the measured
per-detector scorecard (`benchmarks/results.json`) is `0.29 / 0.57 / 1.00 / 1.00` for
PSI / domain-classifier / descriptor-KS / composite.
