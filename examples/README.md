# Reference instances

The DriftGuard **governance framework** (`src/driftguard/governance.py`) is model-agnostic.
This directory holds instances that prove it — each one supplies its own data, model, and
drift detector, but imports the promotion gates and recovery/retention metrics **unchanged**.

| Instance | Data | Model | Where |
|----------|------|-------|-------|
| Text (primary) | `fancyzhx/ag_news` | TF-IDF + LogReg / DistilBERT | the main repo (`src/driftguard/`, service + benchmarks) |
| **Tabular** | OpenML Adult (Census Income) | HistGradientBoosting + LogReg | [`tabular_adult.py`](tabular_adult.py) |

## Tabular (Adult)

```bash
make example-tabular          # or: uv run python examples/tabular_adult.py
```

Trains a gradient-boosting income classifier, injects covariate drift on numeric features,
detects it with the shared `driftguard.detectors` (`PSIDetector` + `DomainClassifierDetector`,
the same classes text uses — see [`docs/DETECTORS.md`](../docs/DETECTORS.md)), retrains, and
feeds the resulting scores to `driftguard.governance` — the *same* `incumbent_gate`,
`promotion_gate`, `recovery_ratio`, and `retention_ratio` the text service uses. Writes
`results_tabular.json`.

Measured (macro-F1; clean holdout: baseline 0.783, primary 0.819):

| severity | detected | recovery ratio | retention ratio | dual gate |
|----------|----------|----------------|-----------------|-----------|
| 0.10     | True     | 0.780          | 0.936           | PASS      |
| 0.20     | True     | 0.765          | 0.861           | FAIL      |
| 0.40     | True     | 0.779          | 0.728           | FAIL      |

Same behaviour as the text instance: as covariate drift deepens, retention falls and the
`dual` gate flips from PASS to fail-closed — the framework's safety property, on a
completely different model family and data type. That is the "model-agnostic" claim as
runnable code, not prose.
