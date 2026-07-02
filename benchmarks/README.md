# DriftGuard drift-injection benchmark

Turns "the domain classifier catches what PSI misses" into numbers. It applies
controlled, seeded drift generators to the AG News test pool and scores the composite
detector (PSI + domain-classifier) on each.

```bash
make benchmark            # 5 seeds, window 600 (per-kind table + per-detector scorecard)
make benchmark-sweep      # gradual_topic severity -> detection boundary
make benchmark-stream     # streaming detection latency across temporal patterns
uv run python benchmarks/eval_harness.py --seeds 10 --window 800
```

- `drift_generators.py` — Garcia-style generators: `no_drift` (FPR control),
  `length_truncate` (token-count shift), `class_prior_shift`, `adjective_swap`,
  `semantic_replace`, `gradual_topic`, `char_noise` (typo/OCR corruption),
  `token_dropout` (degraded/truncated input).
- `eval_harness.py` — runs each kind across seeds, records detection rate, which
  detector fired, mean PSI / domain-AUC, **and a per-detector precision/recall/F1/FPR
  scorecard**; writes `results.json` and a Markdown table.
- `streaming.py` — builds a temporal stream with a change point (abrupt / gradual /
  incremental / recurring, per Gama et al. 2014) and measures **detection latency**,
  missed-detection rate, and pre-change false-alarm rate; writes `results_streaming.json`.

## Latest measured run (5 seeds, window 600)

Mean detection on genuine drift = **0.71**; false-positive rate on `no_drift` = **0.00**.

| drift kind        | detection | mean PSI | mean domain AUC | PSI fired | domain fired |
|-------------------|-----------|----------|-----------------|-----------|--------------|
| no_drift          | 0.00      | 0.0130   | 0.5215          | 0/5       | 0/5          |
| length_truncate   | 1.00      | 12.5169  | 0.9736          | 5/5       | 5/5          |
| class_prior_shift | 1.00      | 0.0535   | 0.7959          | 0/5       | 5/5          |
| adjective_swap    | 1.00      | 0.0130   | 0.9978          | 0/5       | 5/5          |
| semantic_replace  | 1.00      | 0.0130   | 1.0000          | 0/5       | 5/5          |
| gradual_topic     | 0.00      | 0.0130   | 0.7182          | 0/5       | 0/5          |
| char_noise        | 0.00      | 0.0145   | 0.7198          | 0/5       | 0/5          |
| token_dropout     | 1.00      | 3.3188   | 0.6928          | 5/5       | 0/5          |

**Reading it.** PSI fires only on token-count shifts (`length_truncate`, `token_dropout`);
every *semantic* category is carried by the domain classifier — exactly the multi-layer
value. `no_drift` produces zero false positives. Two misses sit just under the 0.75 AUC
gate: `gradual_topic` at 40% injection (0.7182) and `char_noise` at its mild default
severity 0.1 (0.7198) — partial/mild drift is the genuinely hard case, caught at higher
severity or a lower threshold, at some false-positive cost. That trade-off is exactly
what the benchmark quantifies.

### Per-detector scorecard (ground truth = `is_drift`, over every kind × seed)

| detector          | precision | recall | F1   | FPR  |
|-------------------|-----------|--------|------|------|
| psi               | 1.00      | 0.29   | 0.44 | 0.00 |
| domain_classifier | 1.00      | 0.57   | 0.73 | 0.00 |
| **composite**     | 1.00      | **0.71** | **0.83** | 0.00 |

The multi-layer detector more than **doubles recall over PSI alone (0.29 → 0.71) at zero
false-positive cost**. Both single detectors are perfectly precise (no false alarms);
the composite `any`-rule simply unions their coverage. This is the headline number for
"the domain classifier catches what PSI misses", now quantified.

## Detection boundary: `gradual_topic` severity sweep

`make benchmark-sweep` (5 seeds, window 600) traces detection vs. injection fraction:

| severity | detection | mean domain AUC | mean PSI |
|----------|-----------|-----------------|----------|
| 0.10     | 0.00      | 0.5668          | 0.0168   |
| 0.20     | 0.00      | 0.6059          | 0.0168   |
| 0.30     | 0.00      | 0.6724          | 0.0168   |
| 0.40     | 0.00      | 0.7067          | 0.0168   |
| **0.50** | **1.00**  | **0.7639**      | 0.0168   |
| 0.60     | 1.00      | 0.8062          | 0.0168   |
| 0.70     | 1.00      | 0.8577          | 0.0168   |
| 0.80     | 1.00      | 0.9005          | 0.0168   |
| 0.90     | 1.00      | 0.9511          | 0.0168   |

The domain-classifier AUC rises monotonically with injection fraction and crosses the
0.75 gate at **~50% injection** — the detection boundary for gradual topic drift at the
default threshold. PSI stays flat at 0.0168 across the whole sweep: token-count is
structurally blind to topic injection that preserves length. Lowering the AUC gate
shifts the boundary left (earlier detection) at a false-positive cost — a deliberate,
now-quantified operating-point choice.

## Streaming detection latency (`make benchmark-stream`)

`streaming.py` runs the composite detector over a stream of windows with a change point
at window 6, across the four canonical temporal drift patterns (Gama et al. 2014).
Measured run (`semantic_replace`, 16 windows, window 400, 3 seeds):

| pattern     | detection delay (windows) | missed rate | false-alarm rate (pre-change) | post-change detection |
|-------------|---------------------------|-------------|-------------------------------|-----------------------|
| abrupt      | 0.00                      | 0.00        | 0.000                         | 1.00                  |
| gradual     | 1.33                      | 0.00        | 0.000                         | 0.77                  |
| incremental | 0.00                      | 0.00        | 0.000                         | 1.00                  |
| recurring   | 0.00                      | 0.00        | 0.000                         | 0.60                  |

**Reading it.** The detector fires **within one window** of an abrupt or incremental
change, lags by ~1.3 windows on gradual drift (it must accumulate enough drifted traffic
to separate), and **never raises a pre-change false alarm** on any pattern. `recurring`'s
0.60 post-change detection is by design — drift comes and goes in blocks, so only the
drifted blocks should (and do) fire. Delay is the streaming metric the static per-window
benchmark cannot express.

## Closed-loop recovery (`make recovery`)

`closed_loop.py` measures the full self-healing loop under a *vocabulary concept drift*
(a fraction `p` of tokens acquire a new surface form): detect → retrain a candidate on
the drifted labelled data → baseline gate. Measured run (p=0.7, window 600):

- **Detected** by the domain classifier (AUC 1.0000) in **0.245 s**; PSI blind (0.0142).
- Retrain **24.4 s** → detection→decision wall time **24.6 s**.

| macro-F1            | stale primary | retrained candidate |
|---------------------|---------------|---------------------|
| on DRIFTED holdout  | 0.8344        | 0.9170 (Δ **+0.083**) |
| on FIXED holdout    | 0.9197        | 0.8519              |

- Gate on **FIXED** holdout: **FAIL** (0.8519 < 0.8956).
- Gate on **drift-refreshed** holdout: **PASS** (0.9170 ≥ 0.7993).

**Governance finding.** Retraining recovers accuracy on the new distribution
(+0.083 macro-F1), but the candidate scores *worse* on the stale fixed holdout — so a
gate that still evaluates on the fixed holdout **rejects the recovered model**.

**Resolution — the drift-aware `dual` gate** (`DRIFTGUARD_GATE_HOLDOUT_MODE=dual`):

| gate mode | decision |
|-----------|----------|
| fixed     | FAIL (0.8519 < 0.8956) |
| refreshed | PASS (0.9170 ≥ 0.7993) |
| **dual**  | **PASS** — adapts (0.9170 ≥ 0.7993) and clears the fixed-holdout forgetting floor (0.8519 ≥ 0.8956 − 0.05) |

`dual` promotes genuine recovery yet still fails closed on catastrophic forgetting.
Safety intent preserved, recovery unblocked.
