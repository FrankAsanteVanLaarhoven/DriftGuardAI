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
the drifted labelled data → baseline gate.

**Metric definitions.**
- **Recovery ratio** = `(candidate_on_drift − stale_on_drift) / (orig_clean − stale_on_drift)`
  — the fraction of the drift-induced accuracy loss regained on the *new* distribution.
  1.0 = fully restored to the pre-drift clean level; 0 = no recovery. (`orig_clean` is the
  pre-drift primary's score on the clean/fixed holdout.)
- **Retention ratio** = `candidate_on_fixed / stale_on_fixed` — the share of the *original*
  (pre-drift) distribution's performance kept after adapting. 1.0 = no forgetting; lower =
  more of the old distribution given up.
- **Time-to-recovery** = detect + retrain + evaluate wall time to a gate-ready candidate.

Measured run (p=0.7, window 600, full-data retrain):

- **Detected** by the domain classifier (AUC 1.0000) in **0.25 s**; PSI blind (0.0142).
- Retrain **23.0 s** → **time-to-recovery 24.0 s** (detect + retrain + evaluate).
- **Recovery ratio 0.968** — regains 96.8% of the drift-induced loss on the new
  distribution; **retention ratio 0.926** — keeps 92.6% of the old-distribution score.

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

## Recovery vs drift severity (`make recovery-sweep`)

Sweeping the vocabulary-drift fraction `p` across **3 seeds** — each retraining on a
40k-row sub-sample of the drifted data, so the figures carry genuine variation — traces
the recovery/retention trade-off (window 600) → `results_recovery_sweep.json`:

| p (vocab drift) | recovery ratio (mean±std) | retention ratio (mean±std) | TTR (s) | dual gate (pass frac) | safe frac |
|-----------------|---------------------------|----------------------------|---------|-----------------------|-----------|
| 0.30            | 0.352 ± 0.102             | **0.975 ± 0.003**          | 17.3    | 1.00                  | 1.00      |
| 0.50            | 0.726 ± 0.027             | 0.961 ± 0.003              | 16.3    | 1.00                  | 1.00      |
| 0.70            | 0.856 ± 0.011             | 0.923 ± 0.007              | 17.0    | 0.67                  | 1.00      |
| 0.90            | **0.930 ± 0.003**         | **0.787 ± 0.019**          | 17.7    | **0.00**              | **0.00**  |

(`safe frac` = fraction of trials the ground-truth safety oracle labels safe to promote —
see the next section.)

**Reading it.** Retention falls monotonically as drift deepens (0.975 → 0.787): heavier
drift forces the candidate to give up more of the old distribution. The **dual gate tracks
that trade-off** — every seed promotes at `p ≤ 0.50`, the gate sits right on the boundary
at `p=0.70` (2 of 3 seeds promote), and **every seed fails closed at `p=0.90`**, where
retention has collapsed to 0.787 and adaptation has become catastrophic forgetting.
Recovery ratio rises with severity but is small and *noisy* at `p=0.30` (0.352 ± 0.102):
light drift causes little accuracy loss, so the ratio divides two nearby numbers — the
system is healthy there (retention 0.975, gate passes), the *statistic* is just unstable.
(The sweep sub-samples 40k rows per seed to expose variance; the full-data single run
above recovers more — 0.968 at p=0.7.)

## Promotion decision quality (`make recovery-sweep`)

The sweep also scores each gate mode as a *decision-maker*. Every trial gets a
ground-truth safety label from `driftguard.governance.safe_promotion_oracle`: a
promotion is **safe** iff the candidate (a) is at least as good as the incumbent on the
**new** distribution and (b) keeps ≥ 0.90 of the incumbent's **original**-distribution
score (`--safety-retention-floor`). The oracle needs both models scored on both
distributions, so it exists only in the benchmark; the production gates approximate it
from committed baselines. Each gate's promote/block decisions over all 12 trials
(4 severities × 3 seeds) are then scored by
`driftguard.governance.promotion_decision_quality`:

- **promotion precision** — of the promotions, the fraction that were safe;
- **promotion recall** — of the genuinely safe candidates, the fraction promoted.
  A gate that blocks everything has perfect precision and zero recall, so the two are
  only meaningful together;
- **unsafe promotion rate** — unsafe promotions over all trials ("how often did it ship
  a regressive model").

Measured (window 600, 3 seeds, retention floor 0.90):

| gate mode | promotions | unsafe promotions | promotion precision | promotion recall | unsafe promotion rate |
|-----------|------------|-------------------|---------------------|------------------|-----------------------|
| fixed     | 2/12       | 0                 | 1.00                | 0.22             | 0.00                  |
| refreshed | 12/12      | 3                 | **0.75**            | 1.00             | **0.25**              |
| **dual**  | 8/12       | **0**             | **1.00**            | **0.89**         | **0.00**              |

**Reading it.** This is the governance argument as one table. The **fixed** gate never
ships an unsafe model but blocks 7 of the 9 safe recoveries (recall 0.22) — safety by
refusing to adapt. The **refreshed** gate promotes every candidate that recovered on the
new distribution, including all three catastrophic-forgetting candidates at `p=0.90` —
its 0.25 unsafe promotion rate is precisely the *recovery-is-not-safety* failure mode.
The **dual** gate ships **zero** unsafe models *and* promotes 8 of the 9 safe candidates;
its single miss is a `p=0.70` boundary seed it conservatively blocks. The cost of
fail-closed is now a measured number (0.11 of recall), not a claim.
