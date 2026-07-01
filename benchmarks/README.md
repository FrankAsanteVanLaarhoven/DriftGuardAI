# DriftGuard drift-injection benchmark

Turns "the domain classifier catches what PSI misses" into numbers. It applies
controlled, seeded drift generators to the AG News test pool and scores the composite
detector (PSI + domain-classifier) on each.

```bash
make benchmark            # 5 seeds, window 600
uv run python benchmarks/eval_harness.py --seeds 10 --window 800
```

- `drift_generators.py` — Garcia-style generators: `no_drift` (FPR control),
  `length_truncate` (token-count shift), `class_prior_shift`, `adjective_swap`,
  `semantic_replace`, `gradual_topic`.
- `eval_harness.py` — runs each kind across seeds, records detection rate, which
  detector fired, and mean PSI / domain-AUC; writes `results.json` and a Markdown
  table.

## Latest measured run (5 seeds, window 600)

Mean detection on genuine drift = **0.80**; false-positive rate on `no_drift` = **0.00**.

| drift kind        | detection | mean PSI | mean domain AUC | PSI fired | domain fired |
|-------------------|-----------|----------|-----------------|-----------|--------------|
| no_drift          | 0.00      | 0.0130   | 0.5215          | 0/5       | 0/5          |
| length_truncate   | 1.00      | 12.5169  | 0.9736          | 5/5       | 5/5          |
| class_prior_shift | 1.00      | 0.0535   | 0.7959          | 0/5       | 5/5          |
| adjective_swap    | 1.00      | 0.0130   | 0.9978          | 0/5       | 5/5          |
| semantic_replace  | 1.00      | 0.0130   | 1.0000          | 0/5       | 5/5          |
| gradual_topic     | 0.00      | 0.0130   | 0.7182          | 0/5       | 0/5          |

**Reading it.** PSI only fires on the length shift; every *semantic* category is
carried by the domain classifier — exactly the multi-layer value. `no_drift` produces
zero false positives. The one miss, `gradual_topic` at 40% injection, sits just under
the 0.75 AUC threshold (0.7182): partial/gradual drift is the genuinely hard case, and
it is caught at higher injection severity or a lower threshold — at some false-positive
cost. This trade-off is exactly what the benchmark exists to quantify.

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
