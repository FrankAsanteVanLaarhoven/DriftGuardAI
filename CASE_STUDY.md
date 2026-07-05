# DriftGuard case study — measured results

This is the **text-classification reference implementation** validating DriftGuard's
model-agnostic governance framework (see [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md)). The
gates and adaptation-safety metrics measured below are task-independent; AG News is the
instance that exercises them end to end. Read the promotion-gate, recovery, and retention
numbers as evidence that the *framework* behaves correctly — text is where it is proven.

All numbers below were **measured on this repository** with the committed code and
`fancyzhx/ag_news`, seed 42 (DistilBERT on GPU; everything else CPU). They are reproducible
with `make train`, `make test`, `make train-transformer`, `make recovery-sweep`,
`make stack`, and `make demo`. Nothing here is estimated.

## Dataset (fixed, seeded, DVC-versioned)

| Split | Rows    | Notes                              |
|-------|---------|------------------------------------|
| train | 108,000 | 90% of the HF train split          |
| val   | 12,000  | 10% stratified holdout (seed 42)   |
| test  | 7,600   | official test split (frozen holdout)|

`dvc repro` is reproducible: a forced rebuild produced byte-identical output md5s.

## Model quality (frozen test holdout)

| Model                    | Accuracy | Macro-F1 |
|--------------------------|----------|----------|
| Baseline (fallback)      | 0.8958   | 0.8956   |
| Primary (linear TF-IDF)  | 0.9199   | 0.9197   |
| Primary (DistilBERT)     | 0.9413   | 0.9412   |

**Baseline gate:** the linear primary clears the gate (0.9197 ≥ 0.8956 + 0.0). With an
impossible margin (0.5) the same candidate is **blocked** (0.9197 < 1.3956) and the
`production` alias stayed on the previous version — fail-closed confirmed.

**No-worse-than-incumbent gate (DistilBERT promotion):** DistilBERT
(`distilbert-base-uncased`, 3 epochs, seed 42, full 108k train rows, RTX 4080 SUPER)
scored **macro-F1 0.9412** (accuracy 0.9413) on the frozen holdout — measured on GPU and
written to [`artifacts/metrics_transformer.json`](artifacts/metrics_transformer.json);
reproduce end to end with `make train-transformer`. The promotion gate required it to beat
`max(baseline 0.8956, incumbent primary 0.9197)`, so it was gated against **0.9197**, not
the baseline — it cleared it (0.9412 ≥ 0.9197, +0.0215) and was promoted. A weaker
candidate scoring between 0.8956 and 0.9197 would beat the baseline yet be **rejected** as
a downgrade. The promoted bundle loads, passes its canary self-test, and serves
predictions; if `torch` is absent or the latency budget is breached, the service degrades
to the linear baseline unchanged (fallback contract intact).

## Operational resilience (the fallback contract)

Measured against the running service and in the test suite:

- Removing `models/primary_pointer` from a live service → next `/predict` returned
  **HTTP 200** with `"served_by":"baseline"`; `/model-info` flipped to
  `active_tier=baseline, primary_available=false`; `driftguard_model_tier{tier="baseline"}`
  went to **1.0**. The service never returned a 5xx.
- A **corrupt** primary (invalid joblib) fails the canary self-test at load → baseline
  serves.
- A **zero latency budget** forces every primary call over budget → baseline serves,
  `driftguard_primary_latency_breach_total` increments.
- A **hanging model registry** (DNS blackhole to MLflow) is bounded by the
  `primary_load_timeout_s` deadline (20 s default) → startup completes degraded on the
  baseline instead of blocking. Found the hard way: the kind canary drill (below) showed
  the MLflow client's minutes-long retry backoff blowing the startup-probe budget and
  CrashLooping every pod — an outage the in-process fallback could never see. Now a
  chaos test (`test_hanging_registry_degrades_within_deadline`).
- Primary predict latency on CPU (local): sub-3 ms per request (0.7–2.8 ms observed).

### Canary auto-rollback drill (kind, measured 2026-07-05)

Deployed via the Helm chart on a kind cluster (Prometheus scraping at 15 s, guard
CronJob every minute), with the canary's candidate made unloadable — see
`deploy/helm/README.md` for the mechanism and full timeline:

- Broken-canary release rolled at 02:21:34Z → canary Ready but **degraded to its
  in-pod baseline** (every request it served stayed HTTP 200).
- Breach first observable in Prometheus 02:22:10Z → guard **scaled the canary to 0 at
  02:23:00Z** with an audit annotation: **50 s from breach-visible to rollback**,
  86 s including the entire broken deploy.
- In-cluster traffic probe (1 req/s): **1248/1248 HTTP 200** from the moment the fixed
  image rolled, with zero non-200 through the broken-canary deploy and rollback.

**Test suite: 16 passed** — unit + integration + **5 fallback/chaos tests**.

## Drift detection (PSI over token_count, threshold 0.2)

| Sample              | PSI      | Verdict |
|---------------------|----------|---------|
| in-distribution     | 0.0137   | stable  |
| shifted (truncated) | 12.5169  | drift → non-zero exit, retrain triggered |

The drift→retrain pipeline detected the shift, retrained a candidate, ran the
fail-closed gate, held at the human gate, and only then promoted.

### Multi-layer detection

Measured with `python -m driftguard.textdrift` — a text-aware domain classifier
(cross-validated ROC-AUC, threshold 0.75) run alongside PSI (the composite has since
gained a third, descriptor-KS layer — see the benchmark section below):

| Sample                | PSI (token_count) | Domain-classifier AUC | Verdict |
|-----------------------|-------------------|-----------------------|---------|
| in-distribution       | 0.0137 (no)       | 0.4945 (no)           | no drift |
| token shift           | 12.5169 (yes)     | 0.9836 (yes)          | drift (both) |
| **semantic shift**    | **0.0137 (MISS)** | **1.0000 (CATCH)**    | drift via classifier only |

The semantic-shift sample has an **identical length distribution** to
in-distribution data — so PSI scores it exactly 0.0137 and misses it entirely — yet
the domain classifier separates it perfectly (AUC 1.0). This is the concrete,
reproducible case for multi-layer, text-aware drift detection.

### Drift-injection benchmark (`make benchmark`, 5 seeds, window 600)

Controlled generators (Garcia-style) scored by the composite detector — now three
layers: PSI + domain-classifier + **descriptor-KS** (Bonferroni-corrected two-sample
K-S over five text descriptors, absorbed from the head-to-head benchmark where exactly
that classical test beat the two-layer composite; see `benchmarks/README.md`). **Mean
detection on genuine drift = 1.00; false-positive rate on `no_drift` = 0.00.**

| drift kind        | detection | mean PSI | mean domain AUC | PSI fired | domain fired | KS fired |
|-------------------|-----------|----------|-----------------|-----------|--------------|----------|
| no_drift          | 0.00      | 0.0130   | 0.5215          | 0/5       | 0/5          | 0/5      |
| length_truncate   | 1.00      | 12.5169  | 0.9736          | 5/5       | 5/5          | 5/5      |
| class_prior_shift | 1.00      | 0.0535   | 0.7959          | 0/5       | 5/5          | 5/5      |
| adjective_swap    | 1.00      | 0.0130   | 0.9978          | 0/5       | 5/5          | 5/5      |
| semantic_replace  | 1.00      | 0.0130   | 1.0000          | 0/5       | 5/5          | 5/5      |
| gradual_topic     | **1.00**  | 0.0130   | 0.7182          | 0/5       | 0/5          | **5/5**  |
| char_noise        | **1.00**  | 0.0145   | 0.7198          | 0/5       | 0/5          | **5/5**  |
| token_dropout     | 1.00      | 3.3188   | 0.6928          | 5/5       | 0/5          | 5/5      |

PSI fires only on token-count shifts; the domain classifier carries the strong semantic
categories; the descriptor-KS layer closes the two previously documented misses
(`gradual_topic` at 40% injection and `char_noise` at mild severity 0.1, both just under
the 0.75 AUC gate). Zero false positives on in-distribution windows.

**Per-detector scorecard (over every kind × seed), source: `benchmarks/results.json`:**

| detector          | precision | recall   | F1       | FPR  |
|-------------------|-----------|----------|----------|------|
| psi               | 1.00      | 0.29     | 0.44     | 0.00 |
| domain_classifier | 1.00      | 0.57     | 0.73     | 0.00 |
| descriptor_ks     | 1.00      | 1.00     | 1.00     | 0.00 |
| **composite**     | 1.00      | **1.00** | **1.00** | 0.00 |

Before the K-S layer the composite measured **0.71 recall / 0.83 F1** — the benchmark
history is kept in `benchmarks/README.md`. On this suite the corrected K-S alone matches
the composite (every generator moves a surface descriptor); the domain classifier stays
because it reads raw text and covers semantic shifts that preserve all five descriptors,
where a descriptor K-S is structurally blind.

**Streaming detection latency (`make benchmark-stream`), source:
`benchmarks/results_streaming.json`.** Composite detector over a stream with a change
point at window 6, across the Gama et al. (2014) taxonomy — **zero pre-change false
alarms on every pattern**:

| pattern     | detection delay (windows) | missed rate | pre-change false alarms | post-change detection |
|-------------|---------------------------|-------------|-------------------------|-----------------------|
| abrupt      | 0.00                      | 0.00        | 0.000                   | 1.00                  |
| gradual     | 1.33                      | 0.00        | 0.000                   | 0.77                  |
| incremental | 0.00                      | 0.00        | 0.000                   | 1.00                  |
| recurring   | 0.00                      | 0.00        | 0.000                   | 0.60                  |

Fires within one window of an abrupt/incremental change, lags ~1.3 windows on gradual
drift, and never false-alarms before the change point.

**Detection boundary (`make benchmark-sweep`, gradual_topic, 5 seeds).** With the
descriptor-KS layer the composite detects gradual topic drift at **every injection
fraction down to 10%** — foreign-vocabulary docs move `oov_rate` decisively at any
severity. The domain-classifier-only boundary is still visible in the AUC column: it
rises monotonically and crosses the 0.75 gate at ~50% injection (0.40→0.7067,
0.50→0.7639, 0.90→0.9511); PSI stays flat at 0.0168 across the whole range. That curve
remains the operating-point reference for deployments running without the K-S layer.

### Closed-loop recovery (`make recovery`, vocabulary concept drift p=0.7)

Full self-healing loop, measured end to end: detect → retrain candidate on drifted
labelled data → baseline gate.

- Detected by the domain classifier (AUC 1.0000) in **0.25 s**; PSI blind (0.0142).
- Retrain **23.0 s** → **time-to-recovery 24.0 s** (detect + retrain + evaluate).
- **Recovery ratio 0.968** (regains 96.8% of the drift-induced loss on the new
  distribution); **retention ratio 0.926** (keeps 92.6% of the old-distribution score).

| macro-F1            | stale primary | retrained candidate |
|---------------------|---------------|---------------------|
| on DRIFTED holdout  | 0.8344        | 0.9170 (Δ **+0.083**) |
| on FIXED holdout    | 0.9197        | 0.8519              |

Gate on the **fixed** holdout FAILS (0.8519 < 0.8956); gate on the **drift-refreshed**
holdout PASSES (0.9170 ≥ 0.7993).

**Governance finding (measured, not hypothesised).** Retraining recovers +0.083 macro-F1
on the new distribution, but the candidate is *worse* on the stale fixed holdout — so a
fail-closed gate that still scores against the fixed holdout **blocks the recovery**.

**Resolution — the drift-aware `dual` gate.** `registry.promotion_gate(mode="dual")`
requires the candidate to (a) beat the baseline on a *refreshed* (current-distribution)
holdout **and** (b) drop no more than `gate_regression_floor` (default 0.05) on the fixed
holdout — i.e. adapt without catastrophic forgetting. On the same scenario:

| gate mode  | decision | why |
|------------|----------|-----|
| fixed      | **FAIL** | 0.8519 < 0.8956 (blocks recovery) |
| refreshed  | PASS     | 0.9170 ≥ 0.7993 (adapts) |
| **dual**   | **PASS** | adapts (0.9170 ≥ 0.7993) *and* fixed-floor OK (0.8519 ≥ 0.8456) |

The `dual` gate promotes genuine recovery while still failing closed on catastrophic
forgetting (a candidate that scored, say, 0.40 on the fixed holdout would be blocked by
the floor). This resolves the tension between "never promote a regression" and "adapt
under concept drift" — safety intent preserved, recovery unblocked. Unit tests cover all
three modes and both failure directions.

**Recovery vs drift severity (`make recovery-sweep`, 3 seeds, 40k retrain sub-sample),
source: `benchmarks/results_recovery_sweep.json`.** Sweeping the drift fraction `p` traces
the adaptation/forgetting trade-off with variation — and shows the dual gate tracking it:

| p (vocab drift) | recovery ratio (mean±std) | retention ratio (mean±std) | TTR (s) | dual gate (pass frac) |
|-----------------|---------------------------|----------------------------|---------|-----------------------|
| 0.30            | 0.352 ± 0.102             | 0.975 ± 0.003              | 15.3    | 1.00                  |
| 0.50            | 0.726 ± 0.027             | 0.961 ± 0.003              | 14.9    | 1.00                  |
| 0.70            | 0.856 ± 0.011             | 0.923 ± 0.007              | 16.1    | 0.67                  |
| 0.90            | 0.930 ± 0.003             | **0.787 ± 0.019**          | 16.4    | **0.00**              |

Retention falls with severity (0.975 → 0.787). The dual gate promotes every seed at
`p ≤ 0.50`, sits on the boundary at `p=0.70` (2/3 promote), and **fails closed for every
seed at `p=0.90`** — where adaptation has become catastrophic forgetting. (Recovery ratio
is small/noisy at `p=0.30` because light drift leaves little loss to regain; the system is
healthy there — retention 0.975, gate passes.)

## Generalization: the same governance on two more model families

The gates and metrics above are `driftguard.governance`, imported **unchanged** by two
non-text instances (tests assert they are the *same objects*). This is the "model-agnostic"
claim as measured code, not prose.

**Tabular** — `make example-tabular` (OpenML Adult, HistGradientBoosting; clean holdout:
baseline 0.783, primary 0.819 macro-F1):

| severity | detected | recovery | retention | dual gate |
|----------|----------|----------|-----------|-----------|
| 0.10     | True     | 0.780    | 0.936     | PASS      |
| 0.20     | True     | 0.765    | 0.861     | FAIL      |
| 0.40     | True     | 0.779    | 0.728     | FAIL      |

**Embeddings** — `make example-embedding` (20 Newsgroups, LogReg on MiniLM sentence
embeddings; clean holdout: baseline 0.838, primary 0.907 macro-F1). Drift is an
information-preserving rotation, so retraining *fully* recovers the drifted task:

| severity | detected | recovery | retention | dual gate |
|----------|----------|----------|-----------|-----------|
| 0.10     | True     | 1.000    | 0.993     | PASS      |
| 0.50     | True     | 1.000    | 0.937     | PASS      |
| 0.75     | True     | 1.000    | 0.606     | FAIL      |

The embedding case is the sharpest statement of the framework's thesis: **recovery alone is
not safety.** A candidate that perfectly relearns the drifted task (recovery ≈ 1.0) is still
refused promotion once it has forgotten the clean distribution (retention 0.606) — precisely
what the forgetting-aware `dual` gate exists to catch. Same gates, same metrics, three model
families and data types.

## Container & stack

- Multi-stage image builds and runs as **non-root (uid 10001)** with a **read-only
  root filesystem**; Docker `HEALTHCHECK` reports healthy. Image size ≈ 1.5 GB
  (MLflow + scientific Python stack).
- `docker compose up` brought up app + Prometheus + Grafana + MLflow. Prometheus
  scraped the app (`target: up`, `driftguard_model_tier` visible); Grafana
  auto-provisioned the Prometheus datasource and the **DriftGuard Service Health**
  dashboard (including the active-model-tier panel).

## Infrastructure (validated, not applied)

- **Terraform:** `fmt` clean, `init` succeeded, `validate` → *"Success! The
  configuration is valid."* Covers ECR (immutable, scan-on-push), two versioned S3
  buckets, VPC + EKS managed node group, least-privilege IRSA, Secrets Manager.
  `apply` is documented and gated on your AWS credentials — no live infra was
  provisioned.
- **Kubernetes:** `kubeconform` on the rendered manifests → **5 valid, 0 invalid, 0
  errors**, 2 Prometheus-Operator CRDs skipped (no offline schema). Probes wired
  (`/ready`, `/health`), HPA present, degraded-tier `PrometheusRule` present.

## Limitations (stated plainly)
- Metrics are single-run, CPU, on one machine; treat them as reproducible baselines,
  not a leaderboard claim.
- PSI on `token_count` catches covariate shift, not subtle semantic drift — the
  Evidently/embedding path is the documented upgrade.
- AG News is a clean, static benchmark; production news streams have higher velocity
  and label noise.
