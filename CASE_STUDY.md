# DriftGuard case study — measured results

All numbers below were **measured on this repository** with the committed code and
`fancyzhx/ag_news`, seed 42, on CPU. They are reproducible with `make train`,
`make test`, `make stack`, and `make demo`. Nothing here is estimated.

## Dataset (fixed, seeded, DVC-versioned)

| Split | Rows    | Notes                              |
|-------|---------|------------------------------------|
| train | 108,000 | 90% of the HF train split          |
| val   | 12,000  | 10% stratified holdout (seed 42)   |
| test  | 7,600   | official test split (frozen holdout)|

`dvc repro` is reproducible: a forced rebuild produced byte-identical output md5s.

## Model quality (frozen test holdout)

| Model                | Accuracy | Macro-F1 |
|----------------------|----------|----------|
| Baseline (fallback)  | 0.8958   | 0.8956   |
| Primary (candidate)  | 0.9199   | 0.9197   |

**Baseline gate:** the primary clears the gate (0.9197 ≥ 0.8956 + 0.0). With an
impossible margin (0.5) the same candidate is **blocked** (0.9197 < 1.3956) and the
`production` alias stayed on the previous version — fail-closed confirmed.

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
- Primary predict latency on CPU (local): sub-3 ms per request (0.7–2.8 ms observed).

**Test suite: 16 passed** — unit + integration + **5 fallback/chaos tests**.

## Drift detection (PSI over token_count, threshold 0.2)

| Sample              | PSI      | Verdict |
|---------------------|----------|---------|
| in-distribution     | 0.0137   | stable  |
| shifted (truncated) | 12.5169  | drift → non-zero exit, retrain triggered |

The drift→retrain pipeline detected the shift, retrained a candidate, ran the
fail-closed gate, held at the human gate, and only then promoted.

### Multi-layer detection (PSI + domain-classifier)

Measured with `python -m driftguard.textdrift` — a text-aware domain classifier
(cross-validated ROC-AUC, threshold 0.75) run alongside PSI:

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

Controlled generators (Garcia-style) scored by the composite detector. **Mean
detection on genuine drift = 0.80; false-positive rate on `no_drift` = 0.00.**

| drift kind        | detection | mean PSI | mean domain AUC | PSI fired | domain fired |
|-------------------|-----------|----------|-----------------|-----------|--------------|
| no_drift          | 0.00      | 0.0130   | 0.5215          | 0/5       | 0/5          |
| length_truncate   | 1.00      | 12.5169  | 0.9736          | 5/5       | 5/5          |
| class_prior_shift | 1.00      | 0.0535   | 0.7959          | 0/5       | 5/5          |
| adjective_swap    | 1.00      | 0.0130   | 0.9978          | 0/5       | 5/5          |
| semantic_replace  | 1.00      | 0.0130   | 1.0000          | 0/5       | 5/5          |
| gradual_topic     | 0.00      | 0.0130   | 0.7182          | 0/5       | 0/5          |

PSI alone fires only on the length shift; every semantic category is carried by the
domain classifier. Zero false positives on in-distribution windows. The single miss,
`gradual_topic` at 40% injection (AUC 0.7182, just under the 0.75 threshold), is the
honest hard case — partial/gradual drift is caught at higher severity or a lower
threshold, at some false-positive cost. This is exactly the trade-off the benchmark
quantifies rather than hides.

**Detection boundary (`make benchmark-sweep`, gradual_topic, 5 seeds).** The
domain-classifier AUC rises monotonically with injection fraction and crosses the 0.75
gate at ~50% injection (0.40→0.7067 miss, 0.50→0.7639 caught, 0.90→0.9511); PSI stays
flat at 0.0168 across the whole range. So the single miss above is one point on a clean
boundary curve, not a random failure — the operating threshold sets exactly where
gradual drift is caught.

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
