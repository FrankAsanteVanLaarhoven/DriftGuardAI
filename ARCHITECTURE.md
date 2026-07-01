# DriftGuard architecture

DriftGuard is a text-classification service (AG News topics) engineered around one
principle: **a bad model must never take the service down, and a worse model must
never reach production.** Two baselines enforce this — one operational, one
evaluative.

## The closed loop

```
HF ag_news + DVC(S3) ──▶ train (seeded, MLflow track+registry, baseline-gate) ──▶ Docker(ECR)
   ──▶ Jenkins CI/CD (test │ baseline-gate │ build │ trivy │ push │ staging │ smoke
                       │ HUMAN GATE │ prod │ auto-rollback)
   ──▶ FastAPI on EKS  [ primary model  ⇒  falls back to baseline model ]
   ──▶ Prometheus + Grafana (service health)  +  PSI/Evidently drift monitor (model health)
   ──▶ drift? ──▶ retrain ──▶ baseline-gate ──▶ canary ──▶ (human) promote ──▶ back to serving
```

## The fallback-baseline contract (two senses)

### 1. Operational fallback — stay up
- **Two models at startup.** `baseline` loads from `models/baseline.joblib`; if it
  fails its canary self-test the process exits (never serve with no model). `primary`
  loads best-effort from the MLflow registry (`models:/driftguard@production`) or a
  local pointer file; failure is logged and the service runs degraded.
- **Model-agnostic readiness.** `/ready` returns 200 if *any* tier can serve, so a
  bad primary never pulls the pod out of rotation.
- **Request-time fallback.** `/predict` tries the primary; on any exception *or a
  latency-budget breach* it serves the baseline, increments
  `driftguard_fallback_total`, and tags `served_by:"baseline"`. Never a 5xx for a
  model error.
- **Runtime rotation.** If a pointer-sourced primary's artifact is pulled out from
  under a running service, the next request detects it and degrades gracefully.
- **Introspection + alert.** `/model-info` exposes the tier;
  `driftguard_model_tier{tier="baseline"}==1 for 5m` pages (running degraded).

### 2. Evaluative baseline — never regress
- `train.py` writes `baseline_metrics.json` (accuracy + macro-F1 on the frozen
  holdout for the committed baseline).
- The **baseline gate** (`driftguard.gate`, and the CI stage) is fail-closed:
  `candidate_macro_f1 >= baseline_macro_f1 + MARGIN` or the build/retrain stops and
  nothing is registered/promoted.

## Model choices

The **primary** is a larger TF-IDF (1–2 gram, 50k features) + logistic regression;
the **baseline** is a tiny TF-IDF (unigram, 3k features) + logistic regression. The
baseline is deliberately small and fast so it is a genuinely cheaper safety net. The
design leaves room for a DistilBERT primary (the `transformer` extra) with the linear
model as the fallback — if the transformer OOMs or breaches its latency budget, the
service degrades to the fast classic model instead of going down.

## Components & portability

| Concern            | Choice                                   |
|--------------------|------------------------------------------|
| Data versioning    | DVC (local remote for dev, S3 for prod)  |
| Experiment/registry| MLflow (sqlite locally, server in-stack) |
| Orchestration      | ZenML pipelines (optional extra)         |
| Serving            | FastAPI + uvicorn                        |
| Drift              | PSI + domain-classifier (text-aware) + Evidently/embedding options |
| Packaging          | multi-stage Docker, non-root, read-only  |
| Infra              | Terraform (ECR, S3, VPC/EKS, IRSA, SM)   |
| Config mgmt        | Ansible (VM path)                        |
| CI/CD              | Jenkins (human gate + auto-rollback)     |
| Observability      | Prometheus + Grafana + Operator rules    |

Everything is open source and Kubernetes-native, so the stack lifts to Azure, GCP,
or on-prem with only the Terraform provider changing.

## Rollback (idempotent, two layers)
- **Service:** `kubectl -n driftguard rollout undo deployment/driftguard`.
- **Model:** re-point the MLflow `production` alias to the previous version
  (`registry.promote_version`).

## Drift detection is multi-layer

Two complementary signals run over recent traffic (`src/driftguard/textdrift.py`):

1. **PSI on `token_count`** — cheap, dependency-free covariate-shift proxy.
2. **Domain-classifier** (Rabanser et al. 2019) — a TF-IDF classifier trained to
   separate reference vs. current text; cross-validated ROC-AUC ≈ 0.5 means
   indistinguishable, → 1.0 means drift. It reads the *words*, so it catches semantic
   shift that PSI is blind to. An optional sentence-embedding MMD sits behind the
   `embed` extra.

The `composite_drift` verdict fires if *either* signal trips. Measured: a
semantic-shift sample with an identical length distribution scores PSI 0.0137 (PSI
misses it) but domain-classifier AUC 1.0000 (caught). See `CASE_STUDY.md`.

## Trade-offs / limitations
- The domain classifier needs a reference *text* sample (`reference_sample.json`,
  written at train time); embedding MMD needs the optional `embed` extra.
- The human gate adds latency by design; staged/statistical canary analysis can
  reduce it later.
- Full retrain on breach; continual-learning (LoRA/EWC) is a future option.
