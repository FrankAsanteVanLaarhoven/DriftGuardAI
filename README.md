# DriftGuard

A production-grade, **self-healing text-classification service**. It classifies news
text into AG News topics (World / Sports / Business / Sci/Tech), detects data drift,
and retrains safely behind a human gate — while **never going down because of a bad
primary model**.

Two resilience guarantees sit at the core:

- **Operational fallback.** A tiny, dependency-light baseline model
  (`models/baseline.joblib`) is committed in the image and guaranteed to load. The
  service always tries the **primary** first; if the primary is missing, corrupt,
  fails its startup self-test, or throws at inference time, the service **serves the
  baseline and stays up**. It never 5xx's or fails readiness because of a bad primary.
- **Evaluative baseline gate.** A candidate model is only promoted if it beats the
  committed baseline on a fixed holdout by a configurable margin. CI and the retrain
  pipeline **fail closed** — a regression is never promoted.

## Quick start (local)

```bash
make install        # uv venv + pinned deps
make data           # build the fixed, seeded ag_news split (DVC-tracked)
make train          # train primary + baseline, register in MLflow, write gate metrics
make test           # unit + integration + FALLBACK test, all green
make run            # serve on :8000
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"New GPU sets an on-device AI record."}'      # -> Sci/Tech
```

## The closed loop

```
HF data + DVC(S3) -> train (seeded, MLflow track+registry, baseline-gate) -> Docker(ECR)
  -> Jenkins CI/CD (test | baseline-gate | build | scan | push | staging | smoke
                    | HUMAN GATE | prod | auto-rollback)
  -> FastAPI on EKS  [primary model  =>  falls back to baseline model]
  -> Prometheus + Grafana  +  PSI/Evidently drift monitor
  -> drift? -> retrain -> baseline-gate -> canary -> (human) promote -> back to serving
```

See `ARCHITECTURE.md` for the design, `CASE_STUDY.md` for measured numbers, and the
`deploy/` tree for Terraform (AWS), Kubernetes, Ansible, and monitoring.

## Prove it end to end

```bash
make demo
```
Installs, trains, runs the full test suite (including the fallback chaos test),
serves predictions, **removes the primary and shows the service stay up on the
baseline (HTTP 200)**, then flags drift on a shifted sample.

## Docs & layout
- `ARCHITECTURE.md` — the closed loop and the two-sense fallback contract.
- `CASE_STUDY.md` — measured numbers (model quality, PSI, resilience, infra checks).
- `CLAUDE.md` — repository conventions and guardrails.
- `deploy/terraform/README.md` — exact AWS `apply` + `kubeconfig` steps.
- `docs/DISTILBERT.md` — GPU runbook for the DistilBERT primary (linear model as fallback).
- `benchmarks/README.md` — drift-injection benchmark, severity sweep, and closed-loop recovery.

```
src/driftguard/{config,data,train,gate,drift,registry,api/}  tests/  pipelines/
deploy/{k8s,terraform,ansible,monitoring}  Dockerfile  docker-compose.yml  Jenkinsfile
models/baseline.joblib  artifacts/{metrics,baseline_metrics,reference}.json
```

## Status

Built phase by phase; each phase has acceptance criteria that must pass before the
next. Local phases (data → train → serve → stack → drift) run end to end. AWS
provisioning is real Terraform that applies **with your credentials** — nothing is
faked; the exact `apply` steps are documented in `deploy/terraform/README.md`.

## License
Apache-2.0. Copyright 2026 Frank Asante Van Laarhoven.
