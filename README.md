# DriftGuard

**A model-agnostic governance layer that decides whether a model adapted to distribution shift
is *safe to promote* — not just whether drift happened.** You can almost always *retrain to
recover* accuracy on drifted data; the recovered model may also have quietly forgotten the
distribution production still depends on. **Recovery is not safety.** DriftGuard detects the
shift, retrains a candidate, and promotes it **only when it is provably no worse than what is
already in production** — quantifying the recovery-vs-forgetting trade-off that decides whether
adaptation is safe.

> Monitoring tools tell you drift *happened*. DriftGuard makes the **promotion decision**: an
> incumbent- and forgetting-aware gate that says whether the adapted model actually ships.

It ships as two layers:

- **The framework (model-agnostic).** Promotion gates and adaptation-safety metrics that operate
  on scalar quality scores, independent of model type or task — the no-worse-than-incumbent gate,
  the drift-aware `dual` gate, recovery / retention metrics — plus a pluggable drift-detector
  interface. See [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) and [`docs/DETECTORS.md`](docs/DETECTORS.md).
- **Three validated reference instances**, exercising the framework end to end with measured
  numbers: **text** (a production-grade self-healing AG News service — multi-layer detection,
  linear + DistilBERT (macro-F1 **0.9412**) primaries, hard fallback contract), **tabular** (Adult /
  HistGradientBoosting), and **embeddings** (20 Newsgroups / MiniLM) — all reusing the *same*
  governance and detector code, unchanged.

Two resilience guarantees sit at the core of the reference service:

- **Operational fallback.** A tiny, dependency-light baseline model
  (`models/baseline.joblib`) is committed in the image and guaranteed to load. The
  service always tries the **primary** first; if the primary is missing, corrupt,
  fails its startup self-test, or throws at inference time, the service **serves the
  baseline and stays up**. It never 5xx's or fails readiness because of a bad primary.
- **Evaluative gate (no-worse-than-incumbent).** A candidate is promoted only if it beats
  **`max(baseline, incumbent primary)`** on a holdout by a configurable margin — never the
  tiny baseline alone. CI and the retrain pipeline **fail closed**; a regression *or a
  downgrade* is never promoted.

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

## Local observability stack

```bash
make stack                                         # app + Prometheus + Grafana + MLflow
# on a host where 8000/3000 are busy, override the ports:
DRIFTGUARD_APP_PORT=8010 DRIFTGUARD_GRAFANA_PORT=3001 make stack
```

Grafana provisions a **DriftGuard** folder with two dashboards against the app's Prometheus
metrics: **DriftGuard — Adaptation Governance** (the demo view — live serving tier, fallback
events, latency-budget breaches, baseline traffic share, plus the measured recovery/retention
and drift-detection scorecard) and **DriftGuard Service Health** (request rate, p95 latency,
error rate). Tear down with `make stack-down`.

## Reproducible research demo (educational companion)

A self-contained Jupyter notebook — `notebooks/ag_news_drift_demo.ipynb` — **executed with all
figures rendered inline** (styled EDA, confusion matrices, detector scorecards, risk–coverage,
recovery/retention, and a cross-modality comparison) demonstrates the core hypotheses on real
`fancyzhx/ag_news` data, plus narrative sections (challenges, insights, limitations, future work):

- **H1** — uncertainty-aware fallback for graceful degradation
- **H2** — multi-layer text-aware drift detection (the domain classifier catches what PSI misses)
- **H3** — the closed self-healing loop (detect → retrain → gate → promote), and *recovery ≠ safety*
- **G** — the same governance generalising across text, tabular, and embedding instances

**Quick start:** open it in **Colab** or **Kaggle** (CPU-only, ~2 min), or locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install datasets scikit-learn scipy pandas matplotlib jupyter
jupyter lab notebooks/ag_news_drift_demo.ipynb
```

All numbers are generated at runtime from a fixed seed (`SEED=42`), including the honest
failure cases (e.g. PSI blindness on semantic drift). It is the lightweight *research*
companion to the production service in this repo.

## Docs & layout
- `docs/GOVERNANCE.md` — **the model-agnostic framework**: promotion gates + adaptation-
  safety metrics, and how to instantiate them for a non-text model.
- `docs/DETECTORS.md` — **the pluggable drift-detector interface** (PSI / domain-classifier /
  composite), reused across text, tabular, and embeddings with no new detector code.
- `examples/` — **reference instances** proving the framework generalises: a **tabular**
  model on OpenML Adult (`make example-tabular`) and an **embedding** model on 20 Newsgroups
  (`make example-embedding`) reuse the same gates, metrics, and detectors as text — three
  model families and data types, one governance layer.
- `ARCHITECTURE.md` — the closed loop and the two-sense fallback contract.
- `CASE_STUDY.md` — measured numbers (model quality, PSI, resilience, infra checks).
- `docs/MANUSCRIPT.md` — **the full research write-up**: methods, environments/setups,
  training iterations, **failed approaches + mitigations**, benchmark/metrics, outcomes, lessons.
- `docs/DEMO_SCRIPT.md` — **a 6–8 min live demo runbook** (pre-flight, commands, talking points,
  Q&A prep, one-page cheat-sheet) for walking a reviewer through the governance story.
- `docs/DEMO_SLIDES.md` — a **7-slide outline** mirroring the runbook (content · show · say).
- `AGENTS.md` — repository conventions and guardrails.
- `deploy/terraform/README.md` — exact AWS `apply` + `kubeconfig` steps.
- `docs/DISTILBERT.md` — GPU runbook for the DistilBERT primary (linear model as fallback).
- `benchmarks/README.md` — drift-injection benchmark, severity sweep, closed-loop recovery,
  promotion decision quality, and the head-to-head vs Evidently / NannyML.

```
src/driftguard/{config,data,train,gate,governance,drift,textdrift,registry,detectors/,api/}  tests/  pipelines/
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
