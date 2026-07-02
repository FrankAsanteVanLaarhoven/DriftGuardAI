# Repository guide

Conventions and guardrails for anyone (human or coding agent) working in this repo.

## What this is
DriftGuard: a model-agnostic framework for governed model adaptation under distribution
shift, validated by a self-healing text-classification reference service. Read
`docs/GOVERNANCE.md` for the framework, `ARCHITECTURE.md` for the design, and
`CASE_STUDY.md` for measured results.

## Golden rules
1. **Never break the fallback contract.** `/health`, `/ready`, `/predict`,
   `/model-info`, `/metrics` are stable. A schema change is a breaking change. The
   baseline must always load; the primary must never be able to 5xx the service.
2. **The gate is fail-closed.** Do not weaken `driftguard.gate` or promote a model
   that does not beat the committed baseline on the frozen holdout.
3. **Reproducible or it didn't happen.** Seeds are fixed, deps are locked (`uv.lock`),
   the data split is versioned (DVC). Don't introduce nondeterminism.
4. **No secrets in git.** Secrets come from AWS Secrets Manager / the CI store.
5. **Real numbers only.** `CASE_STUDY.md` carries measured values from your own runs.

## Layout
```
src/driftguard/   config, data, train, gate, drift, registry, api/{main,models}
tests/            test_unit, test_api, test_fallback, test_smoke
pipelines/        training_pipeline, drift_pipeline   (ZenML, optional)
deploy/           k8s/  terraform/  ansible/  monitoring/  Jenkinsfile.drift
artifacts/        metrics.json, baseline_metrics.json, reference.json (+ demo samples)
models/           baseline.joblib  (committed fallback)
```

## Workflow
```bash
make install     # uv venv + locked deps
make lint test   # ruff + pytest (must be green before commit)
make train       # rebuild models + gate metrics
make stack       # local app + Prometheus + Grafana + MLflow
make demo        # end-to-end proof
```
Run `make lint test` before every commit. The full suite includes the fallback chaos
test — keep it green.

## Attribution
All commits, docs, and metadata carry only the repository owner's identity
(Frank Asante Van Laarhoven, frankleroyvan@gmail.com). Do not add other names, credit
lines, co-author trailers, or tool/vendor branding.
