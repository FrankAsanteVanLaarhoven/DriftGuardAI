# DriftGuard deployment report

- **Date:** 2026-07-01T15:42:09.922659+00:00
- **Change:** retrain primary text classifier (ag_news, seed 42)
- **Baseline (fallback) holdout:** accuracy 0.8958, macro-F1 0.8956
- **Primary (candidate) holdout:** accuracy 0.9199, macro-F1 0.9197
- **Baseline gate:** FAIL (fail-closed) — candidate macro-F1 0.9197 < baseline 0.8956 + margin 0.5000 (= 1.3956)
- **MLflow run:** 3c0122e5368c40b2a7ae9c4f8755c3af
- **Registered version:** 2 (promoted=false)
- **Tests:** run `make test` (unit + integration + fallback) — must be green.
- **Rollback:** service `kubectl rollout undo`; model — move the `production` alias to
  the previous registry version.
