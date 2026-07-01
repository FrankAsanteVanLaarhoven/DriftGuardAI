# DriftGuard deployment report

- **Date:** 2026-07-01T16:33:25.344473+00:00
- **Change:** retrain primary text classifier (ag_news, seed 42)
- **Baseline (fallback) holdout:** accuracy 0.8958, macro-F1 0.8956
- **Primary (candidate) holdout:** accuracy 0.9199, macro-F1 0.9197
- **Baseline gate:** PASS — candidate macro-F1 0.9197 >= baseline 0.8956 + margin 0.0000 (= 0.8956)
- **MLflow run:** fb31801483c645aba21d4b8663921137
- **Registered version:** 4 (promoted=true)
- **Tests:** run `make test` (unit + integration + fallback) — must be green.
- **Rollback:** service `kubectl rollout undo`; model — move the `production` alias to
  the previous registry version.
