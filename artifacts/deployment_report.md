# DriftGuard deployment report

- **Date:** 2026-07-05T04:17:48.213665+00:00
- **Change:** retrain primary text classifier (ag_news, seed 42)
- **Baseline (fallback) holdout:** accuracy 0.8958, macro-F1 0.8956
- **Primary (candidate) holdout:** accuracy 0.9199, macro-F1 0.9197
- **Promotion gate (no-worse-than-incumbent):** PASS — candidate macro-F1 0.9197 >= max(baseline 0.8956, incumbent 0.9197) + margin 0.0000 (= 0.9197; bar set by incumbent primary)
- **MLflow run:** 092b95199b9b41beace6142425e08d59
- **Registered version:** 6 (promoted=true)
- **Tests:** run `make test` (unit + integration + fallback) — must be green.
- **Rollback:** service `kubectl rollout undo`; model — move the `production` alias to
  the previous registry version.
