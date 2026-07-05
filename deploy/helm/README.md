# DriftGuard Helm chart — install, canary, automated rollback

Packages the `deploy/k8s` manifests (which remain the kustomize path) with one
addition the raw manifests don't have: a **dependency-free canary track with
automated Prometheus-driven rollback**. No service mesh, no Argo — a second
Deployment behind the same Service plus a guard CronJob.

## Install / upgrade

```bash
helm install driftguard deploy/helm/driftguard -n driftguard --create-namespace
helm upgrade driftguard deploy/helm/driftguard -n driftguard --set image.tag=<sha>
```

Defaults mirror `deploy/k8s` exactly: 2 replicas, HPA 2→8 at 70% CPU, hardened
security contexts, ServiceMonitor + PrometheusRules (needs kube-prometheus-stack).
The fallback-contract probes (`/health` liveness, `/ready` readiness) are **fixed in
the template, not configurable** — they are the service's stable API (AGENTS.md
golden rule 1).

## Canary flow

```bash
# 1. Candidate passed the offline gate -> it holds the registry @staging alias.
#    Open the canary at ~1/3 traffic (1 canary vs 2 stable replicas):
helm upgrade driftguard deploy/helm/driftguard -n driftguard \
  --set canary.enabled=true --set canary.replicas=1

# 2. Watch it: canary pods serve models:/driftguard@staging; stable serves @production.
kubectl -n driftguard get pods -L track
#    Alerts DriftGuardCanaryHighErrorRate / DriftGuardCanaryDegraded cover it.

# 3a. Promote: move the @production alias to the candidate (MLflow), then close:
helm upgrade driftguard deploy/helm/driftguard -n driftguard --set canary.enabled=false

# 3b. Or it rolls itself back (see below). Manual rollback = the same helm command.
```

Traffic share = `canary.replicas / (replicaCount + canary.replicas)`; the HPA scales
only the stable track, so scale-out *shrinks* the canary share (conservative).

## Automated rollback

With `canary.autoRollback.enabled` (default when the canary is on), a CronJob runs
**every minute** with a ServiceAccount that can scale/annotate **only** the canary
Deployment. It queries Prometheus for a breach:

- canary 5xx ratio over `window` (2m) above `errorRateThreshold` (0.05), **or**
- any canary pod serving from the **fallback baseline tier** — the candidate failed
  to load, so the canary is adding no signal (`driftguard_model_tier`).

Any returned series ⇒ `kubectl scale --replicas=0` + an audit annotation
(`driftguard.io/rolled-back-at`, `driftguard.io/rollback-reason`). Worst-case
rollback latency ≈ rate window (2m) + guard period (1m) + scale time — comfortably
inside a 5-minute budget. Note the double safety: even a *failing* candidate never
5xxes the service from inside the pod (in-process fallback to baseline); the guard
removes it at the traffic level because a canary running on its fallback tier is
pointless and skews the canary read.

Need weighted routing finer than replica ratio, or step-based analysis? Use Argo
Rollouts with an AnalysisTemplate on the same two Prometheus expressions — this
chart's canary is the zero-dependency default, not the ceiling.

## Validation

```bash
helm lint deploy/helm/driftguard
helm template dg deploy/helm/driftguard -n driftguard --set canary.enabled=true \
  | kubeconform -strict -ignore-missing-schemas -summary
```

`tests/test_helm.py` runs both (skipped when helm isn't installed) and asserts the
rendered chart preserves the fallback-contract probes and the guard's least-privilege
RBAC.

## Measured rollback drill (kind, 2026-07-05)

The full loop was exercised on a kind cluster: chart installed with the canary open
(1 canary / 2 stable, plain Prometheus scraping at 15 s), the canary's candidate made
unloadable (`DRIFTGUARD_PRIMARY_MODEL_URI` pointing at an unreachable MLflow **and**
`DRIFTGUARD_PRIMARY_POINTER_PATH` at a nonexistent file), an in-cluster probe hitting
`/predict` once per second throughout.

| event | time (UTC) |
|---|---|
| broken-canary release rolled | 02:21:34 |
| canary Ready — degraded to in-pod baseline, serving 200s | ~02:21:55 |
| breach first observable in Prometheus (`driftguard_model_tier{tier="baseline"}` on a canary pod) | 02:22:10 |
| guard scaled canary to 0 + audit annotation (`driftguard.io/rolled-back-at`) | 02:23:00 |

**Breach-visible → rollback: 50 s. Release → rollback: 86 s** — against a 5-minute
budget, using only the chart's defaults (per-minute guard, 15 s scrape). The traffic
probe recorded **1248/1248 HTTP 200** across the broken deploy and rollback: the
degraded canary answered every request from its in-pod baseline until the guard removed
it, and subsequent guard runs are no-ops ("canary already at 0 replicas").

The drill also caught a real bug before it produced that number: the first attempt
CrashLooped **every** pod because the MLflow client retries an unreachable registry
with minutes of exponential backoff *during startup*, blowing the startup-probe budget —
a platform-level defeat of the fallback contract that a missing-file chaos test can't
see. The fix (`primary_load_timeout_s`, default 20 s, in `driftguard.registry`) bounds
any registry hang and degrades to baseline; it ships with its own chaos test
(`test_hanging_registry_degrades_within_deadline`).
