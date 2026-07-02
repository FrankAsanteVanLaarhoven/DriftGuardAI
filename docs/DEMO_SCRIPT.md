# DriftGuard — Fuzzy Labs demo runbook

**Target:** 6–8 minutes (live) + Q&A. **Thesis to land:** *recovery is not safety* — DriftGuard
is a model-agnostic **governance** layer that decides whether an adapted model is safe to promote,
not just another drift monitor.

All commands below are copy-paste ready for this machine. Every number is from the committed
result files, so nothing is quoted from memory.

---

## 0 · Pre-flight (run **before** the meeting, once)

Warm the caches so the live run is fast, and pre-build the stack image:

```bash
cd ~/driftguard
make demo                                              # trains + caches AG News, ~30–60s warm
make example-tabular                                   # downloads OpenML Adult (first run only)
DRIFTGUARD_APP_PORT=8010 DRIFTGUARD_GRAFANA_PORT=3001 make stack   # builds image + starts stack
```

Then confirm the dashboard is reachable: open **http://localhost:3001** → folder **DriftGuard** →
**"DriftGuard — Adaptation Governance"** (anonymous viewer, no login). Leave it open in a tab.

> Ports 8000 and 3000 are busy on this host, hence the `DRIFTGUARD_APP_PORT` / `DRIFTGUARD_GRAFANA_PORT`
> overrides. Tear the stack down afterwards with `make stack-down`.

---

## 1 · Positioning (30–40s, no command)

> "Most production ML systems either stay static when data drifts, or retrain without safeguards.
> DriftGuard is a **model-agnostic governance layer** that decides whether a newly adapted model is
> **safe to promote**. It pairs multi-layer drift detection with an **incumbent-aware promotion
> gate**, and explicitly measures the trade-off between *recovering* on new data and *retaining*
> performance on the distribution production still depends on."

**Takeaway:** governance & safety, not just detection.

## 2 · Core reliability — `make demo` (≈2 min)

```bash
make demo
```

> "One command runs the full local proof: trains the models, runs the whole test suite including a
> chaos-style fallback test, starts the service, and checks drift detection. Watch step 5 — we
> **delete the primary model mid-flight and the service stays up on the baseline, HTTP 200**, no
> 5xx. Step 6 flags a shifted sample and exits non-zero."

**Point at:** `served_by":"baseline"` on the fallback line; `DRIFT DETECTED: PSI 12.52` at the end.
**Takeaway:** the service never fails closed on a bad primary; drift detection is reproducible.

## 3 · Governance strength — recovery vs retention (≈2 min)

Show a clean table instead of raw JSON:

```bash
python3 - <<'PY'
import json
for name in ("embedding", "tabular"):
    d = json.load(open(f"examples/results_{name}.json"))
    m = d["macro_f1_clean"]
    print(f"\n{d['instance']}  (clean holdout: baseline {m['baseline_fixed']}, primary {m['primary_fixed_incumbent']})")
    print("  severity   recovery   retention   dual-gate")
    for r in d["rows"]:
        gate = "PASS" if r["dual_gate_passed"] else "FAIL  <-- promotion refused"
        print(f"   {r['severity']:<8} {r['recovery_ratio']:<10} {r['retention_ratio']:<11} {gate}")
PY
```

> "This is where most adaptation systems fall short. On the **embedding** instance, as drift
> deepens **recovery stays at 1.0** — the retrained model perfectly re-learns the drifted task — but
> **retention collapses to 0.61** at severity 0.75. Promoting that model would wreck the original
> distribution. Our **dual gate** refuses it. Recovery alone is not safety; the forgetting-aware
> gate is."

**Verified numbers:** embeddings — recovery 1.000 throughout; retention 0.993 → **0.606** (gate PASS
→ **FAIL**). Tabular — retention 0.936 → **0.728**, gate PASS → **FAIL**.
**Takeaway:** the dual gate protects production from a "perfectly recovered" but forgetful model.

## 4 · Generalizability (≈1 min)

The table above already shows **two** modalities. If you want to run one live:

```bash
make example-tabular          # HistGradientBoosting on OpenML Adult, same governance layer
```

> "The exact same governance primitives — `incumbent_gate`, `promotion_gate`, `recovery_ratio`,
> `retention_ratio` — drive a gradient-boosted **tabular** model and a **sentence-embedding** model,
> imported unchanged. A test asserts they're the *same objects* the text service uses. The framework
> is not tied to text classification."

**Takeaway:** one governance layer, three validated instances (text · tabular · embeddings).

## 5 · Observability & production readiness (≈45–60s)

Switch to the Grafana tab (**http://localhost:3001** → **DriftGuard — Adaptation Governance**).

> "Everything is observable. The app exposes custom Prometheus metrics — active serving tier,
> fallback events, latency-budget breaches, baseline traffic share — and this dashboard pairs the
> live serving signals with the measured recovery/retention and the multi-layer detection scorecard."

**Takeaway:** real production visibility, not just offline numbers.

## 6 · Close (20–30s)

> "DriftGuard gives you a reusable, model-agnostic governance layer; a measurable safety property in
> recovery vs retention; and three validated reference implementations — all reproducible and
> file-backed. We'd be glad to discuss how these patterns map to your use cases, or share the
> benchmark harness."

---

## Q&A prep

- **"How is this different from Evidently / Arize / NannyML?"** Those are excellent at *monitoring
  and detection*. DriftGuard's contribution is the **promotion decision**: an incumbent-aware,
  forgetting-aware gate that says whether the adapted model is *safe to ship*. Detection triggers a
  retrain; the score-based gate decides promotion — we keep them decoupled.
- **"Is the drift synthetic?"** Yes — controlled and seeded, to isolate mechanisms cleanly. The
  benchmark harness is designed so real drift streams drop in. This is stated plainly in the
  case study and manuscript.
- **"Does it need a specific model?"** No — the gates and metrics operate on **scalar quality
  scores**, so any model that reports a holdout metric is governable. We prove it on a linear model,
  a gradient booster, and an embedding classifier.
- **"Production maturity?"** Fallback contract, fail-closed CI gate, container, Terraform/EKS, and a
  monitoring stack all exist; a live long-running deployment is the stated next step.

---

## One-page cheat-sheet (keep this open during the demo)

| # | Command | One-line script | Number to hit |
|---|---------|-----------------|---------------|
| 1 | — | "governance & safety, not just detection" | — |
| 2 | `make demo` | "primary deleted → service stays up on baseline, HTTP 200; drift flagged" | `served_by":"baseline"`, `PSI 12.52` |
| 3 | `python3` table (above) | "recovery 1.0 but retention collapses → dual gate refuses" | embeddings ret **0.606 → FAIL** |
| 4 | `make example-tabular` | "same gates on a tabular model, imported unchanged" | tabular ret **0.728 → FAIL** |
| 5 | Grafana `:3001` | "live tier / fallback / breach metrics + measured governance" | dashboard: *Adaptation Governance* |
| 6 | — | "reusable, measurable, three instances, reproducible" | — |

**Fallbacks if time is short:** skip step 4 live (the step-3 table already shows tabular). If the
network is flaky, the executed notebook `notebooks/ag_news_drift_demo.ipynb` has all figures
rendered inline as a backup.
