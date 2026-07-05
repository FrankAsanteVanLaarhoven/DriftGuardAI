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

## 3b · Below the aggregate — what a PASS accepts (≈30s, optional but strong)

No command — quote the committed closed-loop run (`benchmarks/results_recovery.json`):

> "One more layer of honesty: at p=0.7 our aggregate dual gate **passes** the retrained
> candidate — retention 0.926, defensible. The slice and calibration report states what that
> pass *accepts*: forgetting concentrates in two classes (Sci/Tech −0.085, Business −0.081,
> nearly double Sports), and the candidate is **4× worse calibrated** on old-distribution
> traffic. The gate answers *may it ship*; the risk report says *what shipping means*. Both are
> sealed into a versioned, tamper-evident **PromotionDecisionRecord** for whoever executes the
> promotion."

**Takeaway:** governance below the macro average — slices, calibration, and an auditable contract.

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

Then land the production-readiness numbers (no command — they're measured and committed):

> "And it deploys like a real service: a Helm chart with a **dependency-free canary** — second
> deployment behind the same Service, serving the gate-passed candidate — and a Prometheus-driven
> guard that rolls a bad canary back automatically. We drilled it on a kind cluster: **50 seconds
> from breach-visible to rollback**, 86 including the whole broken deploy, and the traffic probe
> logged **1248/1248 HTTP 200** throughout — the in-pod fallback answered every request while the
> guard removed the canary from traffic. The drill also caught a real bug — unbounded MLflow
> retries at startup — which is now fixed and chaos-tested. That found-and-fixed story is in the
> case study, on purpose."

**Takeaway:** real production visibility *and* measured rollback evidence, not just offline numbers.

## 6 · Close (20–30s)

> "DriftGuard gives you a reusable, model-agnostic governance layer; a measurable safety property in
> recovery vs retention; and three validated reference implementations — all reproducible and
> file-backed. We'd be glad to discuss how these patterns map to your use cases, or share the
> benchmark harness."

---

## Q&A prep

- **"How is this different from Evidently / Arize / NannyML?"** Those are excellent at *monitoring
  and detection* — and we didn't just claim that, we **measured it**: a same-protocol head-to-head
  (shared reference, shared descriptor features, each tool's native decision rule) has DriftGuard's
  composite at **1.00 F1 / 0.00 FPR** vs Evidently 0.92 and NannyML at perfect recall but a **100%
  false-alarm rate** on clean windows at this reference size (`benchmarks/README.md`). The first
  run of that benchmark had a plain scipy K-S *beating* our composite — we published that, then
  absorbed the method as a third detector layer. But the real differentiator is the **promotion
  decision**: the decision-quality table (promotion precision 1.00, recall 0.89, unsafe promotion
  rate 0.00 for the dual gate) measures something none of those tools define.
- **"Is the drift synthetic?"** Yes — controlled and seeded, to isolate mechanisms cleanly. The
  benchmark harness is designed so real drift streams drop in. This is stated plainly in the
  case study and manuscript.
- **"Does it need a specific model?"** No — the gates and metrics operate on **scalar quality
  scores**, so any model that reports a holdout metric is governable. We prove it on a linear model,
  a gradient booster, and an embedding classifier.
- **"Production maturity?"** Fallback contract (6 chaos tests, including a *hanging* registry),
  fail-closed CI gate, container, Helm chart with canary + **measured 50s automated rollback**
  (kind drill; Terraform/EKS for the cloud path), and a monitoring stack. A live long-running
  deployment is the stated next step.
- **"What executes the promotion?"** Whatever you already run. Every decision exports as a
  versioned, tamper-evident **`PromotionDecisionRecord`** (plain JSON, sha-256 sealed, fail-closed
  derived decision, `hold_for_human` first-class) — a CI human gate or deployment controller
  consumes it with three checks. Spec: `docs/PROMOTION_DECISION.md`.

---

## One-page cheat-sheet (keep this open during the demo)

| # | Command | One-line script | Number to hit |
|---|---------|-----------------|---------------|
| 1 | — | "governance & safety, not just detection" | — |
| 2 | `make demo` | "primary deleted → service stays up on baseline, HTTP 200; drift flagged" | `served_by":"baseline"`, `PSI 12.52` |
| 3 | `python3` table (above) | "recovery 1.0 but retention collapses → dual gate refuses" | embeddings ret **0.606 → FAIL** |
| 3b | — (quote committed run) | "dual gate passes; slices + calibration say what that accepts; sealed record" | Sci/Tech **−0.085**, ECE **4×**, `hold_for_human` |
| 4 | `make example-tabular` | "same gates on a tabular model, imported unchanged" | tabular ret **0.728 → FAIL** |
| 5 | Grafana `:3001` | "live metrics + Helm canary with measured auto-rollback" | rollback **50 s**, probe **1248/1248 · 200** |
| 6 | — | "reusable, measurable, three instances, reproducible" | — |

**Numbers to have loaded for Q&A:** head-to-head **1.00 F1 / 0.00 FPR** (Evidently 0.92 · NannyML
FPR 1.00); decision quality **precision 1.00 / recall 0.89 / unsafe rate 0.00**; detection
boundary ≤10% injection; test suite **70 passed**.

**Fallbacks if time is short:** skip step 4 live (the step-3 table already shows tabular). If the
network is flaky, the executed notebook `notebooks/ag_news_drift_demo.ipynb` has all figures
rendered inline as a backup.
