# DriftGuard — demo slide outline

Seven slides mirroring [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md). Each slide lists the on-screen content,
what to **show**, and the **say** (speaker note). Keep slides sparse — the terminal and Grafana do
the talking.

---

## Slide 1 — Title & thesis

**DriftGuard**
*Governed model adaptation under distribution shift*

- A model-agnostic **governance layer** for MLOps
- One line: **recovery is not safety**

**Show:** title slide only.
**Say:** "DriftGuard decides whether a model that's been adapted to drift is actually *safe to
promote* — not just whether drift happened."

---

## Slide 2 — The problem

**When data drifts, teams do one of two risky things**

- Stay **static** → accuracy silently rots
- **Retrain and ship** → the new model may have *forgotten* the distribution production still serves

> Recovering on new data is easy. Recovering **without regressing the incumbent** is the hard part.

**Show:** this slide.
**Say:** "Both failure modes are invisible until they hurt. The missing piece is a *promotion
decision* that weighs recovery against forgetting."

---

## Slide 3 — The approach

**Detect → retrain → gate (decoupled)**

- **Multi-layer detection:** PSI (length) + domain classifier (semantics) + descriptor-KS (classical, corrected)
- **Incumbent-aware gate:** beat `max(baseline, live primary)`, not just a weak floor
- **Dual gate:** promote genuine recovery, **fail closed on forgetting**
- Below the aggregate: **slice gates + calibration (ECE)** state what a PASS accepts
- Metrics on **scalar scores** ⇒ model-agnostic; decisions export as a **sealed, versioned record**

**Show:** this slide (optionally the one-line architecture from `ARCHITECTURE.md`).
**Say:** "Detection triggers a retrain; a score-based gate decides promotion. Keeping them
separate is what makes the governance reusable."

---

## Slide 4 — Reliability, proven (live)

**`make demo` — the fallback contract**

- Trains, runs the full suite incl. a chaos fallback test
- **Deletes the primary mid-flight → service stays up on the baseline (HTTP 200)**
- Flags a shifted sample (PSI 12.52), exits non-zero

**Show:** run `make demo` in the terminal.
**Say:** "The service never fails closed on a bad primary, and drift detection is reproducible."

---

## Slide 5 — Governance strength (the money slide)

**Recovery vs Retention — the dual gate in action**

| instance | recovery @ high drift | retention @ high drift | dual gate |
|----------|----------------------|------------------------|-----------|
| Embeddings · 20 News | **1.000** | **0.606** | **FAIL** |
| Tabular · Adult | 0.779 | 0.728 | **FAIL** |
| Text · AG News | 0.930 | 0.787 | FAIL |

**Show:** the `python3` recovery/retention table from the runbook.
**Say:** "Embeddings make it sharpest — recovery is perfect at every severity, yet retention
collapses to 0.61. Promoting that would wreck production, so the gate refuses it. *Recovery alone
is not safety.* And when the gate *does* pass a candidate, the slice + calibration report states
what that pass accepts — at p=0.7, class-concentrated forgetting and 4× worse old-distribution
calibration, sealed into a tamper-evident decision record."

---

## Slide 6 — Generalizability

**One governance layer · three validated instances**

- **Text** (AG News, TF-IDF/DistilBERT) · **Tabular** (Adult, HistGBM) · **Embeddings** (20 News, MiniLM)
- All import `driftguard.governance` + `driftguard.detectors` **unchanged** (tests assert same objects)
- Adding a modality = supply data + a model; no new governance or detector code

**Show:** optionally `make example-tabular` live.
**Say:** "Not tied to text. The same gates and metrics drive a gradient booster and an embedding
classifier, imported verbatim."

---

## Slide 7 — Observability & close

**Production visibility + the offer**

- Live Prometheus metrics: serving tier, fallback events, latency breaches, baseline share
- Grafana **Adaptation Governance** dashboard (live signals + measured recovery/retention)
- Helm chart: dependency-free **canary + automated rollback — measured 50 s** (breach → rollback),
  probe **1248/1248 HTTP 200** through a broken deploy
- Everything **reproducible & file-backed**; benchmark harness ready to share

**Show:** the Grafana dashboard (`:3001` → *Adaptation Governance*).
**Say:** "Reusable governance, a measurable safety property, three instances, measured rollback
evidence, all reproducible. Happy to explore how these patterns fit your stack — or share the
benchmark harness."

---

### Delivery notes
- **6–8 minutes.** Slides 4–5 are the core; don't rush them.
- If time is tight, drop Slide 6's live run (the Slide-5 table already shows two modalities).
- Differentiator to keep ready, now **measured**: *"vs Evidently/NannyML — same-protocol
  head-to-head: DriftGuard is the **only tool at 1.00 F1 / 0.00 FPR** (Evidently 0.86, scipy-KS
  0.93, NannyML 100% false alarms on clean windows). The arc: a plain K-S beat us in round one —
  we published that, absorbed the method, then added `semantic_rotation`, a drift kind that
  preserves every surface descriptor — descriptor tools score **0.00 structurally**; only reading
  the words catches it. And the decision-quality table (promotion precision 1.00 / unsafe rate
  0.00) measures what monitoring tools don't define: whether the **promotion decision** was safe."*
