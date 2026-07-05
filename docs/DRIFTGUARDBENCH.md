# DriftGuardBench v0.1 — measuring *safe promotion*, not just drift detection

**Thesis: recovery is not safety.** A model retrained after distribution shift can
fully recover on the drifted data while quietly forgetting the distribution production
still depends on, losing calibration, or collapsing on a single slice. Monitoring tools
measure whether drift was *detected*; DriftGuardBench measures whether the resulting
**promotion decision was safe**. This report assembles every measured result from the
committed benchmark suite into the blueprint's eight-metric scorecard. Every number
traces to a result file in this repository and reproduces from a fixed seed.

*Version 0.1 · 2026-07-05 · driftguard 0.1.0 · all runs: AG News reference service
(TF-IDF + logistic regression primary), seeds fixed, deps locked (`uv.lock`), data
split DVC-versioned.*

---

## 1 · The scorecard

| # | DriftGuardBench metric | Measured (v0.1) | Source |
|---|---|---|---|
| 1 | **Drift detection F1** | **1.00 @ 0.00 FPR** (composite, 9 drift kinds × 5 seeds); the only tool at 1.00/0.00 in the head-to-head (Evidently 0.86 · scipy-KS 0.93 · NannyML FPR 1.00) | `benchmarks/results.json`, `results_head_to_head.json` |
| 2 | **Drift localization delay** | 0 windows (abrupt/incremental/recurring), 1.33 (gradual); **zero pre-change false alarms**; gradual-topic boundary ≤ 10% injection | `results_streaming.json`, `sweep_gradual_topic.json` |
| 3 | **Recovery score** | 0.968 at vocab drift p=0.7 (frontier 0.352 → 0.930 across p=0.3→0.9) | `results_recovery.json`, `results_recovery_sweep.json` |
| 4 | **Retention score** | 0.926 at p=0.7 (frontier 0.975 → 0.787) | same |
| 5 | **Promotion precision** | **1.00** (dual gate; recall 0.89 — reported together, see §4) | `results_recovery_sweep.json` |
| 6 | **Unsafe promotion rate** | **0.00** (dual gate, 12 trials; the refreshed-only gate ships 0.25) | same |
| 7 | **Fallback survival rate** | **100%** — 1248/1248 HTTP 200 through a broken canary deploy; 6 chaos tests incl. a *hanging* registry; live primary deletion → HTTP 200 | `CASE_STUDY.md`, `tests/test_fallback.py` |
| 8 | **Canary rollback correctness** | Automated rollback in **50 s** breach→rollback (86 s release→rollback) with audit annotation; guard idempotent after | `deploy/helm/README.md` |

Blueprint KPI targets — unsafe promotion rate 0, promotion precision ≥ 95%, fallback
availability 100%, deterministic gate under fixed seed, rollback < 5 minutes — are all
met, with the operating trade-offs reported rather than hidden (§4).

## 2 · Protocol

- **Data & service.** `fancyzhx/ag_news`; frozen seeded split (DVC). The governed
  service is the reference text classifier with a committed always-loadable baseline.
- **Drift injection.** Nine seeded generators (`benchmarks/drift_generators.py`,
  Garcia-style + two of our own): `no_drift` (FPR control), `length_truncate`,
  `class_prior_shift`, `adjective_swap`, `semantic_replace`, `gradual_topic`,
  `char_noise`, `token_dropout`, and `semantic_rotation` — the descriptor-*preserving*
  kind (§3). Ground truth is by construction (`IS_DRIFT`).
- **Detection.** Three-layer composite, any-rule: PSI on token count; a
  domain-classifier (TF-IDF + logistic regression, cross-validated AUC — Rabanser et
  al. 2019); descriptor-KS (Bonferroni-corrected two-sample K-S over five text
  descriptors).
- **Governance ground truth.** A benchmark-only *safety oracle* labels each retrained
  candidate: safe ⇔ no regression vs the incumbent on the new distribution AND
  retention ≥ 0.90 of the incumbent's original-distribution score. Gate decisions are
  scored against it (promotion precision / recall / unsafe rate) — the oracle needs
  both models on both distributions, so production gates can only approximate it; that
  is precisely what makes the benchmark necessary.
- **Comparators.** Same-protocol head-to-head vs Evidently 0.7 and NannyML 0.13
  (shared held-out reference, identical descriptor frame, each tool's *native*
  decision rule, no tuning) plus a scipy K-S + Bonferroni baseline standing in for
  Alibi Detect's KSDrift (alibi-detect 0.13.0 cannot install on Python 3.13).

## 3 · Detection results — and the arc that produced them

Per-detector scorecard (ground truth = `is_drift`, every kind × seed):

| detector | precision | recall | F1 | FPR |
|---|---|---|---|---|
| PSI | 1.00 | 0.25 | 0.40 | 0.00 |
| domain classifier | 1.00 | 0.62 | 0.77 | 0.00 |
| descriptor-KS | 1.00 | 0.88 | 0.93 | 0.00 |
| **composite** | 1.00 | **1.00** | **1.00** | **0.00** |

**No single layer matches the composite**, and the claim is measured from both sides:
`gradual_topic`/`char_noise` are invisible to the learned layers but caught by the
classical K-S; `semantic_rotation` (frequent in-vocabulary words consistently swapped
with same-length frequent words — every surface descriptor preserved **by
construction**) is invisible to PSI and K-S (0/5 each) and caught only by the detector
that reads the words (domain AUC 0.9648, 5/5).

Head-to-head (each tool's native rule):

| tool | precision | recall | F1 | FPR | s/window |
|---|---|---|---|---|---|
| **driftguard** | **1.00** | **1.00** | **1.00** | **0.00** | 0.218 |
| evidently | 1.00 | 0.75 | 0.86 | 0.00 | 0.165 |
| nannyml | 0.89 | 1.00 | 0.94 | 1.00 | 0.005 |
| scipy KS baseline | 1.00 | 0.88 | 0.93 | 0.00 | 0.008 |

**The arc is the finding.** Round one of this benchmark had the plain corrected K-S
*beating* the then-two-layer composite (1.00 vs 0.87 F1) — published as-is. The
response was to absorb the winning method as a third detector layer (tying it), then
add `semantic_rotation`, the case the classical method cannot see structurally:
Evidently and the K-S baseline score **0.00** on it, not by tuning but by construction.
NannyML's perfect recall comes the same way as its 1.00 FPR — its std-band thresholds
alarm on everything at this reference size (1500), far below what its docs target; its
strength (per D3Bench) is linking drift to performance impact over long horizons, not
small-window alarming. Window-level detection on descriptor-visible drift is
commoditized; the differentiator is everything below this line.

## 4 · Governance results — the metrics nobody else defines

**Decision quality** (12 trials: 4 severities × 3 seeds, scored against the safety
oracle):

| gate mode | promotions | promotion precision | promotion recall | unsafe promotion rate |
|---|---|---|---|---|
| fixed-holdout only | 2/12 | 1.00 | 0.22 | 0.00 |
| refreshed-holdout only | 12/12 | 0.75 | 1.00 | **0.25** |
| **dual (drift-aware)** | 8/12 | **1.00** | **0.89** | **0.00** |

The refreshed-only gate ships every catastrophic-forgetting candidate at p=0.9 —
*recovery is not safety*, quantified. The fixed-only gate is safe by refusing to adapt
(blocks 7 of 9 safe recoveries). The dual gate ships zero unsafe models at a measured
recall cost of 0.11 — its single miss is a boundary seed it conservatively blocks.

**Recovery–retention frontier** (mean ± std, 3 seeds/point): retention falls
monotonically as drift deepens (0.975 → 0.787) while recovery rises (0.352 → 0.930);
the dual gate tracks the trade-off — every seed promotes at p ≤ 0.5, 2/3 at p = 0.7,
**every seed fails closed at p = 0.9** where adaptation has become forgetting.

**Below the aggregate** (same p=0.7 candidate the dual gate passes): forgetting is
class-concentrated — Sci/Tech −0.085 and Business −0.081 on the fixed holdout vs
Sports −0.045, failing a per-slice 0.05 floor (`slice_gate`) — and the candidate's
old-distribution ECE is ~4× the incumbent's (0.019 → 0.070, `calibration_gate` FAIL)
while *better* calibrated on the new distribution. The aggregate gate answers "may it
ship?"; the slice/calibration report states what shipping accepts. Every decision
exports as a versioned, tamper-evident `PromotionDecisionRecord`
(`docs/PROMOTION_DECISION.md`) whose outcome is *derived* from the gates, never
asserted.

## 5 · Operational results

- **Fallback survival.** The committed baseline covers every primary failure mode
  tested: missing primary, corrupt primary, runtime removal, latency-budget breach,
  and a **hanging model registry** — the mode the canary drill discovered (MLflow's
  client retries an unreachable registry for minutes during startup, CrashLooping pods
  past the probe budget; now bounded by a 20 s load deadline with its own chaos test).
  Through the drill's broken canary deploy the in-cluster probe recorded
  **1248/1248 HTTP 200**.
- **Canary rollback** (kind cluster, chart defaults — per-minute guard, 15 s scrape):
  broken-candidate release 02:21:34 → breach visible in Prometheus 02:22:10 → canary
  scaled to zero with audit annotation 02:23:00. **50 s breach→rollback**, 86 s
  end-to-end, no mesh or Argo required.

## 6 · Generalization

The same governance code — `incumbent_gate`, `promotion_gate`, `recovery_ratio`,
`retention_ratio`, the detectors — imported unchanged across three modalities:
**text** (this report), **tabular** (OpenML Adult, HistGradientBoosting: retention
0.936 → 0.728, gate PASS → FAIL), **embeddings** (20 Newsgroups, MiniLM: recovery 1.0
at every severity while retention collapses 0.993 → 0.606 — the gate refuses the
"perfectly recovered" model).

## 7 · Limitations (stated plainly)

- Single primary corpus (AG News) with **synthetic, seeded** drift — chosen to isolate
  mechanisms; real drift streams and additional datasets are the v0.2 axis.
- No label-delay-aware detection; ground-truth labels are assumed available at
  retrain time.
- The rollback drill ran on kind, not a cloud cluster; the guard image is tag-pinned.
- The safety oracle's retention floor (0.90) is a declared parameter, not a universal
  constant — the decision-quality numbers are relative to it.
- Comparators ran at their native defaults on this protocol; NannyML in particular is
  operating far below its intended reference sizes.

## 8 · Reproduce it

```bash
make benchmark          # §3 scorecard            -> benchmarks/results.json
make benchmark-h2h      # §3 head-to-head         -> results_head_to_head.json
make benchmark-sweep    # §3 detection boundary   -> sweep_gradual_topic.json
make benchmark-stream   # metric 2 latency        -> results_streaming.json
make recovery           # §4 closed loop + slices -> results_recovery.json
make recovery-sweep     # §4 frontier + decision quality -> results_recovery_sweep.json
make example-tabular example-embedding            # §6
make demo               # metric 7 live fallback proof
# metric 8: deploy/helm/README.md — the kind drill runbook
```

## 9 · References

Rabanser, Günnemann & Lipton (2019), *Failing Loudly* — domain-discriminator drift ·
Gama et al. (2014) — temporal drift taxonomy · McCloskey & Cohen (1989); Kirkpatrick
et al. (2017) — catastrophic forgetting · Guo et al. (2017); Ovadia et al. (2019) —
calibration under shift · Breck et al. (2017), *The ML Test Score* — deployment gates ·
Mitchell et al. (2019), *Model Cards* — decision provenance · D3Bench — comparator
strengths context.
