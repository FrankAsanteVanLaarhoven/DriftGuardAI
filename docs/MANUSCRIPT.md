# DriftGuard: Governed Model Adaptation Under Distribution Shift

**A reproducible benchmark and reference implementation, with a full account of the
investigation — environments, setups, training iterations, failed approaches and their
mitigations, benchmarks, metrics, outcomes, and lessons.**

Frank Asante Van Laarhoven · `frankleroyvan@gmail.com`

---

## Abstract

Production machine-learning systems fail not when they cannot *recover* from distribution
shift, but when they recover *unsafely* — a model retrained on drifted data can regain
accuracy on the new distribution while silently forgetting the distribution production still
depends on. DriftGuard reframes drift response as a **governance** problem: detect the shift,
retrain a candidate, and promote it **only when it is provably no worse than the incumbent**,
quantifying the recovery-versus-forgetting trade-off that decides whether adaptation is safe.

We contribute (i) a **model-agnostic governance layer** — incumbent-aware and forgetting-aware
promotion gates plus recovery/retention metrics that operate purely on scalar quality scores;
(ii) a **pluggable, modality-agnostic detector interface** (PSI, domain-classifier, MMD,
composite); (iii) a **drift-injection benchmark** with per-detector scorecards, a severity
sweep, streaming detection latency, and closed-loop recovery; and (iv) **three measured
reference instances** — text (AG News), tabular (Adult), and embeddings (20 Newsgroups) — that
reuse the same governance layer *verbatim*. The central, transferable finding is stated most
sharply by the embedding instance: a candidate can achieve **recovery ≈ 1.0 at every drift
severity yet be correctly refused promotion** because retention collapses to 0.61 — *recovery
alone is not safety*.

This manuscript is deliberately an honest engineering record. It documents the approaches that
**failed** (three separate embedding-drift designs; an environment mis-assumption; a
merge-tooling defect; a staging omission that briefly broke `main`) and the mitigation
investigations that resolved each, alongside the setups, training runs, and measured outcomes.

---

## 1 · Introduction

### 1.1 Problem

A deployed classifier meets inputs drawn from a distribution that moves over time — vocabulary
shifts, topics change, inputs degrade, feature semantics drift. The operational reflex is to
retrain on fresh data. But retraining raises a governance question that accuracy-on-new-data
does not answer: **is the retrained model safe to promote?** A candidate specialised to the
drifted distribution may under-perform on the original distribution that the rest of production
still serves. Promoting it trades a visible problem (drift) for an invisible one (forgetting).

### 1.2 Thesis

**Recovery ≠ safety.** Recovering accuracy on the new distribution is usually easy; recovering
*without regressing the incumbent* is the property that must be measured and gated. DriftGuard
makes both quantities first-class and lets a fail-closed gate arbitrate.

### 1.3 Contributions

1. **Governance layer (`driftguard.governance`)** — gates and metrics on scalar scores, so they
   are independent of model family, task, or modality.
2. **Detector interface (`driftguard.detectors`)** — a `DriftDetector` protocol with PSI,
   domain-classifier, MMD, and composite implementations, each adapted to a modality by a small
   extractor rather than by new detector code.
3. **Benchmark harness** — controlled drift generators, per-detector precision/recall/F1/FPR
   scorecards, a detection-boundary severity sweep, streaming detection latency across the
   temporal-drift taxonomy, and a closed-loop recovery loop.
4. **Three measured instances** — text, tabular, embeddings — turning "model-agnostic" from a
   claim into runnable, measured code.

### 1.4 Positioning

DriftGuard is framed as **infrastructure + benchmark**: a reusable governance framework with
text classification as one *validated instance*, not the whole story. The reference text
service is production-shaped (fallback contract, fail-closed CI gate, container, IaC), so the
framework is exercised end to end rather than in isolation.

---

## 2 · System and methods

### 2.1 The closed loop

```
HF data + DVC  →  train (seeded, MLflow track+registry, gate)  →  Docker/ECR
  →  CI/CD (test | gate | build | scan | staging | HUMAN GATE | prod | auto-rollback)
  →  FastAPI on EKS   [primary model  ⇒  falls back to baseline model]
  →  Prometheus + Grafana  +  PSI / domain-classifier drift monitor
  →  drift? → retrain → gate → canary → (human) promote → back to serving
```

### 2.2 The two-sense fallback contract

- **Operational fallback.** A tiny, dependency-light baseline is committed in the image and
  guaranteed to load. The service always tries the primary first; if the primary is missing,
  corrupt, fails its startup self-test, or throws at inference, the service **serves the
  baseline and stays up** — it never 5xx's or fails readiness because of a bad primary.
- **Evaluative fallback.** A candidate is promoted only if it beats `max(baseline, incumbent)`
  on a frozen holdout by a margin — never the tiny baseline alone.

### 2.3 Governance: gates and metrics (on scalar scores)

Let `orig` be the pre-drift primary's score on the clean/fixed holdout, `stale_new` the pre-drift
primary's score on the drifted holdout, `cand_new`/`cand_orig` the retrained candidate's scores
on the drifted/clean holdouts.

- **`baseline_gate`** — CI floor: candidate must clear the committed baseline (fail-closed).
- **`incumbent_gate`** — candidate must beat `max(baseline, incumbent) + margin`; closes the
  "promote a downgrade" gap where a candidate beats the weak baseline but not the live primary.
- **`promotion_gate` (dual mode)** — promote iff the candidate (a) beats the refreshed baseline
  on the drifted holdout **and** (b) clears a forgetting floor on the clean holdout
  (`cand_orig ≥ orig − floor`). This is the mechanism that lets genuine recovery through while
  failing closed on catastrophic forgetting.
- **Recovery ratio** `= (cand_new − stale_new) / (orig − stale_new)` — fraction of drift-induced
  loss regained on the *new* distribution. `1.0` = fully restored, `0` = no recovery.
- **Retention ratio** `= cand_orig / orig` — share of the *original* distribution's score kept
  after adapting. `1.0` = no forgetting.

Governance is **deliberately decoupled from detection**: detection triggers a retrain; the
score-based gate decides promotion. Coupling them would conflate two orthogonal concerns.

### 2.4 The pluggable detector interface

`DriftDetector` is a minimal protocol: `fit(reference) → detect(current) → DetectionResult`.
Detectors are modality-agnostic *by composition* — each is configured with a small extractor,
not new detector code:

| Detector | Adapter it takes | Signal |
|----------|------------------|--------|
| `PSIDetector` | `values_fn`: batch → scalars | Population Stability Index on any scalar (token count, a feature column, an embedding projection). |
| `DomainClassifierDetector` | an sklearn estimator on the raw items | Cross-validated reference-vs-current ROC-AUC (Rabanser et al. 2019). |
| `MMDDetector` | embedding vectors (linear/RBF kernel) | Maximum Mean Discrepancy two-sample test. |
| `CompositeDetector` | — | Unions detectors with a safety-first `any` (or `all`) rule. |

---

## 3 · Experimental environment and setup

### 3.1 Hardware

- **GPU:** NVIDIA RTX 4080 SUPER (16 GB), CUDA available. *(This was itself a corrected
  assumption — see §6.2.)* Transformer fine-tuning auto-selects `cuda`.
- **CPU-only** paths for all linear/tabular/embedding-classifier work and the full test suite.

### 3.2 Software

- Python 3.13, `uv` (locked `uv.lock`), `hatchling`, `ruff`, `pytest`.
- scikit-learn, FastAPI, MLflow (sqlite backend + registry), DVC, Prometheus/Grafana, Docker,
  Terraform (EKS/ECR/S3/IRSA), Jenkins; ZenML optional.
- Transformer extra: `torch` (cu130 build) + `transformers`. Embedding extra:
  `sentence-transformers` (`all-MiniLM-L6-v2`).

### 3.3 Local-environment gotchas (documented so runs are reproducible)

- Port **8000** is occupied by an unrelated local service — the API is served on an alternate
  port in this environment.
- A sourced ROS setup injects Python 3.10 site-packages onto `PYTHONPATH`, which breaks the 3.13
  venv; the Makefile `unexport`s it and ad-hoc runs use `env -u PYTHONPATH`.
- A local `docker` shim forwards only to `docker compose`; the real engine is at `/usr/bin/docker`.

### 3.4 Data and reproducibility

- **Text:** `fancyzhx/ag_news` (4 balanced classes), a fixed seeded split, DVC-versioned.
- **Tabular:** OpenML Adult / Census Income (`fetch_openml("adult", version=2)`).
- **Embeddings:** 20 Newsgroups, four well-separated categories, headers/footers/quotes stripped.
- **Determinism:** `SEED = 42` throughout; locked dependencies; versioned data. Every reported
  number is file-backed (`artifacts/metrics*.json`, `examples/results_*.json`,
  `benchmarks/results*.json`) and regenerable via `make` targets.

---

## 4 · Instances, training, and iterations

Each instance supplies its own data, model, and detector configuration but imports
`driftguard.governance` and `driftguard.detectors` **unchanged** (tests assert the objects are
identical, not re-implementations).

### 4.1 Text (the reference service)

- **Models:** a dependency-light linear baseline (TF-IDF + logistic regression) and two primary
  options — a stronger linear primary and a **DistilBERT** primary.
- **DistilBERT training:** 3 epochs over ~108k rows on the RTX 4080 SUPER →
  **accuracy 0.9413, macro-F1 0.9412**, device `cuda`. Result is file-backed
  (`artifacts/metrics_transformer.json`) and regenerable via `make train-transformer`
  (re-run confirmed identical). The committed default serving pointer remains the **linear**
  model (portable, torch-free tests stay green); DistilBERT is served on demand.
- **Detection:** PSI on token count + a TF-IDF/logistic-regression domain classifier, unioned by
  the composite. The text path was later migrated onto the shared detectors with **byte-exact
  benchmark parity** (§6.5).

### 4.2 Tabular (Adult)

- **Models:** HistGradientBoosting primary (a different, non-linear family) vs a logistic-
  regression baseline. Clean holdout: **baseline 0.783, primary 0.819** macro-F1.
- **Drift:** covariate shift on numeric feature columns (scale + jitter); categoricals untouched.
- **Detection:** shared `PSIDetector` on a feature + `DomainClassifierDetector` on numeric
  features — the same classes text uses.

### 4.3 Embeddings (20 Newsgroups)

- **Models:** logistic regression on `all-MiniLM-L6-v2` sentence embeddings (384-d) as primary;
  a deliberately weaker logistic regression on a **3-D TruncatedSVD projection** as baseline.
  Clean holdout: **baseline 0.838, primary 0.907** macro-F1.
- **Drift:** an information-preserving orthogonal rotation of a fraction of embedding dimensions
  (this design was reached only after three failed attempts — see §6.1).
- **Detection:** shared `DomainClassifierDetector` + `PSIDetector` on the top principal component
  — **zero new detector code** for a third modality.

---

## 5 · Benchmark and metrics

### 5.1 Drift-injection benchmark (5 seeds, window 600)

Mean detection on genuine drift **1.00**, false-positive rate on `no_drift` **0.00**.

| detector | precision | recall | F1 | FPR |
|----------|-----------|--------|----|-----|
| PSI | 1.00 | 0.25 | 0.40 | 0.00 |
| domain_classifier | 1.00 | 0.62 | 0.77 | 0.00 |
| descriptor_ks | 1.00 | 0.88 | 0.93 | 0.00 |
| **composite** | 1.00 | **1.00** | **1.00** | 0.00 |

The layered story, in the order it was measured: PSI fires only on length shifts; the domain
classifier carries every strong semantic category ("the domain classifier catches what PSI
misses" — an independent notebook run reproduces the mechanism starkly: on `semantic_replace`,
PSI = 0.006, blind, while domain-AUC = 1.000). The two-layer composite measured **0.71 recall /
0.83 F1**, and a head-to-head against Evidently, NannyML, and a Bonferroni-corrected scipy K-S
baseline (`benchmarks/head_to_head.py`) then showed the *classical* test winning the suite
outright (1.00 F1) — every generator moves at least one surface descriptor. That finding was
absorbed rather than argued with: a third **descriptor-KS layer** (five text descriptors,
family-wise α=0.05) closed the `gradual_topic`/`char_noise` gap, bringing the composite to
1.00/1.00 at zero FPR. The converse gap was then closed too: a ninth generator,
`semantic_rotation` (frequent in-vocabulary words consistently swapped with same-length
frequent words — all five descriptors preserved **by construction**), is missed by PSI and the
K-S layer (0/5 each) and caught by the domain classifier alone (AUC 0.9648, 5/5). With it, no
single layer matches the composite (K-S 0.93, domain 0.77 vs composite 1.00), and in the
head-to-head Evidently and the classical baseline score **0.00** on that kind structurally —
the multi-layer claim, and the comparison differentiator, are both measurement rather than
design.

### 5.2 Detection boundary (severity sweep)

With the descriptor-KS layer the composite detects gradual topic drift at every injection
fraction down to **10 %** (`oov_rate` moves decisively at any severity). The domain-classifier-
only boundary remains visible in the AUC column: it rises monotonically and crosses the 0.75
gate at ~50 % injection; PSI stays flat — structurally blind to length-preserving shift. That
AUC curve is the operating-point reference for deployments running without the K-S layer.

### 5.3 Streaming detection latency (Gama et al. 2014 taxonomy)

| pattern | detection delay (windows) | missed | pre-change false alarm | post-change detection |
|---------|---------------------------|--------|------------------------|-----------------------|
| abrupt | 0.00 | 0.00 | 0.000 | 1.00 |
| gradual | 1.33 | 0.00 | 0.000 | 0.77 |
| incremental | 0.00 | 0.00 | 0.000 | 1.00 |
| recurring | 0.00 | 0.00 | 0.000 | 0.60 |

The detector fires within one window of an abrupt/incremental change, lags ~1.3 windows on
gradual drift (it must accumulate drifted traffic to separate), and never raises a pre-change
false alarm.

### 5.4 Closed-loop recovery (vocabulary concept drift, p = 0.7)

Detected by the domain classifier (AUC 1.0000) in **0.25 s**; PSI blind (0.0142). Retrain
**23.0 s** → **time-to-recovery 24.0 s**. **Recovery 0.968, retention 0.926.** The candidate
*fails* the fixed-holdout gate (0.8519 < 0.8956) but *passes* the drift-refreshed gate
(0.9170 ≥ 0.7993); the **dual** gate promotes it because it clears the forgetting floor —
recovery unblocked, safety intent preserved.

---

## 6 · Failed approaches and mitigation investigations

This section is the core of the engineering record. Each item is a real dead-end encountered
during the work, the diagnosis, and the fix.

### 6.1 Designing a *learnable* embedding drift (three failures)

The embedding instance needed a drift that (a) degrades the stale model, (b) is recoverable by
retraining, and (c) creates a forgetting risk — otherwise recovery/retention are uninformative.

1. **Random word-contamination → recovery ≈ 0.** Appending random off-topic words to documents
   and re-embedding produced *noise*, not a learnable shift. Retraining cannot adapt to noise:
   at severity 0.9 the stale model dropped only to 0.82 and the candidate to 0.83 — recovery
   ~0.1 by construction. *Diagnosis:* the drift was not a distribution the model could learn.
2. **Embedding scale + jitter → no degradation.** A systematic scale/jitter of half the
   dimensions barely dented a logistic model on well-separated MiniLM embeddings (stale 0.909 ≈
   primary 0.907): with `stale_new ≈ orig`, the recovery denominator collapses. *Diagnosis:* the
   task was too easy and the linear model too robust to this shift.
3. **Random Gaussian rotation → negative recovery.** A full random-matrix "rotation" was
   ill-conditioned and *destroyed* information at higher severity; the candidate retrained on the
   collapsed space scored *below* the stale model (recovery negative). *Diagnosis:* the transform
   was not information-preserving.

**A separate, compounding bug.** Even the corrected rotation initially gave negative recovery
because the transform was drawn from a shared RNG *sequentially* for train and eval — producing
**two different rotations**. The candidate trained under one transform and was tested under
another, so it could not recover. *Mitigation:* generate a **single seeded transform** and apply
it identically to train and eval.

**Resolution.** An **information-preserving orthogonal rotation of a `severity`-fraction of the
embedding dimensions** (QR of a random matrix within the chosen subspace), fixed per severity.
This is systematic, learnable, and detectable. Measured outcome:

| severity | detected | recovery | retention | dual gate |
|----------|----------|----------|-----------|-----------|
| 0.10 | True | 1.000 | 0.993 | PASS |
| 0.25 | True | 1.000 | 0.989 | PASS |
| 0.50 | True | 1.000 | 0.937 | PASS |
| 0.75 | True | 1.000 | **0.606** | **FAIL** |

Because the rotation preserves information, retraining *fully* recovers (recovery ≈ 1.0
throughout) — yet retention falls as more dimensions rotate, and the gate flips exactly when
forgetting turns severe. This is the sharpest statement of *recovery ≠ safety* in the project.

### 6.2 The GPU mis-assumption

The work initially proceeded on the belief the host was CPU-only, and the transformer path was
documented as "not executed". Re-checking the hardware revealed an **RTX 4080 SUPER with working
CUDA**. *Mitigation:* corrected the documentation, ran the real DistilBERT fine-tune
(§4.1), and flagged the correction explicitly rather than quietly. Lesson: verify environment
assumptions before encoding them into docs.

### 6.3 DistilBERT promotion broke the torch-free fallback test

Promoting the DistilBERT bundle repointed serving at a torch-loadable artifact, which broke a
latency/fallback test that runs in a torch-less environment. *Mitigation:* keep the committed
default pointer **linear** (portable), serve DistilBERT on demand via `make run-transformer`,
and restore the linear pointer after any local promote. The evaluative result (0.9412, gate
passed) stands; the operational default stays torch-free so the fallback contract holds in CI.

### 6.4 Merge tooling silently discarded work

Stacked pull requests were "merged" into their parent branches rather than `main` because the
local `gh` (v2.4.0) **silently no-ops `pr edit --base`** (no retarget). *Mitigation:* merge the
top-of-stack branch into `main` directly; do not trust stacked-PR retargeting on this tool.

### 6.5 Byte-exact parity when migrating the text detector

Migrating the text service onto the shared detectors risked changing the committed benchmark
numbers. Exact parity required (i) routing PSI through `PSIDetector.from_reference` reading the
*frozen* training reference (reproducing `drift.compute_psi` to <1e-9, guarded by a regression
test) and (ii) aligning the domain classifier's balancing to **subsample only the larger side**
with `splits=5`. *Outcome:* the drift benchmark is **byte-identical** after migration (the
then-two-layer scorecard 0.29 / 0.57 / 0.71 unchanged), so consolidation was
behaviour-preserving. (The composite later gained the descriptor-KS layer — §5.1 — which
deliberately *did* change the scorecard, to 1.00.)

### 6.6 A staging omission briefly broke `main`

A commit added a new example but its `git add` **omitted the accompanying test file**, so a
fresh checkout of `main` failed a test that referenced a since-removed helper (the local working
tree masked it). *Mitigation:* the follow-up commit repaired `main`; process lesson — `git
status` before every commit and stage *every* file an edit touched, not just the "main" ones.

### 6.7 Smaller mitigations

- **`ruff` lints `.ipynb`** by default and broke `make lint` when the notebook was added →
  `extend-exclude = ["*.ipynb"]`.
- **`OneHotEncoder` sparse output** broke HistGradientBoosting on Adult → `sparse_output=False`.
- **Baseline gap on embeddings:** MiniLM is so information-dense that even a 10-D projection
  scored ~0.90; a visible incumbent gap required a **3-D** projection (0.838 vs 0.907).

### 6.8 Operational stack robustness (local reproduction)

Standing up the local observability stack (`make stack`) surfaced environment-specific failures,
each fixed durably in the repository rather than by manual workaround:

- **A `docker` wrapper first on `PATH` injected a `compose` subcommand**, so `docker compose up`
  became `compose compose` and every stack target failed. *Mitigation:* route the Makefile through
  the real CLI (`/usr/bin/docker`, overridable via `make stack DOCKER=…`).
- **Port collisions** (host 8000 and 3000 already bound) made service URLs appear dead. Every
  compose port is overridable (`DRIFTGUARD_APP_PORT`, `DRIFTGUARD_GRAFANA_PORT`, …), and the demo
  script now auto-selects a free port and verifies *its own* server came up.
- **MLflow could not open its sqlite backend** (`unable to open database file`): it runs as a
  non-root uid but its named volume mounted `/mlflow` root-owned. *Mitigation:* pre-create the
  backend directory owned by that uid in the image so a fresh named volume inherits writable
  ownership; give MLflow its own healthcheck (it had inherited the app's, against the wrong port).
- **A failed `make demo` left the repo broken** by deleting the untracked primary pointer (removed
  at the fallback step) without restoring it, which then failed the latency-breach test.
  *Mitigation:* restore the pointer from an `EXIT` trap, unconditionally.

Individually minor, together these illustrate the working discipline: reproduce the failure, fix the
root cause in-repo, and verify end to end — the full stack now comes up with app, MLflow, Prometheus,
and Grafana all healthy, and a provisioned **DriftGuard adaptation-governance dashboard** surfaces the
live serving-tier / fallback / latency-breach metrics alongside the measured recovery/retention.

---

## 7 · Outcomes and results

- **Text:** linear primary macro-F1 0.9197, DistilBERT 0.9412; shifted-sample PSI 12.52 vs
  stable 0.014; three-layer composite recall 1.00 at 0.00 FPR (0.71 before the descriptor-KS
  layer the head-to-head motivated); closed-loop recovery 0.968 / retention 0.926 with the dual
  gate correctly promoting genuine recovery — and the slice/calibration layer stating what that
  pass accepts (class-concentrated forgetting, 4× old-distribution ECE).
- **Tabular:** baseline 0.783 / primary 0.819; as covariate drift deepens, retention falls
  (0.936 → 0.728) and the dual gate flips PASS → FAIL.
- **Embeddings:** baseline 0.838 / primary 0.907; recovery ≈ 1.0 at all severities while
  retention collapses (0.993 → 0.606) and the gate refuses the "perfectly recovered" model.
- **Generalisation:** all three instances reuse the same governance and detector code; the
  detector layer carries **zero duplicated logic**; 50 automated tests, lint clean.

The three instances make the thesis unavoidable across model families: **a model can perfectly
relearn the drifted task and still be unpromotable because it has forgotten production** — which
is exactly what the forgetting-aware gate exists to catch.

---

## 8 · What we learned

- **Recovery ≠ safety** is the central, transferable idea; the forgetting-aware gate is the
  operational expression of it.
- **Two orthogonal detectors beat one.** PSI (length) and a domain classifier (semantics) cover
  disjoint failure modes; unioning them roughly doubles recall at no false-positive cost.
- **Governance generalises when it operates on scalar scores** — the gates and metrics need
  nothing about the model, so one implementation governs text, tabular, and embedding classifiers
  unchanged. Abstract at the right time: extraction was done once there were *two* instances.
- **Detection and promotion are separate concerns** and stay simpler kept decoupled.
- **Designing an honest experiment is itself research.** Half the embedding-instance effort went
  into finding a drift that produces a *meaningful* recovery/forgetting signal; the failed
  designs (noise, robustness, ill-conditioning, transform mismatch) were as instructive as the
  fix.
- **Process discipline matters as much as modelling:** verify environment assumptions, stage
  every touched file, and re-run the benchmark to *prove* parity rather than assert it.

---

## 9 · Limitations and threats to validity

- Detectors are **unsupervised proxies**, not guarantees; the operating thresholds (PSI 0.2,
  AUC 0.75) are conventional defaults, not per-deployment-tuned.
- Injected drift is **synthetic and controlled** — it isolates mechanisms cleanly but does not
  capture real production messiness (seasonality, feedback loops, label delay).
- Recovery/retention **divide small numbers under light drift**, so the recovery ratio is noisy
  at low severity; the system is healthy there regardless (retention high, gate passes).
- AWS infrastructure is **validated Terraform, not a live long-running deployment** — the
  operational track (§10) is future work.

---

## 10 · Future work

- **Operational track:** Terraform-applied EKS deployment, live Prometheus/Grafana + drift
  monitor, auto-rollback, and runbooks — turning the benchmark into a running service.
- **Real-world drift streams:** replace synthetic generators with temporally realistic and
  adversarial shift; report streaming latency at scale.
- **A held-out drift benchmark:** freeze a labelled drift suite so external methods can be scored
  on the same recovery/retention/gate axes — the path to a *citable* benchmark.
- **Richer detectors and gates:** conformal abstention, calibrated MMD thresholds, and cost-
  sensitive gates that price forgetting against recovery per deployment.

---

## 11 · Reproducibility and artifacts

| Artifact | What it backs |
|----------|---------------|
| `artifacts/metrics.json`, `artifacts/metrics_transformer.json` | text primary / DistilBERT quality |
| `benchmarks/results.json`, `results_streaming.json`, `results_recovery_sweep.json` | benchmark, streaming latency, recovery sweep |
| `examples/results_tabular.json`, `examples/results_embedding.json` | tabular / embedding instance sweeps |
| `deploy/monitoring/grafana/dashboards/*.json` | provisioned Grafana dashboards (adaptation governance + service health) |
| `notebooks/ag_news_drift_demo.ipynb` | executed, plotted demonstration of H1–H3 + generalisation |

Reproduce: `make install && make data && make train && make test`; `make benchmark`,
`make recovery`, `make recovery-sweep`, `make example-tabular`, `make example-embedding`,
`make train-transformer`. Seeds are fixed, dependencies locked, the data split versioned.

---

## 12 · Conclusion

DriftGuard operationalises a single discipline — **detect, retrain, and promote only what is
provably no worse than the incumbent** — and demonstrates it across three model families with
one governance layer. The investigation's honest record, including the approaches that failed and
the mitigations that resolved them, is part of the contribution: a benchmark is only citable if
its results, *and the path to them*, are reproducible and truthfully reported.
