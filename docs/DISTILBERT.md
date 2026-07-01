# DistilBERT primary — GPU runbook

The SOTA option from the spec: a **DistilBERT primary** with the linear TF-IDF model as
the **fallback**. If the transformer OOMs, fails to load, or breaches its per-request
latency budget, the service degrades to the fast classic model instead of going down —
the fallback contract already enforces this, so **no serving code changes are needed**.

> This was **not executed in the CPU-only build environment**. Run it on a GPU host and
> record the measured numbers in `CASE_STUDY.md`. Do not copy the illustrative range
> below as if it were measured here.

## 1. Install the transformer extra

```bash
uv sync --extra transformer          # adds torch + transformers
uv run python -c "import torch; print('cuda', torch.cuda.is_available())"
```

## 2. Build data (once) and fine-tune

```bash
make data
# quick smoke on a subset first:
uv run python scripts/train_distilbert.py --epochs 1 --max-train-rows 8000
# full run:
uv run python scripts/train_distilbert.py --epochs 2 --promote
```

What the script does:
1. Loads the identical seeded ag_news splits.
2. Fine-tunes `distilbert-base-uncased` (4 labels).
3. Evaluates on the frozen test holdout and **runs the baseline gate** against the
   committed linear baseline — fail-closed, exactly like the linear primary.
4. Saves a bundle to `artifacts/primary_transformer.joblib`. With `--promote` (and a
   passing gate) it points the service primary at that bundle.

Published DistilBERT/BERT results on AG News land around **~0.94–0.95 accuracy**
(Zhang et al. 2015 and later transformer baselines) — treat that as the *target to
verify*, not a claim.

## 3. Serve it (DistilBERT primary, linear baseline fallback)

```bash
# The serving container needs the transformer extra in its image; point the primary
# at the transformer bundle (or the MLflow registry) and keep the latency budget:
DRIFTGUARD_PRIMARY_LATENCY_BUDGET_MS=750 make run
```

- `/predict` tries DistilBERT; if it exceeds the latency budget or throws, the request
  is served by the linear baseline and tagged `served_by:"baseline"` (metric increments).
- `/ready` stays 200 as long as the baseline can serve — a heavy/broken transformer
  never takes the pod out of rotation.
- `driftguard_model_tier{tier="baseline"}==1 for 5m` pages if it runs degraded.

## 4. Validate under drift (drift-aware gate)

Because the transformer bundle exposes the same `predict`/`predict_proba` surface, the
closed-loop harness and the drift-aware `dual` gate work unchanged:

```bash
DRIFTGUARD_GATE_HOLDOUT_MODE=dual make recovery
```

## Notes
- CPU inference for DistilBERT is much slower than the linear model — set the latency
  budget deliberately; breaches fall back to the baseline by design.
- The bundle embeds the torch model via joblib; the serving image must install the
  `transformer` extra. Keep the linear `models/baseline.joblib` committed as the
  always-loadable floor.

---

# Post-run checklist

Run through this after `scripts/train_distilbert.py` finishes.

## 1 · Interpret the training output

The script prints, in order:

```
Fine-tuning distilbert-base-uncased for 2 epoch(s) on 108000 rows…
DistilBERT holdout: acc=0.94xx macro_f1=0.94xx
Incumbent primary macro_f1: 0.9197
Promotion gate: PASS — candidate macro-F1 0.94xx >= max(baseline 0.8956, incumbent 0.9197) + margin 0.0000 (= 0.9197; bar set by incumbent primary)
Saved bundle -> .../artifacts/primary_transformer.joblib
Promoted: pointer -> .../models/primary_pointer          # only with --promote AND a pass
```

- **`macro_f1`** is the number that matters — it must clear **`max(baseline 0.8956,
  incumbent primary 0.9197)`**, not just the baseline. The gate is *no-worse-than-
  incumbent*: a DistilBERT that beats 0.8956 but scores below the linear primary already
  serving (0.9197) is **rejected**, because promoting it would be a downgrade (slower
  *and* less accurate). This is why the full GPU run — targeting ~0.94 — matters; a weak
  CPU run at ~0.91 will (correctly) fail to promote.
- The DistilBERT **"LOAD REPORT" listing `UNEXPECTED`/`MISSING` weights is normal** — the
  MLM checkpoint's head is dropped and a fresh classification head is initialised. Not an
  error.
- **Exit code**: `0` = gate passed (promotable), `1` = gate failed (fail-closed). Check
  with `echo $?`.

## 2 · Confirm the gate promotes / rejects correctly

```bash
# (a) Normal run — should PASS only if DistilBERT beats the incumbent primary (0.9197),
#     not merely the baseline (0.8956):
uv run --extra transformer python scripts/train_distilbert.py --epochs 2 --promote; echo "exit=$?"

# (b) Prove fail-closed — an impossible margin must REJECT and NOT promote:
DRIFTGUARD_PROMOTION_MARGIN=0.5 uv run --extra transformer \
  python scripts/train_distilbert.py --epochs 2 --promote; echo "exit=$?"   # expect exit=1
```

- A weak (e.g. CPU-subsampled) DistilBERT scoring **between** the baseline and the
  incumbent primary — say 0.91 — should print `Promotion gate: FAIL … bar set by
  incumbent primary` and exit `1`. That is the no-worse-than-incumbent gate doing its job.

- After a passing `--promote`, confirm the pointer moved:
  ```bash
  cat models/primary_pointer          # -> artifacts/primary_transformer.joblib
  ```
- Under concept drift, validate with the drift-aware gate (adapt + no catastrophic
  forgetting):
  ```bash
  DRIFTGUARD_GATE_HOLDOUT_MODE=dual make recovery
  ```

## 3 · Verify fallback + readiness still hold

Serving the DistilBERT bundle needs torch, so start the API with the extra:

```bash
uv run --extra transformer uvicorn driftguard.api.main:app --host 127.0.0.1 --port 8010
```

```bash
curl -s localhost:8010/ready                       # {"ready":true, ...}  (200)
curl -s localhost:8010/model-info                  # active_tier=primary, source=pointer:...primary_transformer.joblib
curl -s -X POST localhost:8010/predict -H 'content-type: application/json' \
  -d '{"text":"New GPU sets an on-device AI record."}'   # served_by:"primary"
```

Then prove graceful degradation (all must keep returning **HTTP 200**):

```bash
# (a) Latency budget breach -> baseline serves:
DRIFTGUARD_PRIMARY_LATENCY_BUDGET_MS=1 uv run --extra transformer \
  uvicorn driftguard.api.main:app --port 8010     # every DistilBERT call is "too slow"
#   -> /predict returns served_by:"baseline"; driftguard_primary_latency_breach_total climbs.

# (b) Primary artifact pulled -> baseline serves, /ready stays 200, tier gauge flips:
rm models/primary_pointer
curl -s localhost:8010/predict -X POST -H 'content-type: application/json' -d '{"text":"x"}'  # served_by:"baseline"
curl -s localhost:8010/metrics | grep '^driftguard_model_tier'   # tier="baseline" 1.0
printf 'artifacts/primary_transformer.joblib' > models/primary_pointer   # restore

# (c) No torch in the serving env -> primary can't load -> baseline still serves (degraded, up):
uv run uvicorn driftguard.api.main:app --port 8010   # without --extra transformer
curl -s localhost:8010/ready                          # still 200; runs on the linear baseline
```

**Green means:** `/ready` never dropped below 200, `/predict` always returned 200, and any
DistilBERT problem (slow, missing, un-loadable) degraded to the committed linear baseline —
the operational fallback contract holds with the transformer primary in place.

## 4 · Record the numbers
Add the measured DistilBERT `accuracy` / `macro_f1` (and the gate outcome) to
`CASE_STUDY.md`. Real numbers only — no estimates.
