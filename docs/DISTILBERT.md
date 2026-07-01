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
