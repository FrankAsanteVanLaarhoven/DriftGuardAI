"""Training entrypoint: baseline + primary, MLflow tracking/registry, baseline gate.

Produces the committed artifacts the rest of the system depends on:

* ``models/baseline.joblib``        — always-loadable fallback model
* ``artifacts/baseline_metrics.json`` — the evaluative gate reference (holdout)
* ``artifacts/metrics.json``        — primary metrics on the same frozen holdout
* ``artifacts/reference.json``      — drift reference distribution
* ``artifacts/primary.joblib`` + ``models/primary_pointer`` — the served primary
* ``artifacts/deployment_report.md`` — one-page change record
* ``artifacts/current_baseline.json`` / ``current_shifted.json`` — drift demo samples

The primary is promoted (MLflow ``production`` alias) only if it passes the
fail-closed baseline gate on the holdout.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from driftguard import drift, registry
from driftguard.config import Settings, get_settings
from driftguard.data import build_splits, load_split, write_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("driftguard.train")


def _load_or_build(settings: Settings):
    try:
        return {k: load_split(k, settings) for k in ("train", "val", "test")}
    except FileNotFoundError:
        log.info("Processed splits missing; building from ag_news…")
        frames = build_splits(settings)
        write_splits(frames, settings)
        return frames


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


_FOREIGN_VOCAB = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed eiusmod tempor "
    "incididunt labore dolore magna aliqua enim minim veniam quis nostrud"
).split()


def _demo_samples(settings: Settings, test_texts: list[str]) -> None:
    """Write demo samples for the drift monitors.

    * current_baseline    — in-distribution (no drift)
    * current_shifted     — token_count shift (PSI catches it)
    * current_semantic    — SAME length distribution, disjoint vocabulary
                            (PSI misses it; the domain-classifier detector catches it)
    """
    rng = __import__("random").Random(settings.random_seed)
    sample = rng.sample(test_texts, k=min(1000, len(test_texts)))
    _write_json(settings.artifacts_dir / "current_baseline.json", sample)

    shifted = [" ".join(t.split()[:5]) for t in sample]
    _write_json(settings.artifacts_dir / "current_shifted.json", shifted)

    semantic = [" ".join(rng.choice(_FOREIGN_VOCAB) for _ in range(len(t.split()) or 1))
                for t in sample]
    _write_json(settings.artifacts_dir / "current_semantic_shift.json", semantic)


def _deployment_report(settings: Settings, base_m, prim_m, gate, mlflow_info) -> None:
    base_line = f"accuracy {base_m['accuracy']:.4f}, macro-F1 {base_m['macro_f1']:.4f}"
    prim_line = f"accuracy {prim_m['accuracy']:.4f}, macro-F1 {prim_m['macro_f1']:.4f}"
    gate_line = f"{'PASS' if gate.passed else 'FAIL (fail-closed)'} — {gate.reason}"
    version = f"{mlflow_info.get('model_version', 'n/a')} (promoted={mlflow_info.get('promoted')})"
    report = f"""# DriftGuard deployment report

- **Date:** {datetime.now(UTC).isoformat()}
- **Change:** retrain primary text classifier (ag_news, seed {settings.random_seed})
- **Baseline (fallback) holdout:** {base_line}
- **Primary (candidate) holdout:** {prim_line}
- **Promotion gate (no-worse-than-incumbent):** {gate_line}
- **MLflow run:** {mlflow_info.get('run_id', 'n/a')}
- **Registered version:** {version}
- **Tests:** run `make test` (unit + integration + fallback) — must be green.
- **Rollback:** service `kubectl rollout undo`; model — move the `production` alias to
  the previous registry version.
"""
    (settings.artifacts_dir / "deployment_report.md").write_text(report)


def main() -> int:
    settings = get_settings()
    settings.ensure_dirs()

    # Capture the incumbent primary's score *before* we overwrite metrics.json, so the
    # promotion gate can refuse to replace a better model already in production.
    incumbent_f1 = registry.current_primary_macro_f1(settings)

    frames = _load_or_build(settings)
    Xtr, ytr = frames["train"]["text"].tolist(), frames["train"]["label"].tolist()
    Xte, yte = frames["test"]["text"].tolist(), frames["test"]["label"].tolist()

    # --- Baseline (fallback + evaluative reference) --------------------------
    log.info("Training baseline (fallback) model…")
    baseline = registry.build_baseline_pipeline()
    baseline.fit(Xtr, ytr)
    base_m = registry.evaluate(baseline, Xte, yte)
    registry.save_bundle(
        registry.make_bundle(baseline, "baseline", base_m, version="baseline-1"),
        settings.baseline_path,
    )
    _write_json(settings.baseline_metrics_path, base_m)
    log.info("Baseline holdout: acc=%.4f macro_f1=%.4f", base_m["accuracy"], base_m["macro_f1"])

    # --- Primary (candidate) -------------------------------------------------
    log.info("Training primary (candidate) model…")
    primary = registry.build_primary_pipeline()
    primary.fit(Xtr, ytr)
    prim_m = registry.evaluate(primary, Xte, yte)
    _write_json(settings.metrics_path, prim_m)
    log.info("Primary holdout:  acc=%.4f macro_f1=%.4f", prim_m["accuracy"], prim_m["macro_f1"])

    # --- Promotion gate (fail-closed, no-worse-than-incumbent) ---------------
    gate = registry.incumbent_gate(prim_m["macro_f1"], base_m["macro_f1"],
                                   incumbent_f1, settings.promotion_margin)
    log.info("Promotion gate: %s — %s", "PASS" if gate.passed else "FAIL", gate.reason)

    # --- Drift reference + demo samples --------------------------------------
    reference = drift.build_reference(Xtr, bins=settings.psi_bins)
    reference["class_distribution"] = (
        frames["train"]["label_name"].value_counts(normalize=True).round(4).to_dict()
    )
    _write_json(settings.reference_path, reference)
    # Raw reference-text sample for the text-aware (domain-classifier) drift detector.
    ref_rng = __import__("random").Random(settings.random_seed + 1)
    _write_json(settings.reference_sample_path, ref_rng.sample(Xtr, k=min(1500, len(Xtr))))
    _demo_samples(settings, Xte)

    # --- MLflow tracking + registry ------------------------------------------
    mlflow_info: dict[str, str] = {}
    try:
        mlflow_info = registry.log_and_register(
            primary,
            params={"model": "tfidf(1,2)+logreg", "seed": settings.random_seed,
                    "max_features": 50000},
            metrics={"accuracy": prim_m["accuracy"], "macro_f1": prim_m["macro_f1"],
                     "baseline_macro_f1": base_m["macro_f1"]},
            settings=settings,
            promote=gate.passed and settings.auto_promote,
        )
        log.info("MLflow run %s registered version %s (promoted=%s)",
                 mlflow_info.get("run_id"), mlflow_info.get("model_version"),
                 mlflow_info.get("promoted"))
    except Exception as exc:  # noqa: BLE001 - tracking must not lose a trained model
        log.warning("MLflow logging failed (continuing, model still saved locally): %s", exc)

    # --- Persist the served primary only when the gate passes ----------------
    if gate.passed:
        registry.save_bundle(
            registry.make_bundle(primary, "primary", prim_m,
                                 version=mlflow_info.get("model_version", "local")),
            settings.primary_path,
        )
        settings.primary_pointer_path.parent.mkdir(parents=True, exist_ok=True)
        settings.primary_pointer_path.write_text(
            str(settings.primary_path.relative_to(settings.primary_path.parents[1]))
        )
        log.info("Primary promoted → %s (pointer %s)", settings.primary_path,
                 settings.primary_pointer_path)
    else:
        log.error("Baseline gate FAILED — primary NOT promoted. Service keeps prior primary.")

    _deployment_report(settings, base_m, prim_m, gate, mlflow_info)

    if not gate.passed:
        return 1  # fail closed for CI
    return 0


if __name__ == "__main__":
    sys.exit(main())
