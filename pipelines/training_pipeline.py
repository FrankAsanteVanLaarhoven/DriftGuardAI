"""ZenML training pipeline for DriftGuard.

Wraps the reproducible training logic in :mod:`driftguard.train` as ZenML steps so
it can run inside a ZenML stack (with the MLflow experiment tracker). ZenML is an
optional extra (``pip install -e '.[mlops]'``); when it is not installed this module
still runs the identical pipeline as plain functions, so the critical path in
``make train`` never depends on the orchestrator being present.

Run:  python pipelines/training_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftguard import registry  # noqa: E402
from driftguard.config import get_settings  # noqa: E402
from driftguard.data import build_splits, load_split, write_splits  # noqa: E402

try:
    from zenml import pipeline, step

    _HAS_ZENML = True
except Exception:  # noqa: BLE001 - orchestrator is optional
    _HAS_ZENML = False

    def step(fn=None, **_kw):  # type: ignore[no-redef]
        def wrap(f):
            return f
        return wrap(fn) if fn else wrap

    def pipeline(fn=None, **_kw):  # type: ignore[no-redef]
        def wrap(f):
            return f
        return wrap(fn) if fn else wrap


@step
def load_data() -> dict:
    settings = get_settings()
    try:
        frames = {k: load_split(k, settings) for k in ("train", "val", "test")}
    except FileNotFoundError:
        frames = build_splits(settings)
        write_splits(frames, settings)
    return {k: (v["text"].tolist(), v["label"].tolist()) for k, v in frames.items()}


@step
def train_and_gate(data: dict) -> dict:
    settings = get_settings()
    xtr, ytr = data["train"]
    xte, yte = data["test"]

    baseline = registry.build_baseline_pipeline().fit(xtr, ytr)
    base_m = registry.evaluate(baseline, xte, yte)

    primary = registry.build_primary_pipeline().fit(xtr, ytr)
    prim_m = registry.evaluate(primary, xte, yte)

    gate = registry.baseline_gate(prim_m["macro_f1"], base_m["macro_f1"],
                                  settings.promotion_margin)
    registry.save_bundle(registry.make_bundle(baseline, "baseline", base_m, "baseline-1"),
                         settings.baseline_path)
    if gate.passed:
        registry.save_bundle(registry.make_bundle(primary, "primary", prim_m, "primary-zenml"),
                             settings.primary_path)
    return {"baseline": base_m, "primary": prim_m, "gate_passed": gate.passed,
            "reason": gate.reason}


@pipeline
def training_pipeline() -> dict:
    return train_and_gate(load_data())


def main() -> int:
    result = training_pipeline()
    # ZenML returns step artifacts lazily; fall back to a direct call for reporting.
    if _HAS_ZENML:
        print("ZenML training pipeline submitted. Inspect runs with `zenml pipeline runs list`.")
        return 0
    print("Baseline gate:", "PASS" if result["gate_passed"] else "FAIL", "-", result["reason"])
    print("Baseline:", result["baseline"], "| Primary:", result["primary"])
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
