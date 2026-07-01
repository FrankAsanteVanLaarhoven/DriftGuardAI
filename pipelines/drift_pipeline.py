"""ZenML drift -> retrain pipeline for DriftGuard.

Closed loop: collect recent request text -> compute PSI drift -> on breach, retrain
(registering a new candidate WITHOUT promoting) -> baseline gate (fail-closed) ->
canary -> HUMAN GATE -> promote the ``production`` alias.

ZenML is optional; without it the identical stages run as plain functions.

CLI::

    # detect only (fast; dry-run is the default)
    python pipelines/drift_pipeline.py --sample artifacts/current_shifted.json

    # detect -> retrain candidate (still not promoted; waits for the human gate)
    python pipelines/drift_pipeline.py --sample artifacts/current_shifted.json --retrain

    # detect -> retrain -> promote (human approval supplied)
    python pipelines/drift_pipeline.py --sample artifacts/current_shifted.json \
        --retrain --approve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftguard import drift, registry  # noqa: E402
from driftguard.config import get_settings  # noqa: E402

try:
    from zenml import pipeline, step

    _HAS_ZENML = True
except Exception:  # noqa: BLE001
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
def detect_drift(sample_path: str) -> dict:
    settings = get_settings()
    result = drift.run_check(Path(sample_path), settings)
    return {"psi": result["psi"], "status": result["status"], "drift": result["drift"],
            "threshold": result["threshold"]}


@step
def retrain_candidate() -> dict:
    """Retrain and register a NEW candidate version without promoting it."""
    settings = get_settings()
    # Record the model in production now, before the retrain overwrites metrics.json.
    incumbent_f1 = registry.current_primary_macro_f1(settings)
    os.environ["DRIFTGUARD_AUTO_PROMOTE"] = "0"
    get_settings.cache_clear()
    from driftguard import train

    exit_code = train.main()  # writes metrics.json + baseline_metrics.json, registers version
    metrics = json.loads(settings.metrics_path.read_text())
    baseline = json.loads(settings.baseline_metrics_path.read_text())
    gate = registry.incumbent_gate(metrics["macro_f1"], baseline["macro_f1"],
                                   incumbent_f1, settings.promotion_margin)
    return {"gate_passed": gate.passed and exit_code == 0, "reason": gate.reason,
            "candidate_version": registry.latest_version(settings)}


@step
def canary(candidate: dict) -> dict:
    """Lightweight canary: the candidate must have cleared the fail-closed gate.

    In a live setup this shifts a small traffic slice to the candidate and compares
    online metrics; here the offline holdout gate is the canary signal.
    """
    candidate["canary_ok"] = bool(candidate.get("gate_passed"))
    return candidate


@step
def human_gate_and_promote(candidate: dict, approved: bool) -> dict:
    settings = get_settings()
    can_promote = candidate.get("canary_ok") and approved
    if can_promote:
        registry.promote_version(candidate["candidate_version"], settings)
        candidate["promoted"] = True
    else:
        candidate["promoted"] = False
    candidate["approved"] = approved
    candidate["production_version"] = registry.production_version(settings)
    return candidate


@pipeline
def drift_pipeline(sample_path: str, do_retrain: bool, approved: bool) -> dict:
    drift_result = detect_drift(sample_path)
    if not drift_result["drift"] or not do_retrain:
        return {"drift": drift_result, "retrained": False}
    candidate = canary(retrain_candidate())
    decision = human_gate_and_promote(candidate, approved)
    return {"drift": drift_result, "retrained": True, "decision": decision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DriftGuard drift -> retrain pipeline")
    parser.add_argument("--sample", default="artifacts/current_shifted.json")
    parser.add_argument("--retrain", action="store_true",
                        help="On drift, retrain a candidate (not promoted).")
    parser.add_argument("--approve", action="store_true",
                        help="Supply human approval to promote a passing candidate.")
    args = parser.parse_args(argv)

    result = drift_pipeline(args.sample, args.retrain, args.approve)
    print(json.dumps(result, indent=2))

    d = result["drift"]
    if d["drift"]:
        print(f"\nDRIFT DETECTED (PSI {d['psi']:.4f} > {d['threshold']}).", file=sys.stderr)
        if not args.retrain:
            print("Dry-run: would trigger the retrain pipeline.", file=sys.stderr)
        elif not result["retrained"]:
            pass
        else:
            dec = result["decision"]
            if not dec["gate_passed"]:
                print("Candidate FAILED the promotion gate — NOT promoted (fail-closed).",
                      file=sys.stderr)
            elif not dec["promoted"]:
                print("Candidate passed gate + canary — awaiting HUMAN approval to promote.",
                      file=sys.stderr)
            else:
                print(f"Candidate promoted to production (version {dec['candidate_version']}).",
                      file=sys.stderr)
    else:
        print("\nNo drift. No action.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
