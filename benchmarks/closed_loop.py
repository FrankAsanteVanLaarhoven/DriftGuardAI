"""Closed-loop self-healing measurement for DriftGuard.

Simulates a *vocabulary concept drift* (a deterministic fraction of tokens acquire a
new surface form — "the words evolved"), then runs the full loop and measures it:

    detect (composite)  ->  retrain candidate on drifted labelled data  ->  baseline gate

Reported metrics:
* detection time, retrain time, total detection→decision wall time;
* macro-F1 of the stale primary vs the retrained candidate on BOTH the drifted holdout
  and the fixed holdout;
* the baseline-gate decision on the fixed holdout **and** on a drift-refreshed holdout.

The key governance finding this surfaces: under concept drift, a candidate that
recovers on the *new* distribution is **rejected by a gate that still scores it on the
*stale* fixed holdout** — the evaluation holdout must be refreshed alongside the model.

Run:  uv run python benchmarks/closed_loop.py [--p 0.7] [--window 600]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import f1_score  # noqa: E402

from driftguard import drift, registry, textdrift  # noqa: E402
from driftguard.config import get_settings  # noqa: E402
from driftguard.data import load_split  # noqa: E402


def _tok_hash(word: str) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % 100


def vocab_drift(text: str, p: float) -> str:
    """Deterministically append a new surface form to a fraction ``p`` of tokens."""
    cutoff = int(p * 100)
    return " ".join(w + "_v2" if _tok_hash(w) < cutoff else w for w in text.split())


def _macro_f1(pipeline, texts: list[str], labels: list[int]) -> float:
    return float(f1_score(labels, pipeline.predict(texts), average="macro"))


def run(p: float = 0.7, window: int = 600, seed: int = 42) -> dict:
    settings = get_settings()
    train = load_split("train", settings)
    test = load_split("test", settings)
    xtr, ytr = train["text"].tolist(), train["label"].tolist()
    xte, yte = test["text"].tolist(), test["label"].tolist()

    dxtr = [vocab_drift(t, p) for t in xtr]   # drifted, labelled retrain data
    dxte = [vocab_drift(t, p) for t in xte]   # drifted holdout

    stale = registry.load_bundle(settings.primary_path)["pipeline"]      # current production
    baseline = registry.load_bundle(settings.baseline_path)["pipeline"]  # committed fallback
    base_fixed_f1 = json.loads(settings.baseline_metrics_path.read_text())["macro_f1"]

    # --- 1. detect on a drifted traffic window ------------------------------
    reference_texts = textdrift.load_reference_texts(settings)
    reference_dist = drift.load_reference(settings)
    rng = random.Random(seed)
    dwindow = rng.sample(dxte, k=min(window, len(dxte)))
    t0 = time.perf_counter()
    det = textdrift.composite_drift(dwindow, reference_texts, reference_dist, settings)
    detection_time = time.perf_counter() - t0

    # --- 2. retrain a candidate on the drifted labelled data ----------------
    t1 = time.perf_counter()
    candidate = registry.build_primary_pipeline().fit(dxtr, ytr)
    retrain_time = time.perf_counter() - t1

    # --- 3. evaluate on both holdouts ---------------------------------------
    stale_drift_f1 = _macro_f1(stale, dxte, yte)
    stale_fixed_f1 = _macro_f1(stale, xte, yte)
    cand_drift_f1 = _macro_f1(candidate, dxte, yte)
    cand_fixed_f1 = _macro_f1(candidate, xte, yte)
    base_drift_f1 = _macro_f1(baseline, dxte, yte)

    # --- 4. gate on the stale fixed holdout vs a drift-refreshed holdout -----
    gate_fixed = registry.baseline_gate(cand_fixed_f1, base_fixed_f1, settings.promotion_margin)
    gate_refreshed = registry.baseline_gate(cand_drift_f1, base_drift_f1, settings.promotion_margin)
    # Drift-aware "dual" gate: adapt to the new distribution AND don't catastrophically
    # forget the old one. This is the safe resolution to the recovery block.
    gate_dual = registry.promotion_gate(
        candidate_fixed_f1=cand_fixed_f1, baseline_fixed_f1=base_fixed_f1,
        candidate_refreshed_f1=cand_drift_f1, baseline_refreshed_f1=base_drift_f1,
        margin=settings.promotion_margin, mode="dual",
        regression_floor=settings.gate_regression_floor,
    )

    return {
        "scenario": f"vocab_concept_drift(p={p})",
        "detected": det["drift"],
        "detected_by": det["triggered_by"],
        "domain_auc": det["signals"]["domain_classifier"]["auc"],
        "psi": det["signals"]["psi"]["value"],
        "timing_s": {
            "detection": round(detection_time, 3),
            "retrain": round(retrain_time, 3),
            "detection_to_decision": round(detection_time + retrain_time, 3),
        },
        "macro_f1": {
            "stale_on_drift": round(stale_drift_f1, 4),
            "candidate_on_drift": round(cand_drift_f1, 4),
            "recovery_delta_on_drift": round(cand_drift_f1 - stale_drift_f1, 4),
            "stale_on_fixed": round(stale_fixed_f1, 4),
            "candidate_on_fixed": round(cand_fixed_f1, 4),
        },
        "gate_fixed_holdout": {"passed": gate_fixed.passed, "reason": gate_fixed.reason},
        "gate_refreshed_holdout": {"passed": gate_refreshed.passed,
                                   "reason": gate_refreshed.reason},
        "gate_dual_drift_aware": {"passed": gate_dual.passed, "reason": gate_dual.reason,
                                  "regression_floor": settings.gate_regression_floor},
    }


def to_markdown(r: dict) -> str:
    m, t = r["macro_f1"], r["timing_s"]
    return "\n".join([
        f"Scenario: {r['scenario']} — detected={r['detected']} by {r['detected_by']} "
        f"(domain AUC {r['domain_auc']:.4f}, PSI {r['psi']:.4f})",
        "",
        f"Detection {t['detection']}s | retrain {t['retrain']}s | "
        f"detection→decision {t['detection_to_decision']}s",
        "",
        "| macro-F1               | stale primary | retrained candidate |",
        "|------------------------|---------------|---------------------|",
        f"| on DRIFTED holdout     | {m['stale_on_drift']:.4f}        | "
        f"{m['candidate_on_drift']:.4f} (Δ {m['recovery_delta_on_drift']:+.4f}) |",
        f"| on FIXED holdout       | {m['stale_on_fixed']:.4f}        | "
        f"{m['candidate_on_fixed']:.4f}              |",
        "",
        f"Gate FIXED holdout      : {'PASS' if r['gate_fixed_holdout']['passed'] else 'FAIL'}"
        f" — {r['gate_fixed_holdout']['reason']}",
        f"Gate REFRESHED holdout  : {'PASS' if r['gate_refreshed_holdout']['passed'] else 'FAIL'}"
        f" — {r['gate_refreshed_holdout']['reason']}",
        f"Gate DUAL (drift-aware) : {'PASS' if r['gate_dual_drift_aware']['passed'] else 'FAIL'}"
        f" — {r['gate_dual_drift_aware']['reason']}",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DriftGuard closed-loop recovery measurement")
    parser.add_argument("--p", type=float, default=0.7, help="Fraction of vocabulary that drifts.")
    parser.add_argument("--window", type=int, default=600)
    args = parser.parse_args(argv)

    result = run(args.p, args.window)
    out = Path(__file__).resolve().parent / "results_recovery.json"
    out.write_text(json.dumps(result, indent=2))
    print(to_markdown(result))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
