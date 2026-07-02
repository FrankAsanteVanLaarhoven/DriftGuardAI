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


def recovery_ratio(cand_drift_f1: float, stale_drift_f1: float, orig_clean_f1: float) -> float:
    """Fraction of the drift-induced accuracy loss the candidate regains on the *new*
    distribution. 1.0 = fully restored to the pre-drift clean level; 0.0 = no recovery."""
    denom = orig_clean_f1 - stale_drift_f1
    return (cand_drift_f1 - stale_drift_f1) / denom if denom > 1e-9 else 0.0


def retention_ratio(cand_fixed_f1: float, stale_fixed_f1: float) -> float:
    """Old-distribution performance kept after adapting (catastrophic-forgetting guard).
    1.0 = no forgetting; lower = more of the original distribution given up."""
    return cand_fixed_f1 / stale_fixed_f1 if stale_fixed_f1 > 1e-9 else 0.0


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
    t2 = time.perf_counter()
    stale_drift_f1 = _macro_f1(stale, dxte, yte)
    stale_fixed_f1 = _macro_f1(stale, xte, yte)
    cand_drift_f1 = _macro_f1(candidate, dxte, yte)
    cand_fixed_f1 = _macro_f1(candidate, xte, yte)
    base_drift_f1 = _macro_f1(baseline, dxte, yte)
    eval_time = time.perf_counter() - t2

    # Recovery metrics: how much of the drift-induced loss is regained on the new
    # distribution, how much of the old distribution is retained, and how long it took.
    orig_clean_f1 = stale_fixed_f1  # the pre-drift primary's clean-holdout score
    rec_ratio = recovery_ratio(cand_drift_f1, stale_drift_f1, orig_clean_f1)
    ret_ratio = retention_ratio(cand_fixed_f1, stale_fixed_f1)
    time_to_recovery = detection_time + retrain_time + eval_time

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
            "evaluate": round(eval_time, 3),
            "detection_to_decision": round(detection_time + retrain_time, 3),
        },
        "recovery": {
            "time_to_recovery_s": round(time_to_recovery, 3),
            "recovery_ratio": round(rec_ratio, 4),
            "retention_ratio": round(ret_ratio, 4),
            "orig_clean_macro_f1": round(orig_clean_f1, 4),
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
    m, t, rec = r["macro_f1"], r["timing_s"], r["recovery"]
    return "\n".join([
        f"Scenario: {r['scenario']} — detected={r['detected']} by {r['detected_by']} "
        f"(domain AUC {r['domain_auc']:.4f}, PSI {r['psi']:.4f})",
        "",
        f"Detection {t['detection']}s | retrain {t['retrain']}s | "
        f"detection→decision {t['detection_to_decision']}s",
        f"Time-to-recovery {rec['time_to_recovery_s']}s | "
        f"recovery ratio {rec['recovery_ratio']:.3f} | retention {rec['retention_ratio']:.3f}",
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


def sweep_p(ps: list[float], window: int = 600, seed: int = 42) -> dict:
    """Recovery vs drift severity: run the loop across a range of vocab-drift fractions."""
    rows = []
    for p in ps:
        r = run(p=p, window=window, seed=seed)
        rec = r["recovery"]
        rows.append({
            "p": p,
            "detected": r["detected"],
            "recovery_ratio": rec["recovery_ratio"],
            "retention_ratio": rec["retention_ratio"],
            "time_to_recovery_s": rec["time_to_recovery_s"],
            "recovery_delta_on_drift": r["macro_f1"]["recovery_delta_on_drift"],
            "gate_dual_passed": r["gate_dual_drift_aware"]["passed"],
        })
    return {"window": window, "seed": seed, "rows": rows}


def sweep_to_markdown(s: dict) -> str:
    lines = [
        f"Recovery vs drift severity (window={s['window']}, seed={s['seed']}):",
        "",
        "| p (vocab drift) | detected | recovery ratio | retention ratio | TTR (s) | dual gate |",
        "|---|---|---|---|---|---|",
    ]
    for r in s["rows"]:
        lines.append(
            f"| {r['p']:.2f} | {r['detected']} | {r['recovery_ratio']:.3f} | "
            f"{r['retention_ratio']:.3f} | {r['time_to_recovery_s']:.1f} | "
            f"{'PASS' if r['gate_dual_passed'] else 'FAIL'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DriftGuard closed-loop recovery measurement")
    parser.add_argument("--p", type=float, default=0.7, help="Fraction of vocabulary that drifts.")
    parser.add_argument("--window", type=int, default=600)
    parser.add_argument("--sweep-p", default=None,
                        help="Comma-separated p values for a recovery-vs-severity sweep.")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    if args.sweep_p:
        ps = [float(x) for x in args.sweep_p.split(",")]
        result = sweep_p(ps, args.window)
        out = here / "results_recovery_sweep.json"
        out.write_text(json.dumps(result, indent=2))
        print(sweep_to_markdown(result))
        print(f"\nWrote {out}")
        return 0

    result = run(args.p, args.window)
    out = here / "results_recovery.json"
    out.write_text(json.dumps(result, indent=2))
    print(to_markdown(result))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
