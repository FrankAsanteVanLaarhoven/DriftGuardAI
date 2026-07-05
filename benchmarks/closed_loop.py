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
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import f1_score  # noqa: E402

from driftguard import drift, registry, textdrift  # noqa: E402
from driftguard.config import get_settings  # noqa: E402
from driftguard.data import load_split  # noqa: E402

# Adaptation-safety metrics live in the model-agnostic governance layer; re-exported here
# so `from closed_loop import recovery_ratio` keeps working.
from driftguard.governance import (  # noqa: E402,F401
    promotion_decision_quality,
    recovery_ratio,
    retention_ratio,
    safe_promotion_oracle,
)


def _tok_hash(word: str) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % 100


def vocab_drift(text: str, p: float) -> str:
    """Deterministically append a new surface form to a fraction ``p`` of tokens."""
    cutoff = int(p * 100)
    return " ".join(w + "_v2" if _tok_hash(w) < cutoff else w for w in text.split())


def _macro_f1(pipeline, texts: list[str], labels: list[int]) -> float:
    return float(f1_score(labels, pipeline.predict(texts), average="macro"))


def run(p: float = 0.7, window: int = 600, seed: int = 42,
        train_sample: int | None = None,
        safety_retention_floor: float = 0.90) -> dict:
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
    # A seeded sub-sample of the drifted training set gives genuine per-seed model
    # variation (full-data retraining is deterministic); None uses all rows.
    if train_sample and train_sample < len(dxtr):
        idx = random.Random(10_000 + seed).sample(range(len(dxtr)), train_sample)
        fit_x, fit_y = [dxtr[i] for i in idx], [ytr[i] for i in idx]
    else:
        fit_x, fit_y = dxtr, ytr
    t1 = time.perf_counter()
    candidate = registry.build_primary_pipeline().fit(fit_x, fit_y)
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

    # Ground-truth safety label (benchmark-only oracle): the candidate must not regress
    # the incumbent on the new distribution AND must keep >= floor of its old-distribution
    # score. Gate decisions are scored against this in the sweep.
    safe = safe_promotion_oracle(cand_drift_f1, stale_drift_f1,
                                 cand_fixed_f1, stale_fixed_f1,
                                 retention_floor=safety_retention_floor)

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
        "safety": {"safe_to_promote": safe,
                   "retention_floor": safety_retention_floor,
                   "retention_ratio": round(ret_ratio, 4)},
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
        f"Ground-truth safety     : "
        f"{'SAFE' if r['safety']['safe_to_promote'] else 'UNSAFE'} to promote "
        f"(retention {r['safety']['retention_ratio']:.3f} vs floor "
        f"{r['safety']['retention_floor']:.2f})",
    ])


def sweep_p(ps: list[float], window: int = 600, seeds: int = 3,
            train_sample: int | None = 40000,
            safety_retention_floor: float = 0.90) -> dict:
    """Recovery vs drift severity across `seeds` seeds per point, reported as mean ± std.

    Each seed retrains on a different sub-sample of the drifted data (``train_sample``),
    so the recovery/retention figures carry genuine variation rather than a single number.
    Every trial is also labelled by the ground-truth safety oracle, and each gate mode's
    promote/block decisions are scored against it (promotion precision / recall / unsafe
    promotion rate) across the whole sweep.
    """
    gate_modes = ("fixed", "refreshed", "dual")
    decisions: dict[str, list[tuple[bool, bool]]] = {mode: [] for mode in gate_modes}
    rows = []
    for p in ps:
        recs, rets, ttrs, dual, safes = [], [], [], [], []
        for s in range(seeds):
            r = run(p=p, window=window, seed=1000 + s, train_sample=train_sample,
                    safety_retention_floor=safety_retention_floor)
            rec = r["recovery"]
            recs.append(rec["recovery_ratio"])
            rets.append(rec["retention_ratio"])
            ttrs.append(rec["time_to_recovery_s"])
            dual.append(r["gate_dual_drift_aware"]["passed"])
            safe = r["safety"]["safe_to_promote"]
            safes.append(safe)
            decisions["fixed"].append((r["gate_fixed_holdout"]["passed"], safe))
            decisions["refreshed"].append((r["gate_refreshed_holdout"]["passed"], safe))
            decisions["dual"].append((r["gate_dual_drift_aware"]["passed"], safe))
        rows.append({
            "p": p, "seeds": seeds,
            "recovery_ratio_mean": round(statistics.mean(recs), 4),
            "recovery_ratio_std": round(statistics.pstdev(recs), 4),
            "retention_ratio_mean": round(statistics.mean(rets), 4),
            "retention_ratio_std": round(statistics.pstdev(rets), 4),
            "time_to_recovery_s_mean": round(statistics.mean(ttrs), 2),
            "dual_gate_pass_fraction": round(sum(dual) / len(dual), 2),
            "safe_fraction": round(sum(safes) / len(safes), 2),
        })
    quality = {mode: promotion_decision_quality(decisions[mode]) for mode in gate_modes}
    for q in quality.values():
        for key in ("promotion_precision", "promotion_recall", "unsafe_promotion_rate"):
            if q[key] is not None:
                q[key] = round(q[key], 4)
    return {"window": window, "seeds": seeds, "train_sample": train_sample,
            "safety_retention_floor": safety_retention_floor, "rows": rows,
            "decision_quality": quality}


def sweep_to_markdown(s: dict) -> str:
    lines = [
        f"Recovery vs drift severity (window={s['window']}, {s['seeds']} seeds, "
        f"retrain sub-sample={s['train_sample']}):",
        "",
        "| p (vocab drift) | recovery ratio (mean±std) | retention ratio (mean±std) "
        "| TTR (s) | dual gate (pass frac) | safe frac |",
        "|---|---|---|---|---|---|",
    ]
    for r in s["rows"]:
        lines.append(
            f"| {r['p']:.2f} | {r['recovery_ratio_mean']:.3f} ± {r['recovery_ratio_std']:.3f} "
            f"| {r['retention_ratio_mean']:.3f} ± {r['retention_ratio_std']:.3f} "
            f"| {r['time_to_recovery_s_mean']:.1f} | {r['dual_gate_pass_fraction']:.2f} "
            f"| {r['safe_fraction']:.2f} |"
        )
    if "decision_quality" in s:
        lines += [
            "",
            f"Promotion decision quality vs ground-truth safety oracle "
            f"(retention floor {s['safety_retention_floor']:.2f}, all trials):",
            "",
            "| gate mode | promotions | unsafe promotions | promotion precision "
            "| promotion recall | unsafe promotion rate |",
            "|---|---|---|---|---|---|",
        ]
        for mode, q in s["decision_quality"].items():
            fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
            lines.append(
                f"| {mode} | {q['promotions']}/{q['trials']} | {q['unsafe_promotions']} "
                f"| {fmt(q['promotion_precision'])} | {fmt(q['promotion_recall'])} "
                f"| {fmt(q['unsafe_promotion_rate'])} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DriftGuard closed-loop recovery measurement")
    parser.add_argument("--p", type=float, default=0.7, help="Fraction of vocabulary that drifts.")
    parser.add_argument("--window", type=int, default=600)
    parser.add_argument("--sweep-p", default=None,
                        help="Comma-separated p values for a recovery-vs-severity sweep.")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Seeds per sweep point (mean ± std).")
    parser.add_argument("--train-sample", type=int, default=40000,
                        help="Drifted-data retrain sub-sample per seed (for variation).")
    parser.add_argument("--safety-retention-floor", type=float, default=0.90,
                        help="Ground-truth oracle: min share of the incumbent's "
                             "original-distribution score a safe candidate must keep.")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    if args.sweep_p:
        ps = [float(x) for x in args.sweep_p.split(",")]
        result = sweep_p(ps, args.window, args.seeds, args.train_sample,
                         args.safety_retention_floor)
        out = here / "results_recovery_sweep.json"
        out.write_text(json.dumps(result, indent=2))
        print(sweep_to_markdown(result))
        print(f"\nWrote {out}")
        return 0

    result = run(args.p, args.window,
                 safety_retention_floor=args.safety_retention_floor)
    out = here / "results_recovery.json"
    out.write_text(json.dumps(result, indent=2))
    print(to_markdown(result))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
