"""DriftGuard drift-injection benchmark.

Runs the composite detector (PSI + domain-classifier) over controlled drift windows
across multiple seeds and reports, per drift kind: detection rate, which detector
fired, and the mean PSI / domain-AUC. `no_drift` windows measure the false-positive
rate. Turns the qualitative "classifier catches what PSI misses" claim into numbers.

Run:  uv run python benchmarks/eval_harness.py [--seeds 5] [--window 600]
Writes benchmarks/results.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import drift_generators as gen  # noqa: E402

from driftguard import drift, textdrift  # noqa: E402
from driftguard.config import get_settings  # noqa: E402
from driftguard.data import load_split  # noqa: E402


def run(seeds: int = 5, window: int = 600) -> dict:
    settings = get_settings()
    reference_texts = textdrift.load_reference_texts(settings)
    reference_dist = drift.load_reference(settings)
    pool = load_split("test", settings)  # labelled base corpus

    rows = []
    for kind, fn in gen.GENERATORS.items():
        detections, psis, aucs, attributions = [], [], [], []
        for s in range(seeds):
            rng = random.Random(1000 + s)
            current = fn(pool, window, rng)
            result = textdrift.composite_drift(current, reference_texts, reference_dist, settings)
            detections.append(1 if result["drift"] else 0)
            psis.append(result["signals"]["psi"]["value"])
            aucs.append(result["signals"]["domain_classifier"]["auc"])
            attributions.extend(result["triggered_by"])
        n = len(detections)
        rows.append({
            "kind": kind,
            "is_drift": gen.IS_DRIFT[kind],
            "detection_rate": sum(detections) / n,
            "mean_psi": statistics.mean(psis),
            "mean_domain_auc": statistics.mean(aucs),
            "fired_psi": attributions.count("psi"),
            "fired_domain": attributions.count("domain_classifier"),
            "n_seeds": n,
        })

    drift_rows = [r for r in rows if r["is_drift"]]
    fpr_rows = [r for r in rows if not r["is_drift"]]
    summary = {
        "seeds": seeds,
        "window": window,
        "mean_detection_rate_on_drift": statistics.mean(r["detection_rate"] for r in drift_rows),
        "false_positive_rate_no_drift": (
            statistics.mean(r["detection_rate"] for r in fpr_rows) if fpr_rows else 0.0
        ),
        "rows": rows,
    }
    return summary


def to_markdown(summary: dict) -> str:
    lines = [
        f"Seeds={summary['seeds']}, window={summary['window']}. "
        f"Mean detection on drift={summary['mean_detection_rate_on_drift']:.2f}, "
        f"FPR(no_drift)={summary['false_positive_rate_no_drift']:.2f}",
        "",
        "| drift kind | is_drift | detection | mean PSI | mean AUC | PSI fired | domain fired |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['kind']} | {r['is_drift']} | {r['detection_rate']:.2f} | "
            f"{r['mean_psi']:.4f} | {r['mean_domain_auc']:.4f} | "
            f"{r['fired_psi']}/{r['n_seeds']} | {r['fired_domain']}/{r['n_seeds']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DriftGuard drift-injection benchmark")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--window", type=int, default=600)
    args = parser.parse_args(argv)

    summary = run(args.seeds, args.window)
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(to_markdown(summary))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
