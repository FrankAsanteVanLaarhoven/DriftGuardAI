"""Head-to-head drift-detection comparison: DriftGuard vs Evidently vs NannyML.

Same protocol as ``eval_harness``: the eight controlled drift generators, `seeds`
windows each sampled from the AG News test pool, ground truth = ``IS_DRIFT[kind]``.

Fairness rules:

* All tools share the **identical reference sample**: ``ref_size`` texts held out from
  the test pool with a fixed seed and never used for evaluation windows. (Using the
  committed train-side reference here would poison the protocol: reference texts drawn
  from the training corpus have OOV rate identically 0 against the training vocabulary
  while any test window sits near 2%, so every descriptor tool false-alarms on
  ``no_drift`` by construction.) DriftGuard's PSI reference is rebuilt from the same
  shared sample.
* Every comparator is a *tabular* drift tool, so each gets the **identical**
  five-column text-descriptor frame (token_count, char_count, mean_word_len, oov_rate,
  non_alpha_rate — mirroring Evidently's own text-descriptor defaults), with the OOV
  vocabulary taken from the training corpus (the natural production artifact).
  DriftGuard runs its composite detector on **raw text**, which is its design — that
  difference is the point of the comparison, and it is stated, not hidden.
* Each tool uses its **own native decision rule** — no external tuning:
  DriftGuard: any detector in the composite fires. Evidently: ``DataDriftPreset``'s
  dataset-level rule, drifted-column share >= 0.5 (per-column test auto-selected by
  Evidently; normed Wasserstein at these sample sizes). NannyML: any column alerts in
  any analysis chunk (Jensen-Shannon with its std-band thresholds; NannyML has no
  dataset-level rule of its own). ``ks_baseline``: any column significant under
  two-sample K-S with Bonferroni correction at alpha=0.05.
* ``ks_baseline`` stands in for Alibi Detect's ``KSDrift``, which applies the same
  per-feature K-S + Bonferroni scheme: alibi-detect 0.13.0 (its latest release) pins
  numba/llvmlite versions with no Python 3.13 support, so the package itself cannot
  be installed in this environment.
* Latency is wall time per window decision. NannyML's one-off reference fit is
  excluded; DriftGuard trains its domain classifier *inside* every window decision
  (it has no fit-ahead phase), so its latency carries that cost honestly.

Run:  uv run --extra bench python benchmarks/head_to_head.py [--seeds 5] [--window 600]
Writes benchmarks/results_head_to_head.json and prints Markdown tables.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("NML_DISABLE_USAGE_LOGGING", "1")  # no NannyML telemetry
os.environ.setdefault("DO_NOT_TRACK", "1")               # no Evidently telemetry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import drift_generators as gen  # noqa: E402
import pandas as pd  # noqa: E402
from eval_harness import _score  # noqa: E402

from driftguard import drift, textdrift  # noqa: E402
from driftguard.config import get_settings  # noqa: E402
from driftguard.data import load_split  # noqa: E402

TOOLS = ("driftguard", "evidently", "nannyml", "ks_baseline")

# Canonical implementations live in the product (driftguard.textdrift) since the
# composite absorbed the descriptor-KS layer this benchmark motivated; re-exported
# here so the harness and its tests keep one source of truth. The vocab handed to
# the comparators comes from the *training* corpus — never from the reference
# sample itself (see text_descriptors' docstring for why).
DESCRIPTOR_COLUMNS = textdrift.DESCRIPTOR_COLUMNS
reference_vocab = textdrift.reference_vocab
build_descriptors = textdrift.text_descriptors


def detect_driftguard(current_texts: list[str], ctx: dict) -> dict:
    t0 = time.perf_counter()
    res = textdrift.composite_drift(current_texts, ctx["reference_texts"],
                                    ctx["reference_dist"], ctx["settings"])
    return {"drift": bool(res["drift"]),
            "stat": res["signals"]["domain_classifier"]["auc"],
            "time_s": time.perf_counter() - t0}


def detect_evidently(cur_desc: pd.DataFrame, ctx: dict) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    t0 = time.perf_counter()
    snap = Report([DataDriftPreset()]).run(reference_data=ctx["ref_desc"],
                                           current_data=cur_desc)
    share = next(m["value"]["share"] for m in snap.dict()["metrics"]
                 if str(m.get("metric_name", "")).startswith("DriftedColumnsCount"))
    # Evidently's native dataset-level decision: drifted-column share >= drift_share
    # (0.5, the DataDriftPreset default).
    return {"drift": share >= 0.5, "stat": float(share),
            "time_s": time.perf_counter() - t0}


def fit_nannyml(ref_desc: pd.DataFrame, chunk_size: int):
    import nannyml as nml

    calc = nml.UnivariateDriftCalculator(
        column_names=list(ref_desc.columns),
        continuous_methods=["jensen_shannon"],
        chunk_size=chunk_size,
    )
    return calc.fit(ref_desc)


def detect_nannyml(cur_desc: pd.DataFrame, ctx: dict) -> dict:
    t0 = time.perf_counter()
    res = ctx["nannyml_calc"].calculate(cur_desc)
    df = res.to_df()
    analysis = df[df[("chunk", "chunk", "period")] == "analysis"]
    alert_cols = [c for c in analysis.columns if c[-1] == "alert"]
    alerts = analysis[alert_cols].fillna(False).astype(bool).to_numpy()
    return {"drift": bool(alerts.any()), "stat": float(alerts.mean()),
            "time_s": time.perf_counter() - t0}


def detect_ks_baseline(cur_desc: pd.DataFrame, ctx: dict) -> dict:
    from scipy.stats import ks_2samp

    t0 = time.perf_counter()
    pvals = [float(ks_2samp(ctx["ref_desc"][c], cur_desc[c]).pvalue)
             for c in cur_desc.columns]
    alpha = 0.05 / len(pvals)  # Bonferroni, as alibi-detect's KSDrift applies it
    return {"drift": any(p < alpha for p in pvals), "stat": min(pvals),
            "time_s": time.perf_counter() - t0}


DETECTORS = {
    "evidently": detect_evidently,
    "nannyml": detect_nannyml,
    "ks_baseline": detect_ks_baseline,
}


def run(seeds: int = 5, window: int = 600, chunk_size: int = 150,
        ref_size: int = 1500) -> dict:
    settings = get_settings()
    pool_all = load_split("test", settings)
    # Shared reference: a fixed-seed hold-out from the test pool, disjoint from every
    # evaluation window. All tools (DriftGuard included) reference this same sample.
    ref_idx = random.Random(777).sample(range(len(pool_all)), ref_size)
    reference_texts = pool_all["text"].iloc[ref_idx].tolist()
    pool = pool_all.drop(pool_all.index[ref_idx]).reset_index(drop=True)

    vocab = reference_vocab(load_split("train", settings)["text"].tolist())
    ref_desc = build_descriptors(reference_texts, vocab)
    ctx = {
        "settings": settings,
        "reference_texts": reference_texts,
        "reference_dist": drift.build_reference(reference_texts),
        "ref_desc": ref_desc,
        "nannyml_calc": fit_nannyml(ref_desc, chunk_size),
    }

    records: list[dict] = []
    rows: list[dict] = []
    times: dict[str, list[float]] = {t: [] for t in TOOLS}
    for kind, fn in gen.GENERATORS.items():
        detections: dict[str, list[bool]] = {t: [] for t in TOOLS}
        for s in range(seeds):
            rng = random.Random(1000 + s)  # same seeds as eval_harness
            current = fn(pool, window, rng)
            cur_desc = build_descriptors(current, vocab)
            results = {"driftguard": detect_driftguard(current, ctx)}
            for name, detector in DETECTORS.items():
                results[name] = detector(cur_desc, ctx)
            rec = {"is_drift": gen.IS_DRIFT[kind]}
            for name, r in results.items():
                rec[name] = r["drift"]
                detections[name].append(r["drift"])
                times[name].append(r["time_s"])
            records.append(rec)
        rows.append({"kind": kind, "is_drift": gen.IS_DRIFT[kind],
                     **{t: sum(detections[t]) / len(detections[t]) for t in TOOLS}})

    return {
        "seeds": seeds, "window": window, "nannyml_chunk_size": chunk_size,
        "ref_size": ref_size,
        "descriptors": list(DESCRIPTOR_COLUMNS),
        "rows": rows,
        "scorecard": {t: _score(records, t) for t in TOOLS},
        "mean_latency_s": {t: round(statistics.mean(times[t]), 4) for t in TOOLS},
    }


def to_markdown(summary: dict) -> str:
    lines = [
        f"Head-to-head (seeds={summary['seeds']}, window={summary['window']}, "
        f"shared descriptors: {', '.join(summary['descriptors'])}):",
        "",
        "| drift kind | is_drift | " + " | ".join(TOOLS) + " |",
        "|---|---|" + "---|" * len(TOOLS),
    ]
    for r in summary["rows"]:
        cells = " | ".join(f"{r[t]:.2f}" for t in TOOLS)
        lines.append(f"| {r['kind']} | {r['is_drift']} | {cells} |")
    lines += [
        "",
        "Scorecard (ground truth = is_drift, over every kind × seed; "
        "latency = mean s/window decision):",
        "",
        "| tool | precision | recall | F1 | FPR | s/window |",
        "|---|---|---|---|---|---|",
    ]
    for t in TOOLS:
        d = summary["scorecard"][t]
        lines.append(f"| {t} | {d['precision']:.2f} | {d['recall']:.2f} | "
                     f"{d['f1']:.2f} | {d['fpr']:.2f} | "
                     f"{summary['mean_latency_s'][t]:.3f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DriftGuard vs Evidently vs NannyML head-to-head drift benchmark")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--window", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=150,
                        help="NannyML reference/analysis chunk size.")
    parser.add_argument("--ref-size", type=int, default=1500,
                        help="Shared reference sample held out from the test pool.")
    args = parser.parse_args(argv)

    summary = run(args.seeds, args.window, args.chunk_size, args.ref_size)
    here = Path(__file__).resolve().parent
    out = here / "results_head_to_head.json"
    out.write_text(json.dumps(summary, indent=2))
    print(to_markdown(summary))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
