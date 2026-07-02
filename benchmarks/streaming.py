"""DriftGuard streaming drift benchmark — detection latency over a change point.

The static harness (``eval_harness.py``) answers "does the detector fire on a drifted
window?". This answers the streaming question a citable drift benchmark reports: **how
fast** does it fire after drift begins, how often does it miss, and how many false alarms
does it raise before anything changed — across the standard drift-pattern taxonomy of
Gama et al. (2014):

    abrupt        in-distribution, then a sudden full-severity switch at the change point
    gradual       the drifted regime appears with a probability that ramps 0 -> 1
    incremental   severity ramps continuously 0 -> max after the change point
    recurring     drift comes and goes in alternating blocks (seasonality)

For each pattern the composite detector (PSI + domain classifier) is run over a stream of
windows and we report, averaged over seeds:

* detection delay — windows from the change point to the first true alarm;
* missed-detection rate — fraction of seeds that never alarmed post-change;
* false-alarm rate — alarms in the pre-change (in-distribution) segment;
* post-change detection rate — fraction of post-change windows that alarmed.

Run:  uv run python benchmarks/streaming.py [--kind semantic_replace] [--seeds 3]
Writes benchmarks/results_streaming.json and prints a Markdown table.
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

PATTERNS = ("abrupt", "gradual", "incremental", "recurring")
_RECUR_BLOCK = 3


def _window_texts(pattern: str, w_idx: int, cp: int, band: int, pool, window: int,
                  rng: random.Random, kind: str, max_sev: float) -> list[str]:
    """Generate one window of the stream for the given temporal drift pattern."""
    fn = gen.GENERATORS[kind]
    if w_idx < cp:
        return gen.no_drift(pool, window, rng)
    t = w_idx - cp  # windows since the change point
    if pattern == "abrupt":
        return fn(pool, window, rng, severity=max_sev)
    if pattern == "incremental":
        progress = min(1.0, (t + 1) / band)
        return fn(pool, window, rng, severity=max_sev * progress)
    if pattern == "gradual":
        p = min(1.0, (t + 1) / band)
        return fn(pool, window, rng, severity=max_sev) if rng.random() < p \
            else gen.no_drift(pool, window, rng)
    if pattern == "recurring":
        drifted = (t // _RECUR_BLOCK) % 2 == 0
        return fn(pool, window, rng, severity=max_sev) if drifted \
            else gen.no_drift(pool, window, rng)
    raise ValueError(f"unknown pattern: {pattern!r}")


def run(kind: str = "semantic_replace", n_windows: int = 16, change_point: int = 6,
        window: int = 400, seeds: int = 3, band: int = 6, max_sev: float = 0.6,
        patterns: tuple[str, ...] = PATTERNS) -> dict:
    settings = get_settings()
    reference_texts = textdrift.load_reference_texts(settings)
    reference_dist = drift.load_reference(settings)
    pool = load_split("test", settings)

    out = []
    for pattern in patterns:
        delays, missed, false_rates, post_rates = [], [], [], []
        for s in range(seeds):
            rng = random.Random(3000 + s)
            fired = []
            for w in range(n_windows):
                texts = _window_texts(pattern, w, change_point, band, pool, window,
                                      rng, kind, max_sev)
                res = textdrift.composite_drift(texts, reference_texts, reference_dist, settings)
                fired.append(bool(res["drift"]))
            pre, post = fired[:change_point], fired[change_point:]
            false_rates.append(sum(pre) / len(pre) if pre else 0.0)
            post_rates.append(sum(post) / len(post) if post else 0.0)
            first = next((i for i, f in enumerate(post) if f), None)
            if first is None:
                missed.append(1)
            else:
                missed.append(0)
                delays.append(first)
        out.append({
            "pattern": pattern,
            "detection_delay_windows": round(statistics.mean(delays), 2) if delays else None,
            "missed_detection_rate": round(statistics.mean(missed), 2),
            "false_alarm_rate_prechange": round(statistics.mean(false_rates), 3),
            "post_change_detection_rate": round(statistics.mean(post_rates), 2),
            "n_detected_seeds": len(delays),
        })
    return {
        "kind": kind, "n_windows": n_windows, "change_point": change_point,
        "window": window, "seeds": seeds, "band": band, "max_severity": max_sev,
        "detection_delay_unit": "windows after change point",
        "patterns": out,
    }


def to_markdown(s: dict) -> str:
    lines = [
        f"Streaming drift benchmark — kind=`{s['kind']}`, {s['n_windows']} windows "
        f"(change point @ {s['change_point']}), window={s['window']}, seeds={s['seeds']}. "
        f"Delay unit: {s['detection_delay_unit']}.",
        "",
        "| pattern | detection delay | missed rate | false-alarm rate (pre) "
        "| post-change detection |",
        "|---|---|---|---|---|",
    ]
    for r in s["patterns"]:
        d = r["detection_delay_windows"]
        delay = "—" if d is None else f"{d:.2f}"
        lines.append(
            f"| {r['pattern']} | {delay} | {r['missed_detection_rate']:.2f} | "
            f"{r['false_alarm_rate_prechange']:.3f} | {r['post_change_detection_rate']:.2f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DriftGuard streaming drift benchmark")
    parser.add_argument("--kind", default="semantic_replace", choices=list(gen.GENERATORS))
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--change-point", type=int, default=6)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--band", type=int, default=6)
    parser.add_argument("--max-severity", type=float, default=0.6)
    args = parser.parse_args(argv)

    result = run(kind=args.kind, n_windows=args.windows, change_point=args.change_point,
                 window=args.window, seeds=args.seeds, band=args.band,
                 max_sev=args.max_severity)
    out = Path(__file__).resolve().parent / "results_streaming.json"
    out.write_text(json.dumps(result, indent=2))
    print(to_markdown(result))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
