"""Dependency-free PSI drift detector (with an optional Evidently report).

The core signal is ``token_count`` (whitespace tokens per document), a robust,
model-agnostic covariate-shift proxy for text streams. The reference distribution
is fitted at train time and persisted to ``artifacts/reference.json``; at monitor
time we bin the current sample against the frozen reference edges and compute the
Population Stability Index.

PSI convention: <0.1 stable, 0.1–0.2 moderate, >0.2 action. The action threshold is
configurable via ``DRIFTGUARD_PSI_THRESHOLD``.

CLI::

    python -m driftguard.drift artifacts/current_shifted.json      # exit 1 on drift
    python -m driftguard.drift artifacts/current_shifted.json --evidently
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from driftguard.config import Settings, get_settings

EPS = 1e-6


def token_count_signal(texts: list[str]) -> np.ndarray:
    return np.array([len(t.split()) for t in texts], dtype=float)


def build_reference(texts: list[str], bins: int = 10) -> dict[str, Any]:
    """Fit reference bin edges (quantile) and per-bin proportions on reference text."""
    values = token_count_signal(texts)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(values, quantiles)).astype(float)
    if edges.size < 3:
        # Degenerate (near-constant) signal: fall back to a uniform span.
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
    # Open the outer edges so unseen extremes still fall in the end bins.
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=edges)
    proportions = counts / max(counts.sum(), 1)
    return {
        "signal": "token_count",
        "bin_edges": edges.tolist(),
        "reference_proportions": proportions.tolist(),
        "n": int(len(values)),
    }


def _proportions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.array(edges, dtype=float))
    return counts / max(counts.sum(), 1)


def compute_psi(current_texts: list[str], reference: dict[str, Any]) -> dict[str, Any]:
    edges = reference["bin_edges"]
    expected = np.array(reference["reference_proportions"], dtype=float)
    current = _proportions(token_count_signal(current_texts), edges)

    exp = np.clip(expected, EPS, None)
    cur = np.clip(current, EPS, None)
    per_bin = (cur - exp) * np.log(cur / exp)
    psi = float(np.sum(per_bin))
    return {
        "signal": reference["signal"],
        "psi": psi,
        "per_bin_psi": per_bin.tolist(),
        "current_proportions": current.tolist(),
        "reference_proportions": expected.tolist(),
        "n_current": int(len(current_texts)),
    }


def classify_psi(psi: float, threshold: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < threshold:
        return "moderate"
    return "drift"


def load_reference(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    return json.loads(settings.reference_path.read_text())


def _read_texts(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("texts", [])
    if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
        raise ValueError(f"{path} must be a JSON list of strings or {{'texts': [...]}}")
    return data


def evidently_report(current_texts: list[str], reference_texts: list[str],
                     out_html: Path) -> str | None:
    """Optional richer report. Returns the path written, or None if Evidently is absent."""
    try:
        import pandas as pd
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception:  # noqa: BLE001 - Evidently is an optional extra
        return None
    ref = pd.DataFrame({"token_count": token_count_signal(reference_texts)})
    cur = pd.DataFrame({"token_count": token_count_signal(current_texts)})
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref, current_data=cur)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out_html))
    return str(out_html)


def run_check(sample_path: Path, settings: Settings | None = None,
              use_evidently: bool = False) -> dict[str, Any]:
    settings = settings or get_settings()
    reference = load_reference(settings)
    current_texts = _read_texts(sample_path)
    result = compute_psi(current_texts, reference)
    result["threshold"] = settings.psi_threshold
    result["status"] = classify_psi(result["psi"], settings.psi_threshold)
    result["drift"] = result["status"] == "drift"
    if use_evidently:
        ref_path = settings.artifacts_dir / "current_baseline.json"
        ref_texts = _read_texts(ref_path) if ref_path.exists() else current_texts
        html = evidently_report(current_texts, ref_texts, settings.artifacts_dir / "evidently.html")
        result["evidently_report"] = html
    return result


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    use_evidently = "--evidently" in argv
    positional = [a for a in argv if not a.startswith("--")]
    if not positional:
        print("usage: python -m driftguard.drift <sample.json> [--evidently]", file=sys.stderr)
        return 2
    settings = get_settings()
    result = run_check(Path(positional[0]), settings, use_evidently=use_evidently)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_bin_psi", "current_proportions",
                                   "reference_proportions")}, indent=2))
    if result["drift"]:
        print(f"DRIFT DETECTED: PSI {result['psi']:.4f} > threshold {result['threshold']}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
