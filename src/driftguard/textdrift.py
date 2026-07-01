"""Text-aware drift detection — a second layer beside token_count PSI.

PSI over ``token_count`` is a cheap, robust covariate-shift proxy, but it is blind to
*semantic* shift: two corpora can share an identical length distribution yet talk
about completely different things. This module adds detectors that read the words.

1. **Domain-classifier drift** (dependency-free, the core signal). Train a classifier
   to separate reference text (label 0) from current text (label 1) on TF-IDF n-grams
   and measure cross-validated ROC-AUC. AUC ≈ 0.5 ⇒ indistinguishable (no drift);
   AUC → 1.0 ⇒ the two corpora are easily told apart ⇒ drift. This is the
   domain-discriminator idea from Rabanser, Günnemann & Lipton (2019).

2. **Embedding-MMD drift** (optional, behind the ``embed`` extra). Sentence-transformer
   embeddings + a linear-kernel Maximum Mean Discrepancy.

A ``composite_drift`` combines PSI with the domain classifier: drift if *either* fires.

CLI::

    python -m driftguard.textdrift artifacts/current_shifted.json
    python -m driftguard.textdrift <sample.json> --reference artifacts/reference_sample.json
    python -m driftguard.textdrift <sample.json> --embed        # if sentence-transformers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from driftguard import drift
from driftguard.config import Settings, get_settings


def _balanced(reference: list[str], current: list[str], seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    n = min(len(reference), len(current))
    ref = list(rng.choice(reference, size=n, replace=False)) if len(reference) > n else reference
    cur = list(rng.choice(current, size=n, replace=False)) if len(current) > n else current
    return ref, cur


def domain_classifier_drift(reference_texts: list[str], current_texts: list[str],
                            seed: int = 42, threshold: float = 0.75) -> dict[str, Any]:
    """Reference-vs-current separability via cross-validated ROC-AUC."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    ref, cur = _balanced(reference_texts, current_texts, seed)
    x = ref + cur
    y = np.array([0] * len(ref) + [1] * len(cur))

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    folds = min(5, len(ref), len(cur))
    cv = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=seed)
    auc = float(np.mean(cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc")))
    return {
        "detector": "domain_classifier",
        "auc": auc,
        "threshold": threshold,
        "drift": auc >= threshold,
        "n_reference": len(ref),
        "n_current": len(cur),
    }


def embedding_mmd_drift(reference_texts: list[str], current_texts: list[str],
                        model_name: str, seed: int = 42) -> dict[str, Any] | None:
    """Optional: sentence-embedding MMD. Returns None if the extra is not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # noqa: BLE001 - optional extra
        return None
    ref, cur = _balanced(reference_texts, current_texts, seed)
    model = SentenceTransformer(model_name)
    er = model.encode(ref, normalize_embeddings=True, show_progress_bar=False)
    ec = model.encode(cur, normalize_embeddings=True, show_progress_bar=False)
    # Linear-kernel MMD^2 = ||mean(er) - mean(ec)||^2 for normalized embeddings.
    diff = er.mean(axis=0) - ec.mean(axis=0)
    mmd = float(np.dot(diff, diff))
    return {"detector": "embedding_mmd", "model": model_name, "mmd": mmd,
            "n_reference": len(ref), "n_current": len(cur)}


def composite_drift(current_texts: list[str], reference_texts: list[str],
                    reference_dist: dict[str, Any], settings: Settings | None = None,
                    use_embed: bool = False) -> dict[str, Any]:
    settings = settings or get_settings()
    psi = drift.compute_psi(current_texts, reference_dist)
    psi_status = drift.classify_psi(psi["psi"], settings.psi_threshold)
    dom = domain_classifier_drift(reference_texts, current_texts, settings.random_seed,
                                  settings.domain_auc_threshold)

    signals = {
        "psi": {"value": psi["psi"], "threshold": settings.psi_threshold,
                "drift": psi_status == "drift"},
        "domain_classifier": {"auc": dom["auc"], "threshold": dom["threshold"],
                              "drift": dom["drift"]},
    }
    if use_embed:
        emb = embedding_mmd_drift(current_texts, reference_texts, settings.embed_model,
                                  settings.random_seed)
        if emb is not None:
            signals["embedding_mmd"] = {"mmd": emb["mmd"]}

    drift_flag = any(s.get("drift") for s in signals.values() if "drift" in s)
    return {
        "signals": signals,
        "drift": drift_flag,
        "triggered_by": [k for k, s in signals.items() if s.get("drift")],
    }


def _read_texts(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("texts", [])
    return data


def load_reference_texts(settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    for candidate in (settings.reference_sample_path,
                      settings.artifacts_dir / "current_baseline.json"):
        if candidate.exists():
            return _read_texts(candidate)
    raise FileNotFoundError("No reference text sample found — run `make train`.")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    use_embed = "--embed" in argv
    ref_override = None
    if "--reference" in argv:
        ref_override = Path(argv[argv.index("--reference") + 1])
    positional = [a for a in argv if not a.startswith("--")
                  and (ref_override is None or a != str(ref_override))]
    if not positional:
        print("usage: python -m driftguard.textdrift <sample.json> "
              "[--reference ref.json] [--embed]", file=sys.stderr)
        return 2

    settings = get_settings()
    current = _read_texts(Path(positional[0]))
    reference_texts = _read_texts(ref_override) if ref_override else load_reference_texts(settings)
    reference_dist = drift.load_reference(settings)

    result = composite_drift(current, reference_texts, reference_dist, settings, use_embed)
    print(json.dumps(result, indent=2))
    if result["drift"]:
        print(f"DRIFT DETECTED by: {', '.join(result['triggered_by'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
