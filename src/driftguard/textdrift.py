"""Text-aware drift detection — a second layer beside token_count PSI.

PSI over ``token_count`` is a cheap, robust covariate-shift proxy, but it is blind to
*semantic* shift: two corpora can share an identical length distribution yet talk
about completely different things. This module adds detectors that read the words.

1. **Domain-classifier drift** (dependency-free, the core signal). Train a classifier
   to separate reference text (label 0) from current text (label 1) on TF-IDF n-grams
   and measure cross-validated ROC-AUC. AUC ≈ 0.5 ⇒ indistinguishable (no drift);
   AUC → 1.0 ⇒ the two corpora are easily told apart ⇒ drift. This is the
   domain-discriminator idea from Rabanser, Günnemann & Lipton (2019).

2. **Descriptor-KS drift** (dependency-free). Bonferroni-corrected two-sample K-S over
   five text descriptors (token_count, char_count, mean_word_len, oov_rate,
   non_alpha_rate). Absorbed from the head-to-head benchmark, where exactly this
   classical test beat the learned composite on descriptor-visible drift
   (``benchmarks/head_to_head.py``) — it closes the composite's ``gradual_topic`` /
   ``char_noise`` recall gap at zero false-positive cost.

3. **Embedding-MMD drift** (optional, behind the ``embed`` extra). Sentence-transformer
   embeddings + a linear-kernel Maximum Mean Discrepancy.

A ``composite_drift`` combines PSI, the domain classifier, and descriptor-KS:
drift if *any* fires (default rule).

CLI::

    python -m driftguard.textdrift artifacts/current_shifted.json
    python -m driftguard.textdrift <sample.json> --reference artifacts/reference_sample.json
    python -m driftguard.textdrift <sample.json> --embed        # if sentence-transformers
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from driftguard import drift
from driftguard.config import Settings, get_settings
from driftguard.detectors import (
    DescriptorKSDetector,
    DomainClassifierDetector,
    MMDDetector,
    PSIDetector,
)

log = logging.getLogger("driftguard.textdrift")

DESCRIPTOR_COLUMNS = ("token_count", "char_count", "mean_word_len",
                      "oov_rate", "non_alpha_rate")


def reference_vocab(texts: list[str]) -> set[str]:
    return {w for t in texts for w in t.lower().split()}


def text_descriptors(texts: list[str], vocab: set[str]):
    """The five-column descriptor frame the K-S layer (and the head-to-head
    benchmark) score. ``vocab`` must NOT be built from the same texts used as the
    K-S reference sample: a self-vocabulary makes the reference's oov_rate
    identically 0, so any other window trivially separates (a guaranteed false
    alarm — measured in the head-to-head protocol notes)."""
    import pandas as pd

    rows = []
    for t in texts:
        words = t.split()
        n_words = len(words) or 1
        oov = sum(1 for w in words if w.lower() not in vocab)
        non_alpha = sum(1 for c in t if not (c.isalpha() or c.isspace()))
        rows.append((len(words), len(t),
                     sum(len(w) for w in words) / n_words,
                     oov / n_words,
                     non_alpha / max(len(t), 1)))
    return pd.DataFrame(rows, columns=list(DESCRIPTOR_COLUMNS))


def _text_domain_estimator():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def domain_classifier_drift(reference_texts: list[str], current_texts: list[str],
                            seed: int = 42, threshold: float = 0.75) -> dict[str, Any]:
    """Reference-vs-current separability via cross-validated ROC-AUC — the text
    instantiation of the shared :class:`~driftguard.detectors.DomainClassifierDetector`
    (TF-IDF + logistic regression), the same detector the tabular/embedding paths use."""
    det = DomainClassifierDetector(estimator=_text_domain_estimator(), threshold=threshold,
                                   seed=seed, splits=5).fit(reference_texts)
    r = det.detect(current_texts)
    return {
        "detector": "domain_classifier",
        "auc": r.statistic,
        "threshold": threshold,
        "drift": r.drift,
        "n_reference": r.extra["n_reference"],
        "n_current": r.extra["n_current"],
    }


def descriptor_ks_drift(reference_texts: list[str], current_texts: list[str],
                        alpha: float = 0.05) -> dict[str, Any]:
    """Bonferroni-corrected K-S over the shared text descriptors, via the shared
    :class:`~driftguard.detectors.DescriptorKSDetector`.

    The reference is split deterministically: even indices build the OOV vocabulary,
    odd indices form the K-S reference sample — the split that keeps the reference's
    own descriptors in-family with live windows (see :func:`text_descriptors`).
    """
    vocab = reference_vocab(reference_texts[0::2])
    det = DescriptorKSDetector(
        features_fn=lambda texts: text_descriptors(texts, vocab), alpha=alpha,
    ).fit(reference_texts[1::2])
    r = det.detect(current_texts)
    return {"detector": "descriptor_ks", "p_min": 1.0 - r.statistic,
            "corrected_alpha": r.extra["corrected_alpha"], "alpha": alpha,
            "p_values": r.extra["p_values"], "drift": r.drift}


def embedding_mmd_drift(reference_texts: list[str], current_texts: list[str],
                        model_name: str) -> dict[str, Any] | None:
    """Optional: sentence-embedding MMD via the shared :class:`~driftguard.detectors.MMDDetector`
    (the encoder is text-specific; the MMD math is the shared detector). Returns None if the
    sentence-transformers extra is not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # noqa: BLE001 - optional extra
        return None
    model = SentenceTransformer(model_name)
    er = model.encode(list(reference_texts), normalize_embeddings=True, show_progress_bar=False)
    ec = model.encode(list(current_texts), normalize_embeddings=True, show_progress_bar=False)
    result = MMDDetector().fit(er).detect(ec)
    return {"detector": "embedding_mmd", "model": model_name, "mmd": result.statistic,
            "n_reference": len(reference_texts), "n_current": len(current_texts)}


def composite_drift(current_texts: list[str], reference_texts: list[str],
                    reference_dist: dict[str, Any], settings: Settings | None = None,
                    use_embed: bool = False) -> dict[str, Any]:
    settings = settings or get_settings()
    # PSI over token_count via the shared detector, reading the frozen training reference —
    # reproduces drift.compute_psi exactly, now through the DriftDetector interface.
    psi_det = PSIDetector.from_reference(reference_dist, values_fn=drift.token_count_signal,
                                         threshold=settings.psi_threshold)
    psi = psi_det.detect(current_texts)
    dom = domain_classifier_drift(reference_texts, current_texts, settings.random_seed,
                                  settings.domain_auc_threshold)
    ks = descriptor_ks_drift(reference_texts, current_texts, settings.descriptor_ks_alpha)

    signals = {
        "psi": {"value": psi.statistic, "threshold": settings.psi_threshold,
                "drift": psi.drift},
        "domain_classifier": {"auc": dom["auc"], "threshold": dom["threshold"],
                              "drift": dom["drift"]},
        "descriptor_ks": {"p_min": ks["p_min"], "corrected_alpha": ks["corrected_alpha"],
                          "drift": ks["drift"]},
    }
    if use_embed:
        emb = embedding_mmd_drift(current_texts, reference_texts, settings.embed_model)
        if emb is not None:   # informational signal (not part of the drift decision)
            signals["embedding_mmd"] = {"mmd": emb["mmd"]}

    decisive = [s.get("drift") for s in signals.values() if "drift" in s]
    triggered_by = [k for k, s in signals.items() if s.get("drift")]
    rule = settings.drift_composite_rule.lower()
    if rule == "all":
        drift_flag = bool(decisive) and all(decisive)
    else:  # "any" (default, safety-first)
        drift_flag = any(decisive)

    if drift_flag:
        log.warning(
            "DRIFT declared (rule=%s) by %s | PSI=%.4f (thr %.2f) domain_auc=%.4f (thr %.2f) "
            "ks_p_min=%.2e (alpha %.4f)",
            rule, triggered_by,
            signals["psi"]["value"], settings.psi_threshold,
            signals["domain_classifier"]["auc"], settings.domain_auc_threshold,
            signals["descriptor_ks"]["p_min"], signals["descriptor_ks"]["corrected_alpha"],
        )
    return {
        "signals": signals,
        "rule": rule,
        "drift": drift_flag,
        "triggered_by": triggered_by,
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
