"""Embedding reference instance — the DriftGuard governance framework on 20 Newsgroups.

A **third, distinct instance** (Text + Tabular + **Embeddings**) that reuses
``driftguard.governance`` *verbatim* and the shared ``driftguard.detectors`` on dense
sentence-embedding vectors — proving the "third modality is free" claim on real data, not a
synthetic stub. Only the data (20 Newsgroups), the representation (MiniLM sentence
embeddings) and the models are embedding-specific; every gate, metric, and detector class is
imported unchanged from the framework.

    uv run --extra embed python examples/embedding_20news.py

Requires the ``embed`` extra (sentence-transformers) and downloads 20 Newsgroups on first
use. Writes examples/results_embedding.json. The offline test (tests/test_embedding.py)
exercises the same wiring with a deterministic hashing encoder — no torch, no network.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The SAME framework layer the text and tabular instances use — imported, not re-implemented.
from driftguard.governance import (  # noqa: E402
    incumbent_gate,
    promotion_gate,
    recovery_ratio,
    retention_ratio,
)

SEED = 42
CATEGORIES = ("sci.space", "rec.sport.baseball", "talk.politics.mideast", "comp.graphics")


def load_20news():
    from sklearn.datasets import fetch_20newsgroups

    strip = ("headers", "footers", "quotes")   # force the model onto content, not metadata
    tr = fetch_20newsgroups(subset="train", categories=list(CATEGORIES), remove=strip)
    te = fetch_20newsgroups(subset="test", categories=list(CATEGORIES), remove=strip)
    return tr.data, np.asarray(tr.target), te.data, np.asarray(te.target)


def sentence_encoder() -> Callable[[Sequence[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    from driftguard.config import get_settings

    model = SentenceTransformer(get_settings().embed_model)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True,
                                      show_progress_bar=False)


def drift_embeddings(E, severity: float) -> np.ndarray:
    """Covariate shift **in embedding space**: an information-preserving orthogonal rotation
    of a ``severity`` fraction of the embedding dimensions. The transform is fixed for a given
    severity (seeded), so train and eval undergo the *same* shift. Because it preserves
    information, retraining fully recovers the drifted task — but a shift-specialised model
    increasingly forgets the clean distribution as more dimensions rotate. That recovery-vs-
    forgetting tension is exactly what the governance gates arbitrate — the embedding-space
    analogue of the tabular numeric-feature shift, on learned features. (The embeddings come
    from real 20 Newsgroups text; the drift acts on the vectors, since random word edits are
    unrecoverable noise a model cannot adapt to.)"""
    rng = np.random.default_rng(SEED)
    d = E.shape[1]
    k = max(1, int(severity * d))
    idx = rng.choice(d, k, replace=False)
    rotation, _ = np.linalg.qr(rng.normal(0.0, 1.0, (k, k)))   # orthogonal ⇒ info-preserving
    Ed = np.array(E, dtype=float, copy=True)
    Ed[:, idx] = E[:, idx] @ rotation
    return Ed


def build_primary(E, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000).fit(E, y)


def build_baseline(E, y, dim: int):
    """Deliberately weaker incumbent-baseline: logistic regression on a low-rank projection
    of the embeddings, so there is a real quality gap for the gates to protect."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    d = max(2, min(dim, E.shape[1] - 1))
    return Pipeline([("svd", TruncatedSVD(n_components=d, random_state=SEED)),
                     ("clf", LogisticRegression(max_iter=1000))]).fit(E, y)


def macro_f1(model, E, y) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, model.predict(E), average="macro"))


def evaluate(train_texts, ytr, test_texts, yte, encode, severity: float = 0.5,
             baseline_dim: int = 3) -> dict:
    E_tr = np.asarray(encode(train_texts))
    E_te = np.asarray(encode(test_texts))

    primary = build_primary(E_tr, ytr)                      # incumbent: full embeddings
    baseline = build_baseline(E_tr, ytr, baseline_dim)      # weaker: compressed projection
    base_fixed = macro_f1(baseline, E_te, yte)
    prim_fixed = macro_f1(primary, E_te, yte)

    # --- inject a covariate shift in embedding space (same fixed transform on both) ----
    E_tr_d = drift_embeddings(E_tr, severity)
    E_te_d = drift_embeddings(E_te, severity)
    stale_drift = macro_f1(primary, E_te_d, yte)

    # --- detect with the SHARED detectors on the embedding vectors (no new detector code)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression

    from driftguard.detectors import CompositeDetector, DomainClassifierDetector, PSIDetector

    pc = TruncatedSVD(n_components=1, random_state=SEED).fit(E_te)   # reference top PC
    detector = CompositeDetector([
        PSIDetector(values_fn=lambda E: pc.transform(np.asarray(E))[:, 0], threshold=0.2),
        DomainClassifierDetector(estimator=LogisticRegression(max_iter=200), threshold=0.75),
    ], rule="any").fit(E_te)
    det = detector.detect(E_te_d)
    psi = det.extra["signals"]["psi"]["statistic"]
    auc = det.extra["signals"]["domain_classifier"]["statistic"]
    detected = det.drift

    # --- retrain a candidate on the drifted embeddings, then score both holdouts -------
    candidate = build_primary(E_tr_d, ytr)
    cand_drift = macro_f1(candidate, E_te_d, yte)
    cand_fixed = macro_f1(candidate, E_te, yte)
    base_drift = macro_f1(baseline, E_te_d, yte)

    # --- GOVERNANCE (identical functions to the text + tabular instances) --------------
    gate_inc = incumbent_gate(cand_fixed, base_fixed, prim_fixed)
    gate_dual = promotion_gate(
        candidate_fixed_f1=cand_fixed, baseline_fixed_f1=base_fixed,
        candidate_refreshed_f1=cand_drift, baseline_refreshed_f1=base_drift,
        mode="dual", regression_floor=0.05,
    )
    rec = recovery_ratio(cand_drift, stale_drift, prim_fixed)
    ret = retention_ratio(cand_fixed, prim_fixed)

    return {
        "instance": "embedding/20newsgroups",
        "model": f"LogReg on MiniLM-{E_tr.shape[1]}d (primary) + "
                 f"LogReg on TruncatedSVD-{baseline_dim} (baseline)",
        "severity": severity,
        "macro_f1": {
            "baseline_fixed": round(base_fixed, 4),
            "primary_fixed_incumbent": round(prim_fixed, 4),
            "stale_on_drift": round(stale_drift, 4),
            "candidate_on_drift": round(cand_drift, 4),
            "candidate_on_fixed": round(cand_fixed, 4),
        },
        "detection": {"psi_top_pc": round(psi, 4), "domain_auc": round(auc, 4),
                      "detected": detected},
        "governance": {
            "incumbent_gate_passed": gate_inc.passed,
            "incumbent_gate_reason": gate_inc.reason,
            "dual_gate_passed": gate_dual.passed,
            "dual_gate_reason": gate_dual.reason,
            "recovery_ratio": round(rec, 4),
            "retention_ratio": round(ret, 4),
        },
    }


def sweep(severities=(0.1, 0.25, 0.5, 0.75)) -> dict:
    """Recovery/retention vs drift severity (= fraction of embedding dims rotated) — the
    embedding analogue of the text `make recovery-sweep`, driven by the same governance gates
    and metrics. The encoder and dataset are loaded once and reused across severities."""
    tr_texts, ytr, te_texts, yte = load_20news()
    encode = sentence_encoder()
    rows, first = [], None
    for sev in severities:
        r = evaluate(tr_texts, ytr, te_texts, yte, encode, severity=sev)
        first = first or r
        g, d = r["governance"], r["detection"]
        rows.append({
            "severity": sev,
            "detected": d["detected"],
            "recovery_ratio": g["recovery_ratio"],
            "retention_ratio": g["retention_ratio"],
            "dual_gate_passed": g["dual_gate_passed"],
        })
    return {"instance": first["instance"], "model": first["model"],
            "macro_f1_clean": first["macro_f1"], "rows": rows}


def to_markdown(s: dict) -> str:
    m = s["macro_f1_clean"]
    lines = [
        f"Instance: {s['instance']} — {s['model']}",
        f"Clean holdout: baseline {m['baseline_fixed']:.4f} | "
        f"primary/incumbent {m['primary_fixed_incumbent']:.4f} (macro-F1).",
        "",
        "Recovery/retention vs drift severity (fraction of embedding dims rotated):",
        "",
        "| severity | detected | recovery ratio | retention ratio | dual gate |",
        "|----------|----------|----------------|-----------------|-----------|",
    ]
    for r in s["rows"]:
        lines.append(
            f"| {r['severity']:.2f} | {r['detected']} | {r['recovery_ratio']:.3f} | "
            f"{r['retention_ratio']:.3f} | {'PASS' if r['dual_gate_passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Recovery stays ~1.0 (the rotation is information-preserving, so retraining fully "
        "recovers the drifted task), yet retention falls as more dimensions rotate and the "
        "`dual` gate flips PASS -> FAIL — promoting the recovered model would wreck the clean "
        "distribution. Same governance as text + tabular, imported unchanged, on embeddings.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    result = sweep()
    out = Path(__file__).resolve().parent / "results_embedding.json"
    out.write_text(json.dumps(result, indent=2))
    print(to_markdown(result))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
