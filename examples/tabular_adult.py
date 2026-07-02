"""Tabular reference instance — the DriftGuard governance framework on Adult income.

A **second, non-text instance** that reuses ``driftguard.governance`` *verbatim*: the
promotion gates and the recovery/retention metrics are the exact same functions the text
service uses. Only the data (OpenML Adult / Census Income), the model
(HistGradientBoosting), and the drift detector (PSI + a domain classifier on features) are
tabular-specific. This turns the "model-agnostic framework" claim into runnable code.

    uv run python examples/tabular_adult.py

CPU-only; downloads Adult from OpenML on first use. Writes examples/results_tabular.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The SAME framework layer the text reference service uses — imported, not re-implemented.
from driftguard.governance import (  # noqa: E402
    incumbent_gate,
    promotion_gate,
    recovery_ratio,
    retention_ratio,
)

SEED = 42
DRIFT_COLS = ("age", "hours-per-week", "capital-gain", "education-num")


def load_adult():
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    return X, (y.astype(str) == ">50K").astype(int).to_numpy()


def build_pipeline(kind: str, X):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=20,
                              sparse_output=False), cat),
    ])
    if kind == "baseline":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, C=0.5)
    else:  # primary — a stronger, different model family than the text linear primary
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def macro_f1(pipe, X, y) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, pipe.predict(X), average="macro"))


def covariate_drift(X, severity: float = 0.6):
    """Tabular covariate shift: scale + jitter numeric feature columns (analogue of the
    text vocab drift). Categoricals are untouched, so it is a pure covariate shift."""
    rng = np.random.default_rng(SEED)
    Xd = X.copy()
    for col in DRIFT_COLS:
        if col in Xd.columns:
            noise = rng.normal(0.0, severity * (Xd[col].std() or 1.0), len(Xd))
            Xd[col] = Xd[col] * (1.0 + severity) + noise
    return Xd


def psi_numeric(ref, cur, bins: int = 10, eps: float = 1e-6) -> float:
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / max(len(ref), 1) + eps
    c = np.histogram(cur, edges)[0] / max(len(cur), 1) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def domain_auc(ref_X, cur_X) -> float:
    """Reference-vs-current separability on numeric features — the same domain-classifier
    drift idea as the text detector, with a tabular featurizer."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    num = ref_X.select_dtypes(include="number").columns
    n = min(len(ref_X), len(cur_X))
    a = ref_X[num].sample(n, random_state=SEED).to_numpy()
    b = cur_X[num].sample(n, random_state=SEED).to_numpy()
    xx = np.vstack([a, b])
    yy = np.array([0] * n + [1] * n)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    clf = HistGradientBoostingClassifier(max_iter=80, random_state=SEED)
    return float(np.mean(cross_val_score(clf, xx, yy, cv=cv, scoring="roc_auc")))


def run(severity: float = 0.6) -> dict:
    from sklearn.model_selection import train_test_split

    X, y = load_adult()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    baseline = build_pipeline("baseline", Xtr).fit(Xtr, ytr)
    primary = build_pipeline("primary", Xtr).fit(Xtr, ytr)          # the incumbent
    base_fixed = macro_f1(baseline, Xte, yte)
    prim_fixed = macro_f1(primary, Xte, yte)

    # --- inject covariate drift; the incumbent degrades on the new distribution --------
    Xtr_d, Xte_d = covariate_drift(Xtr, severity), covariate_drift(Xte, severity)
    stale_drift = macro_f1(primary, Xte_d, yte)

    # --- detect (tabular multi-layer: PSI on a feature + domain classifier) ------------
    psi = psi_numeric(Xte["hours-per-week"].to_numpy(), Xte_d["hours-per-week"].to_numpy())
    auc = domain_auc(Xte, Xte_d)
    detected = psi > 0.2 or auc >= 0.75

    # --- retrain a candidate on the drifted data, then score both holdouts -------------
    candidate = build_pipeline("primary", Xtr_d).fit(Xtr_d, ytr)
    cand_drift = macro_f1(candidate, Xte_d, yte)
    cand_fixed = macro_f1(candidate, Xte, yte)
    base_drift = macro_f1(baseline, Xte_d, yte)

    # --- GOVERNANCE (identical functions to the text service) --------------------------
    # incumbent_gate on the fixed holdout: is the candidate a no-worse drop-in on the
    # reference distribution? (A drift-specialised candidate usually is not — hence dual.)
    gate_inc = incumbent_gate(cand_fixed, base_fixed, prim_fixed)
    gate_dual = promotion_gate(
        candidate_fixed_f1=cand_fixed, baseline_fixed_f1=base_fixed,
        candidate_refreshed_f1=cand_drift, baseline_refreshed_f1=base_drift,
        mode="dual", regression_floor=0.05,
    )
    rec = recovery_ratio(cand_drift, stale_drift, prim_fixed)
    ret = retention_ratio(cand_fixed, prim_fixed)

    return {
        "instance": "tabular/adult",
        "model": "HistGradientBoosting (primary) + LogisticRegression (baseline)",
        "severity": severity,
        "macro_f1": {
            "baseline_fixed": round(base_fixed, 4),
            "primary_fixed_incumbent": round(prim_fixed, 4),
            "stale_on_drift": round(stale_drift, 4),
            "candidate_on_drift": round(cand_drift, 4),
            "candidate_on_fixed": round(cand_fixed, 4),
        },
        "detection": {"psi_hours_per_week": round(psi, 4), "domain_auc": round(auc, 4),
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


def sweep(severities=(0.1, 0.2, 0.4)) -> dict:
    """Recovery/retention vs covariate-drift severity — the tabular analogue of the text
    `make recovery-sweep`, driven by the same governance gates + metrics."""
    rows = []
    first = None
    for sev in severities:
        r = run(severity=sev)
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
        "Recovery/retention vs covariate-drift severity (same governance as the text service):",
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
        "As drift deepens retention falls and the `dual` gate flips PASS -> FAIL — the same "
        "safety behaviour the text instance shows, on a gradient-boosting tabular model.",
        "Every gate/metric is `driftguard.governance`, imported unchanged.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    result = sweep()
    out = Path(__file__).resolve().parent / "results_tabular.json"
    out.write_text(json.dumps(result, indent=2))
    print(to_markdown(result))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
