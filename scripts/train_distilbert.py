"""Fine-tune a DistilBERT primary and gate it against the incumbent primary.

Runs on a GPU when one is present (the classifier auto-selects CUDA). It reuses the exact
seeded splits, the model-bundle format, and the no-worse-than-incumbent gate, so the
resulting model is served, gated, and fallen-back-from with no serving-code changes.

    uv sync --extra transformer
    make train-transformer          # 3 epochs, gated + promoted

Measured on an RTX 4080 SUPER: accuracy 0.9413, macro-F1 0.9412 — it beat the incumbent
linear primary (0.9197) and was promoted. Numbers recorded in CASE_STUDY.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftguard import registry  # noqa: E402
from driftguard.config import get_settings  # noqa: E402
from driftguard.data import load_split  # noqa: E402
from driftguard.transformer_primary import TransformerClassifier  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the DistilBERT primary")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="Subsample the train split for a quick smoke run (0 = full).")
    parser.add_argument("--promote", action="store_true",
                        help="If the gate passes, point the service primary at this model.")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()

    train = load_split("train", settings)
    test = load_split("test", settings)
    if args.max_train_rows:
        train = train.sample(n=args.max_train_rows, random_state=settings.random_seed)
    xtr, ytr = train["text"].tolist(), train["label"].tolist()
    xte, yte = test["text"].tolist(), test["label"].tolist()

    clf = TransformerClassifier(epochs=args.epochs, batch_size=args.batch_size,
                                max_length=args.max_length, seed=settings.random_seed)
    print(f"Fine-tuning {clf.model_name} for {args.epochs} epoch(s) on {len(xtr)} rows…")
    clf.fit(xtr, ytr)

    metrics = registry.evaluate(clf, xte, yte)
    baseline_m = json.loads(settings.baseline_metrics_path.read_text())
    # Gate against max(baseline, current primary): a slow transformer must beat the model
    # actually serving, not just the tiny baseline — otherwise promotion is a downgrade.
    incumbent_f1 = registry.current_primary_macro_f1(settings)
    gate = registry.incumbent_gate(metrics["macro_f1"], baseline_m["macro_f1"],
                                   incumbent_f1, settings.promotion_margin)
    inc_txt = f"{incumbent_f1:.4f}" if incumbent_f1 is not None else "none"
    print(f"DistilBERT holdout: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"Incumbent primary macro_f1: {inc_txt}")
    print(f"Promotion gate: {'PASS' if gate.passed else 'FAIL'} — {gate.reason}")

    out = settings.artifacts_dir / "primary_transformer.joblib"
    registry.save_bundle(registry.make_bundle(clf, "primary", metrics, "distilbert-1"), out)
    print(f"Saved bundle -> {out}")

    if args.promote and gate.passed:
        settings.primary_pointer_path.write_text(str(out.relative_to(out.parents[1])))
        print(f"Promoted: pointer -> {settings.primary_pointer_path}")
    elif args.promote:
        print("Gate failed — not promoted (fail-closed).")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
