"""Fixed, seeded data pipeline for AG News.

Loads the real Hugging Face dataset ``fancyzhx/ag_news``, carves a deterministic
stratified validation split out of the train partition, keeps the official test
partition as the frozen holdout, and writes processed parquet under ``data/``.

Determinism contract: given the same ``random_seed`` and ``val_fraction`` the row
membership of every split is identical, so ``dvc repro`` reproduces byte-identical
parquet and stable content hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from driftguard.config import AG_NEWS_LABELS, Settings, get_settings

SPLITS = ("train", "val", "test")


def _content_hash(df: pd.DataFrame) -> str:
    """Order-independent hash over logical rows (label-agnostic to parquet encoding)."""
    payload = "\n".join(
        f"{label}\t{text}"
        for label, text in sorted(zip(df["label"].tolist(), df["text"].tolist(), strict=True))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_splits(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Return the deterministic train/val/test frames (in memory)."""
    from datasets import load_dataset

    settings = settings or get_settings()

    ds = load_dataset("fancyzhx/ag_news")
    train_full = ds["train"]
    test = ds["test"]

    if settings.max_train_rows and settings.max_train_rows < train_full.num_rows:
        # Deterministic head after a seeded shuffle keeps class coverage for demos.
        train_full = train_full.shuffle(seed=settings.random_seed).select(
            range(settings.max_train_rows)
        )

    split = train_full.train_test_split(
        test_size=settings.val_fraction,
        seed=settings.random_seed,
        stratify_by_column="label",
    )

    frames = {
        "train": split["train"].to_pandas(),
        "val": split["test"].to_pandas(),
        "test": test.to_pandas(),
    }
    for name, df in frames.items():
        df["label_name"] = df["label"].map(dict(enumerate(AG_NEWS_LABELS)))
        frames[name] = df.reset_index(drop=True)[["text", "label", "label_name"]]
    return frames


def write_splits(frames: dict[str, pd.DataFrame], settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    settings.ensure_dirs()
    manifest: dict[str, dict] = {
        "seed": settings.random_seed,
        "val_fraction": settings.val_fraction,
        "max_train_rows": settings.max_train_rows,
        "splits": {},
    }
    for name in SPLITS:
        df = frames[name]
        out = settings.data_dir / f"{name}.parquet"
        # index=False + fixed engine keeps the encoding reproducible across runs.
        df.to_parquet(out, engine="pyarrow", index=False)
        manifest["splits"][name] = {
            "rows": int(len(df)),
            "path": str(out.relative_to(settings.data_dir.parent)),
            "content_sha256": _content_hash(df),
            "class_counts": {k: int(v) for k, v in df["label_name"].value_counts().items()},
        }
    (settings.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_split(name: str, settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path: Path = settings.data_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `make data` first.")
    return pd.read_parquet(path)


def main() -> None:
    settings = get_settings()
    frames = build_splits(settings)
    manifest = write_splits(frames, settings)
    print("Processed AG News splits written to", settings.data_dir)
    for name, meta in manifest["splits"].items():
        print(f"  {name:<5} rows={meta['rows']:>7}  sha256={meta['content_sha256'][:12]}…")


if __name__ == "__main__":
    main()
