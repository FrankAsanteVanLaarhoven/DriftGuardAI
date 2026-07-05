"""Central, environment-driven configuration for DriftGuard.

All settings can be overridden with ``DRIFTGUARD_*`` environment variables, e.g.
``DRIFTGUARD_PSI_THRESHOLD=0.3``. Paths default to a layout that works both for
local development and inside the container image.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two parents up from this file (src/driftguard/config.py).
ROOT = Path(__file__).resolve().parents[2]

# AG News class order is fixed by the dataset and must never be reordered.
AG_NEWS_LABELS: tuple[str, ...] = ("World", "Sports", "Business", "Sci/Tech")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DRIFTGUARD_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "driftguard"
    log_level: str = "INFO"

    # Reproducibility
    random_seed: int = 42
    val_fraction: float = 0.1  # carved out of the HF train split
    max_train_rows: int = 0  # 0 = use the full split; >0 subsamples for speed

    # Filesystem layout
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "artifacts"
    models_dir: Path = ROOT / "models"

    # Fallback contract
    baseline_path: Path = ROOT / "models" / "baseline.joblib"
    primary_pointer_path: Path = ROOT / "models" / "primary_pointer"
    primary_latency_budget_ms: float = 750.0

    # MLflow tracking + registry
    mlflow_tracking_uri: str = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow_experiment: str = "driftguard"
    registered_model_name: str = "driftguard"
    # When set to a ``models:/`` URI (e.g. ``models:/driftguard@production``) the API
    # loads the primary from the MLflow registry first, then falls back to the local
    # pointer file. Empty (the local/dev default) uses the pointer only, which keeps
    # the demo and fallback test hermetic. Production sets this via env.
    primary_model_uri: str = ""
    # Hard deadline on resolving the primary from the registry at (re)load time.
    # A hanging registry (DNS blackhole, network partition) must degrade the service
    # to baseline within seconds — not block startup until the platform kills the pod.
    # Measured in the kind canary drill: unbounded MLflow retries exceed any sane
    # startup-probe budget and turn a fallback-contract degrade into a CrashLoop.
    primary_load_timeout_s: float = 20.0

    # Promotion / baseline gate
    promotion_margin: float = 0.0  # candidate macro-F1 must beat baseline by >= this
    # `make train` promotes a passing candidate automatically. The drift pipeline
    # turns this off so promotion waits behind the human gate.
    auto_promote: bool = True
    # How the evaluative gate scores a candidate under drift:
    #   "fixed"     — vs baseline on the frozen holdout (default; safe in stable regimes)
    #   "refreshed" — vs baseline on a current-distribution (labelled) holdout
    #   "dual"      — must beat baseline on the refreshed holdout AND not drop more than
    #                 `gate_regression_floor` on the fixed holdout (no catastrophic
    #                 forgetting). This resolves the concept-drift recovery block.
    gate_holdout_mode: str = "fixed"
    gate_regression_floor: float = 0.05

    # Drift detection
    psi_threshold: float = 0.2  # >0.2 = action per common PSI convention
    psi_bins: int = 10
    # Text-aware (domain-classifier) drift: reference-vs-current separability AUC.
    # 0.5 = indistinguishable (no drift); -> 1.0 = strongly separable (drift).
    domain_auc_threshold: float = 0.75
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # How the composite verdict combines the per-detector signals:
    #   "any" (default) = drift if PSI OR the domain classifier fires (safety-first);
    #   "all"           = drift only if every signal fires (fewer false positives).
    # The defaults ("any", psi 0.2, auc 0.75) reproduce the documented results table.
    drift_composite_rule: str = "any"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def baseline_metrics_path(self) -> Path:
        return self.artifacts_dir / "baseline_metrics.json"

    @property
    def reference_path(self) -> Path:
        return self.artifacts_dir / "reference.json"

    @property
    def reference_sample_path(self) -> Path:
        # A raw reference-text sample for the text-aware (domain-classifier) detector.
        return self.artifacts_dir / "reference_sample.json"

    @property
    def primary_path(self) -> Path:
        return self.artifacts_dir / "primary.joblib"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifacts_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
