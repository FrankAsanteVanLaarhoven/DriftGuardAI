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

    # Promotion / baseline gate
    promotion_margin: float = 0.0  # candidate macro-F1 must beat baseline by >= this

    # Drift detection
    psi_threshold: float = 0.2  # >0.2 = action per common PSI convention
    psi_bins: int = 10

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
    def primary_path(self) -> Path:
        return self.artifacts_dir / "primary.joblib"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifacts_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
