"""The tabular reference instance must reuse the governance layer verbatim.

These tests are offline (no OpenML download): they prove the example imports the *same*
gate/metric objects the text service uses, and that its tabular drift primitives work.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

pytest.importorskip("sklearn")
import tabular_adult as tab  # noqa: E402

from driftguard import governance  # noqa: E402


def test_tabular_instance_reuses_the_governance_layer():
    # The whole point of the framework claim: identical objects, not re-implementations.
    assert tab.incumbent_gate is governance.incumbent_gate
    assert tab.promotion_gate is governance.promotion_gate
    assert tab.recovery_ratio is governance.recovery_ratio
    assert tab.retention_ratio is governance.retention_ratio


def test_covariate_drift_shifts_numeric_columns_and_psi_flags_it():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, 500).astype(float),
        "hours-per-week": rng.integers(20, 60, 500).astype(float),
        "workclass": rng.choice(["a", "b", "c"], 500),   # categorical, untouched
    })
    drifted = tab.covariate_drift(df, severity=0.6)

    assert float((drifted["age"] - df["age"]).abs().mean()) > 0.0   # numeric shifted
    assert (drifted["workclass"] == df["workclass"]).all()          # categorical untouched
    psi = tab.psi_numeric(df["age"].to_numpy(), drifted["age"].to_numpy())
    assert psi > 0.2                                                # covariate shift flagged
