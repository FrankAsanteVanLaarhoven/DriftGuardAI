"""Chart contract tests: the rendered Helm output must preserve the fallback
contract (probe endpoints) and the canary guard's least-privilege RBAC.
Skipped when helm isn't on PATH (e.g. minimal CI runners)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "driftguard"

pytestmark = pytest.mark.skipif(shutil.which("helm") is None,
                                reason="helm not installed")


def _template(*extra: str) -> str:
    cmd = ["helm", "template", "dg", str(CHART), "-n", "driftguard", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def test_chart_lints_clean():
    proc = subprocess.run(["helm", "lint", str(CHART)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_stable_render_preserves_fallback_contract_probes():
    out = _template()
    # Liveness/startup on /health, readiness on /ready — the stable API.
    assert out.count("path: /health") == 2
    assert out.count("path: /ready") == 1
    # No canary resources unless enabled (comments may mention the word).
    assert "track: canary" not in out
    assert "dg-driftguard-canary" not in out


def test_canary_render_is_complete_and_least_privilege():
    out = _template("--set", "canary.enabled=true")
    # Both tracks carry the contract probes.
    assert out.count("path: /ready") == 2
    # Canary serves the staging alias; stable keeps production.
    assert "models:/driftguard@staging" in out
    assert "models:/driftguard@production" in out
    # Guard RBAC is namespaced (Role, not ClusterRole) and pinned to the canary.
    assert "kind: ClusterRole" not in out
    assert 'resourceNames: ["dg-driftguard-canary"]' in out
    # The guard enforces both breach conditions.
    assert "driftguard_model_tier" in out
    assert 'status=~"5.*"' in out


def test_disabling_monitoring_removes_crd_resources():
    out = _template("--set", "monitoring.serviceMonitor.enabled=false",
                    "--set", "monitoring.prometheusRule.enabled=false")
    assert "kind: ServiceMonitor" not in out
    assert "kind: PrometheusRule" not in out
