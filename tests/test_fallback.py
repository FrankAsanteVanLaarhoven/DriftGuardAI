"""The fallback (chaos) test — the core resilience guarantee.

Removing or corrupting the primary must NOT take the service down: readiness stays
200, predictions still succeed via the baseline, ``served_by`` flips to "baseline",
and ``driftguard_model_tier{tier="baseline"}`` goes to 1.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from driftguard.api import main as api
from driftguard.api.main import build_app, refresh_models
from driftguard.config import Settings, get_settings


def _baseline_path() -> Path:
    # The committed, always-loadable fallback.
    return get_settings().baseline_path


def _tier_gauge(tier: str) -> float:
    return api.MODEL_TIER.labels(tier=tier)._value.get()


def test_missing_primary_still_serves_baseline():
    settings = Settings(
        baseline_path=_baseline_path(),
        primary_model_uri="",
        primary_pointer_path=Path("/nonexistent/primary_pointer"),
    )
    with TestClient(build_app(settings)) as client:
        assert client.get("/ready").status_code == 200          # never out of rotation
        assert client.get("/ready").json()["ready"] is True

        r = client.post("/predict", json={"text": "The central bank raised interest rates."})
        assert r.status_code == 200                             # never a 5xx for a bad primary
        assert r.json()["served_by"] == "baseline"

        info = client.get("/model-info").json()
        assert info["primary_available"] is False
        assert info["active_tier"] == "baseline"
        assert _tier_gauge("baseline") == 1.0


def test_corrupt_primary_fails_canary_and_falls_back(tmp_path):
    corrupt = tmp_path / "primary.joblib"
    corrupt.write_bytes(b"this is not a valid joblib payload")
    pointer = tmp_path / "primary_pointer"
    pointer.write_text(str(corrupt))

    settings = Settings(
        baseline_path=_baseline_path(),
        primary_model_uri="",
        primary_pointer_path=pointer,
    )
    with TestClient(build_app(settings)) as client:
        r = client.post("/predict", json={"text": "Team wins the championship final."})
        assert r.status_code == 200
        assert r.json()["served_by"] == "baseline"
        assert client.get("/model-info").json()["primary_available"] is False


def test_primary_removed_at_runtime_flips_to_baseline():
    # Start healthy with the real trained primary, then rip it out and reload.
    app = build_app(get_settings())
    with TestClient(app) as client:
        before = client.post("/predict", json={"text": "New chip breaks AI speed record."})
        assert before.status_code == 200

        fallback_before = api.FALLBACK_TOTAL._value.get()
        app.state.settings = Settings(
            baseline_path=_baseline_path(),
            primary_model_uri="",
            primary_pointer_path=Path("/nonexistent/primary_pointer"),
        )
        refresh_models(app)

        after = client.post("/predict", json={"text": "New chip breaks AI speed record."})
        assert after.status_code == 200
        assert after.json()["served_by"] == "baseline"
        assert client.get("/ready").json()["ready"] is True
        assert _tier_gauge("baseline") == 1.0
        # Metric surface exposes the degraded tier.
        metrics = client.get("/metrics").text
        assert 'driftguard_model_tier{tier="baseline"} 1.0' in metrics
        assert api.FALLBACK_TOTAL._value.get() >= fallback_before


def test_pointer_removed_on_live_service_degrades_next_request(tmp_path):
    # A copy of the committed model acts as a stand-in primary in a temp location.
    import shutil

    primary_copy = tmp_path / "primary.joblib"
    shutil.copy(_baseline_path(), primary_copy)
    pointer = tmp_path / "primary_pointer"
    pointer.write_text(str(primary_copy))

    settings = Settings(baseline_path=_baseline_path(), primary_model_uri="",
                        primary_pointer_path=pointer)
    with TestClient(build_app(settings)) as client:
        first = client.post("/predict", json={"text": "Elections were held across the region."})
        assert first.json()["served_by"] == "primary"

        pointer.unlink()  # operator pulls the primary out from under the running service
        second = client.post("/predict", json={"text": "Elections were held across the region."})
        assert second.status_code == 200
        assert second.json()["served_by"] == "baseline"
        assert client.get("/model-info").json()["primary_available"] is False


def test_latency_budget_breach_falls_back_to_baseline():
    # A zero budget forces every primary prediction to be "too slow" -> baseline.
    settings = Settings(primary_latency_budget_ms=0.0)
    breach_before = api.BUDGET_BREACH_TOTAL._value.get()
    with TestClient(build_app(settings)) as client:
        r = client.post("/predict", json={"text": "Markets rally on strong earnings."})
        assert r.status_code == 200
        assert r.json()["served_by"] == "baseline"
    assert api.BUDGET_BREACH_TOTAL._value.get() > breach_before
