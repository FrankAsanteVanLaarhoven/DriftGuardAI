"""Integration tests for the serving contract (happy path + validation)."""

from fastapi.testclient import TestClient

from driftguard.api.main import build_app
from driftguard.config import AG_NEWS_LABELS


def test_health_ready_and_predict_happy_path():
    with TestClient(build_app()) as client:
        assert client.get("/health").json()["status"] == "ok"

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True
        assert "baseline" in ready.json()["servable_tiers"]

        r = client.post("/predict", json={"text": "New GPU sets an on-device AI record."})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in AG_NEWS_LABELS
        assert body["served_by"] in ("primary", "baseline")
        assert 0.0 <= body["confidence"] <= 1.0
        assert abs(sum(body["scores"].values()) - 1.0) < 1e-3


def test_predict_rejects_empty_and_missing_text():
    with TestClient(build_app()) as client:
        assert client.post("/predict", json={"text": ""}).status_code == 422
        assert client.post("/predict", json={}).status_code == 422


def test_model_info_shape():
    with TestClient(build_app()) as client:
        info = client.get("/model-info").json()
        assert set(info) >= {"active_tier", "primary_available", "baseline_version"}
        assert info["baseline_version"]
