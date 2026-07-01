"""Smoke test — post-deploy sanity, reused by the CI staging stage.

If ``SERVICE_URL`` is set it hits the live service (used after deploy in Jenkins);
otherwise it runs in-process against the app so `make test` covers it locally.
"""

import os

import httpx
from fastapi.testclient import TestClient

from driftguard.api.main import build_app
from driftguard.config import AG_NEWS_LABELS

SERVICE_URL = os.getenv("SERVICE_URL")


def _check(get, post):
    assert get("/health").json()["status"] == "ok"
    assert get("/ready").json()["ready"] is True
    body = post("/predict", {"text": "The football team secured a dramatic win."}).json()
    assert body["label"] in AG_NEWS_LABELS
    assert body["served_by"] in ("primary", "baseline")


def test_smoke():
    if SERVICE_URL:
        with httpx.Client(base_url=SERVICE_URL, timeout=10.0) as c:
            _check(lambda p: c.get(p), lambda p, j: c.post(p, json=j))
    else:
        with TestClient(build_app()) as c:
            _check(lambda p: c.get(p), lambda p, j: c.post(p, json=j))
