import os
import pytest

from fastapi.testclient import TestClient

os.environ.setdefault("MINIFLOW_CONFIG", "configs/baseline.yml")
os.environ.setdefault("RELEASE_ID", "test-release")

import src.app as app_mod


def test_health_returns_200():
    client = TestClient(app_mod.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready_distinct_from_health_when_config_missing():
    with TestClient(app_mod.app) as client:
        app_mod.app.state.app_config = {
            "asr": {},
            "llm": {},
            "tts": {},
        }
        health = client.get("/health")
        ready = client.get("/ready")

    assert health.status_code == 200
    assert ready.status_code == 503
    payload = ready.json()
    assert payload["status"] == "not_ready"
    assert payload["reason"] == "configuration incomplete"
