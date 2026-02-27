import os

from fastapi.testclient import TestClient

os.environ.setdefault("MINIFLOW_CONFIG", "configs/baseline.yml")
os.environ.setdefault("RELEASE_ID", "test-release")

import src.app as app_mod


def test_ws_returns_not_implemented_message():
    client = TestClient(app_mod.app)
    with client.websocket_connect("/ws") as ws:
        payload = ws.receive_json()
        assert payload["status"] == "not_implemented"
        assert "deferred to v2" in payload["message"]
