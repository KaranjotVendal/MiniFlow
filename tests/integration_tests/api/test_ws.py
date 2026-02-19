from fastapi.testclient import TestClient

import src.app as app_mod


def test_ws_returns_not_implemented_message():
    client = TestClient(app_mod.app)
    with client.websocket_connect("/ws") as ws:
        payload = ws.receive_json()
        assert payload["status"] == "not_implemented"
        assert "deferred to v2" in payload["message"]
