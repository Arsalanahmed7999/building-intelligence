from fastapi.testclient import TestClient

from building_with_rag.main import app
from building_with_rag.modes import MODE_REGISTRY

client = TestClient(app)
CHAT = {"model": "rag-semantic", "messages": [{"role": "user", "content": "What is murder?"}]}


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_query_placeholder():
    r = client.post("/v1/query", json={"question": "What is murder?", "pattern": "hybrid"})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "not_implemented" and body["message"]
    assert body["results"] == [] and body["generation"] is None


def test_models():
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == list(MODE_REGISTRY.values()) and len(ids) == 6


def test_chat_json():
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.json()["choices"][0]["message"]["content"].startswith("The 'semantic' mode")


def test_chat_sse():
    payload = {**CHAT, "stream": True, "rag_options": {"pattern": "semantic", "limit": 3}}
    r = client.post("/v1/chat/completions", json=payload)
    assert r.text.rstrip().endswith("data: [DONE]")
    assert '"finish_reason": "stop"' in r.text


def test_chat_error_envelope():
    r = client.post("/v1/chat/completions", json={**CHAT, "model": "nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
