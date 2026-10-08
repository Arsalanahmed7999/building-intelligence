"""Story 3.2 offline tests: fake provider, no network."""

import pytest
from fastapi.testclient import TestClient

from building_with_rag.app import create_app
from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer
from building_with_rag.routes import chat

GOOD = '{"claims": [{"i": 0, "supported": true}]}'


def _retrieval() -> QueryResult:
    chunk = RetrievedChunk(
        chunk_id="bns:303:0", section_id="bns:303", act="BNS_2023", text="Theft text.",
        heading="Theft", score=0.9, chapter="XVII", section_number=303, source_pdf="bns.pdf",
    )
    return QueryResult(pattern="semantic", status="ok", message="m", trace={}, results=[chunk])


@pytest.fixture()
def provider(monkeypatch):
    """Script stream attempts (lists of deltas or exceptions) and validator replies."""
    state = {"stream_calls": 0, "validator_calls": 0}

    def install(attempts, validator_replies=()):
        streams, replies = iter(attempts), iter(validator_replies)

        def fake_stream(messages, meta):
            state["stream_calls"] += 1
            for item in next(streams):
                if isinstance(item, Exception):
                    raise item
                yield item

        def fake_complete(messages):
            state["validator_calls"] += 1
            return next(replies)

        monkeypatch.setattr(answer, "_stream_completion", fake_stream)
        monkeypatch.setattr(answer, "_complete", fake_complete)
        monkeypatch.setattr(chat, "retrieve", lambda request: _retrieval())
        return state

    return install


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def _chat(client, stream=False):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "rag-semantic",
            "stream": stream,
            "messages": [{"role": "user", "content": "theft?"}],
        },
    )


def _run():
    events = list(answer.stream_answer("q", _retrieval()))
    return events[:-1], events[-1]


def test_passing_attempt_single_call_high(provider):
    state = provider([["Theft is ", "punishable [E1]."]], [GOOD])
    events, final = _run()
    streamed = "".join(e.text for e in events)
    assert state["stream_calls"] == 1
    assert streamed == answer.DRAFT_LINE + final.text
    assert final.outcome == "answered" and final.confidence == "high"
    assert final.citations[0].section_id == "bns:303" and final.issues == []


def test_failed_attempt_then_passing_retry(provider):
    provider([["Theft is punishable."], ["Theft is punishable [E1]."]], [GOOD])
    events, final = _run()
    streamed = "".join(e.text for e in events)
    assert streamed.count(answer.DRAFT_LINE) == 2 and "Retrying (attempt 2 of 2)" in streamed
    assert [a["status"] for a in final.attempts] == ["failed", "passed"]
    assert [i["check"] for i in final.issues] == ["claim_cited"]
    assert final.confidence == "high"


def test_both_attempts_fail_low_confidence(provider, client):
    provider([["Theft is punishable [E9]."]] * 2)
    body = _chat(client).json()["choices"][0]["message"]["content"]
    assert body.count(answer.DRAFT_LINE) == 2 and "DRAFT — low confidence" in body
    provider([["Theft is punishable [E9]."]] * 2)
    _, final = _run()
    assert final.outcome == "malformed" and final.text == "" and final.confidence == "low"
    assert "[E9]" in final.draft_answer and final.low_confidence_reason
    assert len(final.issues) == 2 and final.citations == []


def test_provider_failure_mid_stream_still_terminates(provider, client):
    provider([["Theft is ", answer.ProviderError("connection")]])
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "rag-semantic", "stream": True,
              "messages": [{"role": "user", "content": "theft?"}]},
    ) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes()).decode()
    assert "Answer generation unavailable" in raw and '"finish_reason": "stop"' in raw
    assert raw.strip().endswith("data: [DONE]")
    provider([["Theft is ", answer.ProviderError("connection")]])
    _, final = _run()
    assert final.outcome == "unavailable" and final.confidence is None
    assert final.draft_answer == "Theft is"
