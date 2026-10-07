"""Parser unit tests for Story 3.1 (no network)."""

import json

from building_with_rag.contracts import RetrievedChunk
from building_with_rag.generation.answer import build_result
from building_with_rag.generation.context import assemble_context


def _ctx():
    chunk = RetrievedChunk(
        chunk_id="bns:303:0", section_id="bns:303", act="BNS_2023", text="Theft text.",
        heading="Theft", score=0.9, chapter="XVII", section_number=303, source_pdf="bns.pdf",
    )
    return assemble_context([chunk])


def _run(content: str):
    return build_result(content, _ctx(), "m", {})


def test_unknown_label_malformed():
    p = {"outcome": "answered", "answer": "a", "claims": [{"text": "t", "evidence": ["E9"]}]}
    assert _run(json.dumps(p)).outcome == "malformed"


def test_non_json_malformed():
    assert _run("not json").outcome == "malformed"


def test_answered_without_claims_malformed():
    p = {"outcome": "answered", "answer": "a", "claims": []}
    assert _run(json.dumps(p)).outcome == "malformed"


def test_valid_answer_resolves_citations():
    p = {"outcome": "answered", "answer": "a", "claims": [{"text": "t", "evidence": ["E1"]}]}
    r = _run("```json\n" + json.dumps(p) + "\n```")
    assert r.outcome == "answered" and r.text == "a"
    assert r.citations[0].chunk_id == "bns:303:0"
    assert r.supporting_passages[0].section_id == "bns:303"
