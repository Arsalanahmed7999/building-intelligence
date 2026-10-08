"""Claim derivation and label resolution for grounded answers (no network)."""

from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer
from building_with_rag.generation.context import assemble_context


def _retrieval() -> QueryResult:
    chunk = RetrievedChunk(
        chunk_id="bns:303:0", section_id="bns:303", act="BNS_2023", text="Theft text.",
        heading="Theft", score=0.9, chapter="XVII", section_number=303, source_pdf="bns.pdf",
    )
    return QueryResult(pattern="semantic", status="ok", message="m", trace={}, results=[chunk])


def test_claims_split_sentences_and_trailing_labels():
    claims = answer.derive_claims("BNS says A. [E1] BNS says B [E1][E2].\n- IPC says C.")
    assert [c.evidence_labels for c in claims] == [["E1"], ["E1", "E2"], []]


def test_unknown_label_flagged():
    ctx = assemble_context(_retrieval().results)
    claims = answer.derive_claims("Theft is punished [E9].")
    assert answer.structural_issues("Theft is punished [E9].", claims, ctx)[0]["check"] == (
        "citation_labels"
    )


def test_citations_resolve_in_first_cited_order():
    ctx = assemble_context(_retrieval().results)
    claims = answer.derive_claims("Theft is punished [E1].")
    citations, passages = answer._resolve(claims, ctx)
    assert citations[0].section_id == "bns:303" and passages[0].chunk_id == "bns:303:0"


def test_sentinel_is_not_streamed(monkeypatch):
    monkeypatch.setattr(
        answer, "_stream_completion", lambda m, meta: iter(["INSUFFICIENT_EVIDENCE: ", "nope"])
    )
    items = list(answer.stream_answer("q", _retrieval()))
    assert all(not isinstance(i, answer.Event) for i in items)
    assert items[-1].outcome == "insufficient_evidence" and items[-1].text == ""
    assert items[-1].trace["reason"] == "nope"
