"""Story 4.1 offline tests: RRF fusion and run_hybrid with faked searches (no network)."""

import pytest

from building_with_rag.contracts import QueryRequest, RetrievedChunk
from building_with_rag.registry import Pattern
from building_with_rag.retrieval import hybrid, semantic


def _hits(*ids):
    return [{"chunk_id": c, "score": 1.0 - i / 10} for i, c in enumerate(ids)]


def test_chunk_in_both_routes_ranks_first_with_rrf_score():
    rows = hybrid.fuse(_hits("a", "b"), _hits("c", "b"), limit=5)
    assert rows[0]["chunk_id"] == "b"
    assert (rows[0]["semantic_rank"], rows[0]["keyword_rank"]) == (2, 2)
    assert rows[0]["fused_score"] == pytest.approx(2 / (hybrid.RRF_K + 2))
    assert [r["fused_rank"] for r in rows] == [1, 2, 3]


def test_single_route_chunks_leave_other_route_none():
    rows = {r["chunk_id"]: r for r in hybrid.fuse(_hits("a"), _hits("c"), limit=5)}
    assert rows["a"]["keyword_score"] is None and rows["a"]["keyword_rank"] is None
    assert rows["c"]["semantic_score"] is None and rows["c"]["semantic_rank"] is None


def test_tie_order_is_semantic_rank_then_chunk_id():
    # a (semantic rank 1) and z (keyword rank 1) tie; semantic_rank wins, missing last.
    assert [r["chunk_id"] for r in hybrid.fuse(_hits("a"), _hits("z"), limit=5)] == ["a", "z"]
    assert [r["chunk_id"] for r in hybrid.fuse([], _hits("y", "x"), limit=1)] == ["y"]
    assert [r["chunk_id"] for r in hybrid.fuse([], [{"chunk_id": c, "score": 1} for c in "b"], 5)] == ["b"]


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, section_id="bns:303", act="BNS_2023", text="t", heading="h", score=score
    )


def _fake_backend(monkeypatch):
    monkeypatch.setattr(semantic, "_validate_scope", lambda r: None)
    monkeypatch.setattr(semantic, "_ensure_ready", lambda: None)
    monkeypatch.setattr(hybrid, "_ensure_keyword_ready", lambda: None)
    monkeypatch.setattr(semantic, "_embed_query", lambda q: [0.0])
    monkeypatch.setattr(semantic, "_search", lambda *a: _hits("a", "b"))
    monkeypatch.setattr(hybrid, "_keyword_search", lambda *a: _hits("b", "c"))
    monkeypatch.setattr(
        semantic, "_resolve", lambda hits: ([_chunk(h["chunk_id"], h["score"]) for h in hits], 0)
    )


def test_run_hybrid_sets_score_and_route_fields(monkeypatch):
    _fake_backend(monkeypatch)
    result = hybrid.run_hybrid(QueryRequest(question="theft", pattern=Pattern.HYBRID, limit=3))
    assert result.status == "ok" and result.pattern == "hybrid"
    top = result.results[0]
    assert top.chunk_id == "b" and top.score == top.fused_score and top.fused_rank == 1
    assert (top.semantic_rank, top.keyword_rank) == (2, 1)
    assert result.trace["contribution"] == {"both": 1, "semantic_only": 1, "keyword_only": 1}


def test_missing_keyword_index_is_not_ready(monkeypatch):
    class Coll:
        def list_search_indexes(self, name):
            return []

    monkeypatch.setattr(hybrid, "_keyword_ready", False)
    monkeypatch.setattr(semantic, "_db", lambda: {"chunks": Coll()})
    with pytest.raises(semantic.RetrievalError) as exc:
        hybrid._ensure_keyword_ready()
    assert exc.value.status_code == 503 and exc.value.code == "retrieval_not_ready"
    assert "keyword_index" in exc.value.message
