"""Hybrid retrieval: Atlas Search keyword route + semantic vector route, fused by RRF.

Both routes use the same corpus and embeddings as semantic mode. Fusion uses ranks only
(BM25 and cosine scores are not comparable): fused_score = sum 1/(RRF_K + rank).
"""

from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalError

RRF_K = 60
WEIGHTS = {"semantic": 1.0, "keyword": 1.0}

_SCORE_NOTE = "The fused score ranks passages only; it does not prove a passage is correct or answers the question."

_keyword_ready = False


def route_depth(limit: int) -> int:
    return max(limit, min(50, max(20, 4 * limit)))


def fuse(semantic_hits: list[dict], keyword_hits: list[dict], limit: int) -> list[dict]:
    """Reciprocal Rank Fusion over chunk_id. Hits are ranked best-first with `chunk_id`, `score`.

    Order: fused_score desc, then semantic_rank (missing last), then chunk_id.
    fused_rank is 1-based over the full fused list; the top `limit` rows are returned.
    """
    rows: dict[str, dict] = {}
    for route, hits in (("semantic", semantic_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            row = rows.setdefault(
                hit["chunk_id"],
                {
                    "chunk_id": hit["chunk_id"],
                    "semantic_score": None,
                    "semantic_rank": None,
                    "keyword_score": None,
                    "keyword_rank": None,
                    "fused_score": 0.0,
                },
            )
            if row[f"{route}_rank"] is not None:
                continue  # a route lists a chunk once; ignore duplicates
            row[f"{route}_score"] = hit["score"]
            row[f"{route}_rank"] = rank
            row["fused_score"] += WEIGHTS[route] / (RRF_K + rank)
    ordered = sorted(
        rows.values(),
        key=lambda r: (
            -r["fused_score"],
            r["semantic_rank"] if r["semantic_rank"] is not None else float("inf"),
            r["chunk_id"],
        ),
    )
    for position, row in enumerate(ordered, start=1):
        row["fused_rank"] = position
    return ordered[:limit]


def _ensure_keyword_ready() -> None:
    global _keyword_ready
    if _keyword_ready:
        return
    message = (
        f"Keyword index '{schema.KEYWORD_INDEX_NAME}' is missing or not queryable; "
        "run `uv run python -m building_with_rag.ingestion.keyword_index`."
    )
    try:
        coll = semantic._db()[schema.CHUNKS_COLLECTION]
        indexes = list(coll.list_search_indexes(schema.KEYWORD_INDEX_NAME))
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    if not indexes or not indexes[0].get("queryable"):
        raise semantic._not_ready(message)
    _keyword_ready = True


def _keyword_search(question: str, filters: dict[str, list[str]], depth: int) -> list[dict]:
    clauses = [{"in": {"path": field, "value": values}} for field, values in filters.items() if values]
    compound: dict = {"must": [{"text": {"query": question, "path": "text"}}]}
    if clauses:
        compound["filter"] = clauses
    pipeline = [
        {"$search": {"index": schema.KEYWORD_INDEX_NAME, "compound": compound}},
        {"$limit": depth},
        {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "searchScore"}}},
    ]
    try:
        return list(semantic._db()[schema.CHUNKS_COLLECTION].aggregate(pipeline))
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB keyword search failed.") from None


def run_hybrid(request: QueryRequest) -> QueryResult:
    global _keyword_ready
    semantic._validate_scope(request)
    semantic._ensure_ready()
    _ensure_keyword_ready()
    filters = semantic._effective_filters(request)
    depth = route_depth(request.limit)
    candidates = semantic.num_candidates(depth)
    try:
        vector = semantic._embed_query(request.question)
        semantic_hits = semantic._search(vector, filters, depth, candidates)
        keyword_hits = _keyword_search(request.question, filters, depth)
        fused = fuse(semantic_hits, keyword_hits, request.limit)
        resolved, unresolved = semantic._resolve(
            [{"chunk_id": r["chunk_id"], "score": r["fused_score"]} for r in fused]
        ) if fused else ([], 0)
    except RetrievalError:
        semantic._ready = False  # re-check readiness after any failure
        _keyword_ready = False
        raise
    by_id = {row["chunk_id"]: row for row in fused}
    results = []
    for chunk in resolved:
        row = by_id[chunk.chunk_id]
        results.append(
            chunk.model_copy(
                update={
                    "score": row["fused_score"],
                    "semantic_score": row["semantic_score"],
                    "semantic_rank": row["semantic_rank"],
                    "keyword_score": row["keyword_score"],
                    "keyword_rank": row["keyword_rank"],
                    "fused_score": row["fused_score"],
                    "fused_rank": row["fused_rank"],
                }
            )
        )
    both = sum(1 for r in results if r.semantic_rank and r.keyword_rank)
    sem_only = sum(1 for r in results if r.semantic_rank and not r.keyword_rank)
    trace = {
        "mode": "hybrid",
        "query": request.question,
        "embedding": {
            "model": schema.EMBEDDING_MODEL,
            "input_type": "query",
            "dimensions": schema.EMBEDDING_DIMENSIONS,
        },
        "filters": filters,
        "caller_id": request.caller_id or semantic.get_settings().webui_demo_caller_id,
        "result_count": len(results),
        "unresolved_hits": unresolved,
        "semantic": {
            "index": schema.VECTOR_INDEX_NAME,
            "limit": depth,
            "num_candidates": candidates,
            "hit_count": len(semantic_hits),
        },
        "keyword": {
            "index": schema.KEYWORD_INDEX_NAME,
            "path": "text",
            "operator": "text",
            "limit": depth,
            "hit_count": len(keyword_hits),
        },
        "fusion": {"method": "rrf", "k": RRF_K, "weights": WEIGHTS, "route_depth": depth},
        "contribution": {
            "both": both,
            "semantic_only": sem_only,
            "keyword_only": len(results) - both - sem_only,
        },
    }
    if results:
        status, message = "ok", f"Returned {len(results)} passage(s). {_SCORE_NOTE}"
    else:
        status, message = "no_results", f"No passages matched the filters. {_SCORE_NOTE}"
    return QueryResult(pattern="hybrid", status=status, message=message, trace=trace, results=results)
