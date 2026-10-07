"""Semantic retrieval: embed the question, run Atlas $vectorSearch, resolve source passages."""

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from voyageai import Client as VoyageClient

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

# numCandidates = limit * 10, clamped to [50, 200] (always >= limit since limit <= 20).
CANDIDATE_MULTIPLIER = 10
MIN_CANDIDATES = 50
MAX_CANDIDATES = 200

_MONGO_TIMEOUT_MS = 5000
_MONGO_SOCKET_TIMEOUT_MS = 20000
_VOYAGE_TIMEOUT_S = 20

_SCORE_NOTE = "Scores rank similarity only; they do not prove a passage is correct or answers the question."

_mongo: MongoClient | None = None
_voyage: VoyageClient | None = None
_ready = False


class RetrievalError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def num_candidates(limit: int) -> int:
    return min(MAX_CANDIDATES, max(limit, MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER))


def _db():
    global _mongo
    settings = get_settings()
    if _mongo is None:
        _mongo = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=_MONGO_TIMEOUT_MS,
            connectTimeoutMS=_MONGO_TIMEOUT_MS,
            socketTimeoutMS=_MONGO_SOCKET_TIMEOUT_MS,
        )
    name = settings.mongodb_test_db_name if settings.app_env == "testing" else settings.mongodb_db_name
    return _mongo[name]


def _voyage_client() -> VoyageClient:
    global _voyage
    if _voyage is None:
        _voyage = VoyageClient(api_key=get_settings().voyage_api_key, max_retries=1, timeout=_VOYAGE_TIMEOUT_S)
    return _voyage


def _not_ready(message: str) -> RetrievalError:
    return RetrievalError(503, "retrieval_not_ready", message)


def _ensure_ready() -> None:
    """Cheap readiness check; a positive result is cached per process."""
    global _ready
    if _ready:
        return
    settings = get_settings()
    if not settings.mongodb_uri:
        raise _not_ready("MONGODB_URI is not set.")
    if not settings.voyage_api_key:
        raise _not_ready("VOYAGE_API_KEY is not set.")
    try:
        coll = _db()[schema.EMBEDDINGS_COLLECTION]
        indexes = list(coll.list_search_indexes(schema.VECTOR_INDEX_NAME))
        if not indexes or not indexes[0].get("queryable"):
            raise _not_ready(
                f"Vector index '{schema.VECTOR_INDEX_NAME}' is missing or not queryable; "
                "run `uv run python -m building_with_rag.ingestion.ingest`."
            )
        sample = coll.find_one({}, {"model": 1, "model_version": 1, "dimensions": 1})
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    if sample is None:
        raise _not_ready("The embeddings collection is empty; run Story 2.2 ingestion.")
    if (
        sample.get("model") != schema.EMBEDDING_MODEL
        or sample.get("model_version") != schema.EMBEDDING_MODEL_VERSION
        or sample.get("dimensions") != schema.EMBEDDING_DIMENSIONS
    ):
        raise _not_ready("Stored embeddings do not match the configured model/dimensions.")
    _ready = True


def _effective_filters(request: QueryRequest) -> dict[str, list[str]]:
    caller = request.filters
    return {
        "act": list(caller.act) if caller else [],
        "status": list(caller.status) if caller else [],
        "access_level": [schema.DEFAULT_ACCESS_LEVEL],  # server-fixed; callers cannot widen
    }


def _validate_scope(request: QueryRequest) -> None:
    demo = get_settings().webui_demo_caller_id
    if request.caller_id is not None and request.caller_id != demo:
        raise RetrievalError(422, "invalid_caller", "caller_id is not permitted.")
    if request.required_acts is not None:
        raise RetrievalError(422, "unsupported_option", "required_acts is not supported by semantic mode.")
    if request.chapter is not None:
        raise RetrievalError(422, "unsupported_option", "chapter is not supported by semantic mode.")


def _embed_query(question: str) -> list[float]:
    try:
        result = _voyage_client().embed([question], model=schema.EMBEDDING_MODEL, input_type="query")
    except Exception:  # noqa: BLE001 - Voyage/HTTP failures; never leak details or keys
        raise RetrievalError(502, "retrieval_upstream_error", "Voyage embedding request failed.") from None
    vector = result.embeddings[0]
    if len(vector) != schema.EMBEDDING_DIMENSIONS:
        raise _not_ready(f"Query embedding has {len(vector)} dimensions, expected {schema.EMBEDDING_DIMENSIONS}.")
    return vector


def _search(vector: list[float], filters: dict[str, list[str]], limit: int, candidates: int) -> list[dict]:
    clauses = [{field: {"$in": values}} for field, values in filters.items() if values]
    stage: dict = {
        "index": schema.VECTOR_INDEX_NAME,
        "path": "vector",
        "queryVector": vector,
        "numCandidates": candidates,
        "limit": limit,
    }
    if clauses:
        stage["filter"] = {"$and": clauses}
    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    try:
        return list(_db()[schema.EMBEDDINGS_COLLECTION].aggregate(pipeline))
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB vector search failed.") from None


def _resolve(hits: list[dict]) -> tuple[list[RetrievedChunk], int]:
    db = _db()
    try:
        chunks = {c["chunk_id"]: c for c in db[schema.CHUNKS_COLLECTION].find({"chunk_id": {"$in": [h["chunk_id"] for h in hits]}})}
        section_ids = list({c["section_id"] for c in chunks.values()})
        sections = {s["_id"]: s for s in db[schema.SECTIONS_COLLECTION].find({"_id": {"$in": section_ids}})}
    except PyMongoError:
        raise RetrievalError(502, "retrieval_upstream_error", "MongoDB request failed.") from None
    results: list[RetrievedChunk] = []
    unresolved = 0
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is None or section is None:
            unresolved += 1
            continue
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                section_id=section["section_id"],
                act=section["act"],
                text=chunk["text"],
                heading=section.get("heading") or "",
                score=hit["score"],
                chunk_index=chunk.get("chunk_index"),
                act_label=section.get("act_label"),
                status=section.get("status"),
                chapter=section.get("chapter"),
                chapter_title=section.get("chapter_title"),
                section_number=section.get("section_number"),
                source_pdf=section.get("source_pdf"),
                source_sha256=section.get("source_sha256"),
                needs_review=section.get("needs_review"),
            )
        )
    return results, unresolved


def run_semantic(request: QueryRequest) -> QueryResult:
    _validate_scope(request)
    _ensure_ready()
    filters = _effective_filters(request)
    candidates = num_candidates(request.limit)
    try:
        vector = _embed_query(request.question)
        hits = _search(vector, filters, request.limit, candidates)
        results, unresolved = _resolve(hits) if hits else ([], 0)
    except RetrievalError:
        global _ready
        _ready = False  # re-check readiness after any failure
        raise
    trace = {
        "mode": "semantic",
        "query": request.question,
        "embedding": {
            "model": schema.EMBEDDING_MODEL,
            "input_type": "query",
            "dimensions": schema.EMBEDDING_DIMENSIONS,
        },
        "index": schema.VECTOR_INDEX_NAME,
        "limit": request.limit,
        "num_candidates": candidates,
        "filters": filters,
        "caller_id": request.caller_id or get_settings().webui_demo_caller_id,
        "result_count": len(results),
        "unresolved_hits": unresolved,
    }
    if results:
        status, message = "ok", f"Returned {len(results)} passage(s). {_SCORE_NOTE}"
    else:
        status, message = "no_results", f"No passages matched the filters. {_SCORE_NOTE}"
    return QueryResult(pattern="semantic", status=status, message=message, trace=trace, results=results)
