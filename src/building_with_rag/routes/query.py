"""Query endpoint: semantic goes to the retrieval module; other modes use run_pattern."""

from fastapi import APIRouter, HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import RetrievalError, run_semantic

router = APIRouter()


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern == Pattern.SEMANTIC:
        try:
            return run_semantic(request)
        except RetrievalError as exc:
            raise HTTPException(
                status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}
            ) from None
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
