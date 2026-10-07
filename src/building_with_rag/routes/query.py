"""Query endpoint: semantic goes to the retrieval module; other modes use run_pattern."""

from fastapi import APIRouter, HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.answer import generate_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import RetrievalError, run_semantic

router = APIRouter()

_OUTCOME_MESSAGES = {
    "answered": "Answer generated from retrieved evidence.",
    "insufficient_evidence": "Retrieved evidence is insufficient to answer.",
    "unavailable": "Answer generation is unavailable; retrieval results are returned.",
    "malformed": "Answer generation returned an invalid response; retrieval results are returned.",
}


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern == Pattern.SEMANTIC:
        try:
            result = run_semantic(request)
        except RetrievalError as exc:
            raise HTTPException(
                status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}
            ) from None
        if request.generate_answer:
            result.generation = generate_answer(request.question, result)
            result.message = f"{result.message} {_OUTCOME_MESSAGES[result.generation.outcome]}"
        return result
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
