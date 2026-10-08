"""Query endpoint: shares retrieve + answer_events with the chat adapter."""

from fastapi import APIRouter, HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.pipeline import (
    REAL_PATTERNS,
    RetrievalError,
    answer_events,
    final_result,
    retrieve,
)

router = APIRouter()

_OUTCOME_MESSAGES = {
    "answered": "Answer generated from retrieved evidence.",
    "insufficient_evidence": "Retrieved evidence is insufficient to answer.",
    "unavailable": "Answer generation is unavailable; retrieval results are returned.",
    "malformed": "Answer generation returned an invalid response; retrieval results are returned.",
    "low_confidence": "Answer failed evidence checks (low confidence); see generation.issues.",
}


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    try:
        result = retrieve(request)
    except RetrievalError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}
        ) from None
    if request.pattern in REAL_PATTERNS and request.generate_answer:
        result.generation = final_result(answer_events(request.question, result))
        key = "low_confidence" if result.generation.confidence == "low" else result.generation.outcome
        result.message = f"{result.message} {_OUTCOME_MESSAGES[key]}"
    return result
