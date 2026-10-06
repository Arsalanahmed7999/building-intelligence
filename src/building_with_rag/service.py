from building_with_rag.contracts import QueryRequest, QueryResult


def run_pattern(request: QueryRequest) -> QueryResult:
    """Shared path for every endpoint. Seed: honest placeholder for every mode."""
    return QueryResult(
        pattern=request.pattern,
        status="not_implemented",
        message=f"The '{request.pattern}' mode is not implemented yet.",
        trace=[f"mode={request.pattern}", "placeholder"],
    )
