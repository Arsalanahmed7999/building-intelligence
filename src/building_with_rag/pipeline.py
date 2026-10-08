"""One shared path for /v1/query and /v1/chat/completions:
retrieve -> context assembly -> generation -> validation -> final GenerationResult."""

from collections.abc import Iterator

from building_with_rag.contracts import GenerationResult, QueryRequest, QueryResult
from building_with_rag.generation.answer import Event, stream_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.semantic import RetrievalError, run_semantic

__all__ = ["REAL_PATTERNS", "RetrievalError", "answer_events", "chat_pieces", "final_result", "render_footer", "retrieve"]

UNAVAILABLE_LINE = "Answer generation unavailable — the text above is an unchecked draft."
UNCHECKED_LINE = "The answer could not be checked — the text above is an unchecked draft."
LOW_CONFIDENCE_LINE = "DRAFT — low confidence, not the final answer."

# Patterns with real retrieval + grounded generation; every other mode is a placeholder.
REAL_PATTERNS = frozenset({Pattern.SEMANTIC, Pattern.HYBRID})


def retrieve(request: QueryRequest) -> QueryResult:
    """Semantic and hybrid run real retrieval; other modes keep their placeholder."""
    if request.pattern == Pattern.SEMANTIC:
        return run_semantic(request)
    if request.pattern == Pattern.HYBRID:
        return run_hybrid(request)
    return QueryResult(**run_pattern(request.pattern, request.question, request.caller_id))


def answer_events(question: str, retrieval: QueryResult) -> Iterator[Event | GenerationResult]:
    """Text/notice events, then the one final GenerationResult."""
    yield from stream_answer(question, retrieval)


def final_result(events: Iterator[Event | GenerationResult]) -> GenerationResult:
    result = None
    for item in events:
        if isinstance(item, GenerationResult):
            result = item
    return result


def _source_line(c) -> str:
    act = c.act.split("_")[0]
    parts = [c.label, f"{act} §{c.section_number}" if c.section_number else act]
    if c.heading:
        parts.append(c.heading)
    parts.append(c.section_id)
    return " · ".join(parts)


def render_footer(g: GenerationResult) -> str:
    """Readable text shown after the streamed draft, derived from the final result."""
    if g.outcome == "answered":
        sources = "\n".join(_source_line(c) for c in g.citations)
        return f"\n\nEvidence check passed — confidence: high\nSources:\n{sources}"
    if g.outcome == "insufficient_evidence":
        reason = str(g.trace.get("reason") or "").strip()[:200]
        sentence = "The retrieved evidence is insufficient to answer this question."
        return f"{sentence} {reason}".strip()
    if g.outcome == "unavailable":
        return f"\n\n{UNAVAILABLE_LINE}"
    if g.confidence == "low":
        last = max((i["attempt"] for i in g.issues), default=None)
        details = "\n".join(
            f"- {i['check']}: {i['detail'][:120]}" for i in g.issues if i["attempt"] == last
        )
        return f"\n\n{LOW_CONFIDENCE_LINE}\n{g.low_confidence_reason}\n{details}"
    return f"\n\n{UNCHECKED_LINE}"


def chat_pieces(question: str, retrieval: QueryResult) -> Iterator[str]:
    """Text pieces for chat: streamed draft text/notices, then the footer from the final result."""
    if Pattern(retrieval.pattern) not in REAL_PATTERNS:
        yield retrieval.message
        return
    final = None
    for item in answer_events(question, retrieval):
        if isinstance(item, GenerationResult):
            final = item
        else:
            yield item.text
    retrieval.generation = final
    yield render_footer(final)
