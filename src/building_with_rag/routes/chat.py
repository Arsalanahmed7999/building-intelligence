"""OpenAI-compatible chat adapter.

Maps the selected rag-<pattern> model to the same QueryRequest and the same shared
pipeline (retrieve + answer_events) as /v1/query, with server-side demo caller_id and
generate_answer. Only answer text streams; role/content/stop/[DONE] framing, no custom
SSE events. The OpenAI-style error envelope applies before streaming begins.
"""

import hmac
import json
import time
import uuid

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from building_with_rag.contracts import ChatCompletionRequest, QueryRequest, SemanticFilters
from building_with_rag.pipeline import RetrievalError, chat_pieces, retrieve
from building_with_rag.registry import MODEL_ID_TO_PATTERN
from building_with_rag.settings import get_settings

router = APIRouter()

_FINISH_STOP = "stop"
_DONE = "[" + "DONE" + "]"


def _error(status: int, message: str, code: str, type_: str = "invalid_request_error"):
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": type_, "code": code}},
    )


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _check_api_key(authorization: str | None) -> None:
    expected = get_settings().capstone_api_key
    if not expected:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        token.strip().encode(), expected.encode()
    ):
        raise _error(401, "Invalid API key.", "invalid_api_key")


def _build_query_request(request: ChatCompletionRequest) -> QueryRequest:
    pattern = MODEL_ID_TO_PATTERN.get(request.model)
    if pattern is None:
        raise _error(400, f"Model '{request.model}' not found.", "model_not_found")
    latest_user = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if latest_user is None:
        raise _error(400, "At least one user message is required.", "missing_user_message")
    options = request.rag_options
    nested = options.filters if options and options.filters else None
    return QueryRequest(
        question=latest_user,
        pattern=pattern,
        caller_id=get_settings().webui_demo_caller_id,  # server-side demo caller
        filters=(
            SemanticFilters(
                act=options.act or (nested.act if nested else []),
                status=options.status or (nested.status if nested else []),
                access_level=options.access_level,
            )
            if options
            else None
        ),
        limit=options.limit if options else 5,
        generate_answer=True,  # server-side decision; adapter never trusts client
        required_acts=options.required_acts if options else None,
        chapter=options.chapter if options else None,
    )


def _sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, authorization: str | None = Header(None)):
    _check_api_key(authorization)
    query_request = _build_query_request(request)
    try:
        retrieval = retrieve(query_request)
    except RetrievalError as exc:
        raise _error(exc.status_code, exc.message, exc.code, "server_error") from None
    pieces = chat_pieces(query_request.question, retrieval)
    base = {"id": _completion_id(), "created": int(time.time()), "model": request.model}

    if not request.stream:
        text = "".join(pieces)
        return {
            **base,
            "object": "chat.completion",
            "choices": [
                {
                    "index": i,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _FINISH_STOP,
                }
                for i in range(request.n)
            ],
        }

    def chunk(delta: dict, finish: str | None = None) -> str:
        return _sse_frame(
            {
                **base,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": i, "delta": delta, "finish_reason": finish} for i in range(request.n)
                ],
            }
        )

    def generate():
        yield chunk({"role": "assistant"})
        for piece in pieces:
            if piece:
                yield chunk({"content": piece})
        yield chunk({}, _FINISH_STOP)
        yield "data: " + _DONE + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
