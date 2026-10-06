import json
import time
import uuid
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from building_with_rag.config import get_settings
from building_with_rag.contracts import (
    ChatCompletionRequest,
    QueryRequest,
    QueryResult,
    SemanticFilters,
)
from building_with_rag.modes import MODE_REGISTRY, MODEL_TO_MODE
from building_with_rag.service import run_pattern

app = FastAPI(title="Building with RAG")


class OpenAIError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str | None = None,
                 param: str | None = None, type_: str = "invalid_request_error"):
        self.message, self.status_code, self.code = message, status_code, code
        self.param, self.type_ = param, type_


def _envelope(message: str, type_: str, param: str | None, code: str | None) -> dict:
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


@app.exception_handler(OpenAIError)
async def _openai_error(_: Request, exc: OpenAIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.message, exc.type_, exc.param, exc.code),
    )


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    if request.url.path == "/v1/chat/completions":
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return JSONResponse(
            status_code=400,
            content=_envelope(first.get("msg", "Invalid request"), "invalid_request_error",
                              loc or None, "invalid_request"),
        )
    return await request_validation_exception_handler(request, exc)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest) -> QueryResult:
    return run_pattern(request)


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "building-with-rag"}
            for model_id in MODE_REGISTRY.values()
        ],
    }


def _to_query_request(body: ChatCompletionRequest) -> QueryRequest:
    pattern = MODEL_TO_MODE.get(body.model)
    if pattern is None:
        raise OpenAIError(f"Unknown model '{body.model}'.", 404, "model_not_found", "model")
    if body.n != 1:
        raise OpenAIError("Only n=1 is supported.", 400, "unsupported_n", "n")
    question = next((m.content for m in reversed(body.messages) if m.role == "user"), "").strip()
    if not question or len(question) > 4000:
        raise OpenAIError("A user message of 1-4000 characters is required.", 400,
                          "invalid_question", "messages")
    opts = body.rag_options
    filters = None
    kwargs: dict = {}
    if opts:
        filters = opts.filters
        kwargs = {"required_acts": opts.required_acts or [], "chapter": opts.chapter}
        if opts.limit is not None:
            kwargs["limit"] = opts.limit
    # Identity and generation settings are server-side only.
    return QueryRequest(
        question=question,
        pattern=pattern,
        caller_id=get_settings().webui_demo_caller_id,
        filters=filters,
        generate_answer=False,
        **kwargs,
    )


def _sse(model: str, completion_id: str, created: int, delta: dict,
         finish: str | None) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _stream(model: str, text: str) -> Iterator[str]:
    cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    yield _sse(model, cid, created, {"role": "assistant"}, None)
    yield _sse(model, cid, created, {"content": text}, None)
    yield _sse(model, cid, created, {}, "stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest):
    result = run_pattern(_to_query_request(body))
    text = result.message
    if body.stream:
        return StreamingResponse(_stream(body.model, text), media_type="text/event-stream")
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
