# Story 1.1 — Architecture and project seed

Status: ready for implementation
Audience: a later coding assistant, working in a three-day classroom RAG course.

## Purpose

Build one small application: a FastAPI service that later stories grow into a RAG system over the BNS and IPC. This story does two things, in order:

1. Write `docs/architecture.md` and get the instructor to approve it.
2. After approval, seed the Python project, the shared contracts, and the placeholder endpoints.

Do not build a custom chat frontend or a second demo. Open WebUI (supplied by the trainer) is the only chat client.

## Prerequisites

- Python 3.12 and UV installed.
- Project path: read `docs/config.yaml` if it exists and use the project path it configures. If it does not exist, use the repository root. (At the time this story was written, no `docs/config.yaml`, `docs/architecture.md`, or other stories existed.)
- Supplied data is already present in `data/raw/`, including `data/raw/PROVENANCE.md`. Preserve it. Never delete or rewrite supplied files.
- The trainer's Open WebUI bundle (ZIP shared over the local LAN).
- Later stories (not this one) need: an Atlas free-tier (M0) `MONGODB_URI` whose IP access list allows your machine; a free Voyage key (rate limited, so Story 2.2 embedding takes about 40 minutes); and trainer-supplied `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` for an OpenAI-compatible LiteLLM proxy (needed from Story 3.1).

## Work to do

### Phase A — Architecture (stop for approval)

1. Read `docs/config.yaml`, `docs/architecture.md`, the repository layout, and existing stories. Preserve existing work.
2. Create or update `docs/architecture.md`. Keep it short and in plain English. It must record:
   - **Evidence rule:** BNS and IPC documents are the only future answer evidence. An answer must not claim support without retrieved evidence.
   - **Act-qualified identifiers:** always pair the act with the section (for example `BNS:103`, `IPC:302`) so the two acts are never confused.
   - **Provenance:** supplied provenance is `data/raw/PROVENANCE.md`.
   - **Passages are data:** retrieved passages are evidence, never application instructions.
   - **Light trust-boundary notes:** validate API input; preserve source origin; never put secrets in code, responses, or logs. Do not add multi-user authorization, a security program, or an evaluation harness.
   - **Fixed embedding choice:** Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding. Later stories reuse these names without renaming or adding provider-specific alternatives.
   - **Derived data:** `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`.
   - **Course modes and model IDs:** the registry described below.
   - **Contracts:** the endpoints and models described below, extended additively by later stories.
3. **STOP.** Ask the instructor to approve `docs/architecture.md`. Until approval is explicit, do not create seed files, install dependencies, or scaffold code.
4. Resume this same story only after explicit approval.

### Phase B — Seed (only after approval)

**Project files.** Create `src/building_with_rag/`, `tests/`, `.env.example`, `.gitignore`, `pyproject.toml`, and `uv.lock`. Use Python 3.12 and UV with FastAPI, Pydantic settings, PyMongo (for later use only), Ruff, and a minimal test runner (pytest). Preserve `data/raw/`.

**`.env.example`** is the classroom's canonical file. Use exactly these values:

```
APP_ENV=development
MONGODB_URI=
VOYAGE_API_KEY=
CAPSTONE_API_KEY=
GENERATION_API_BASE_URL=
GENERATION_API_KEY=
RERANK_API_KEY=
MONGODB_DB_NAME=building_with_rag
MONGODB_TEST_DB_NAME=building_with_rag_test
WEBUI_DEMO_CALLER_ID=demo-public
GENERATION_MODEL_NAME=gpt-4o-mini
RERANK_API_BASE_URL=https://api.voyageai.com/v1
RERANK_MODEL_NAME=rerank-2.5
RERANK_REQUEST_TIMEOUT_SECONDS=30
RERANK_CANDIDATE_LIMIT=20
RERANK_SEND_LIMIT=10
RERANK_RETURN_LIMIT=5
```

Document (in `.gitignore` and a short note in the README or `.env.example` comments) that `.env` is untracked and secrets are never committed. `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` stay blank in `.env.example`; fill them in `.env` only when Story 3.1 needs them.

**Application.** It must start with no database or model credentials. Expose a safe `GET /healthz` that reveals no secrets and touches no external service.

**Mode registry.** One shared registry containing only these modes and model IDs:

| Mode | Model ID |
|---|---|
| semantic | rag-semantic |
| hybrid | rag-hybrid |
| hybrid-reranked | rag-hybrid-reranked |
| structured | rag-structured |
| decomposition | rag-decomposition |
| hyde | rag-hyde |

Every mode returns an honest `not_implemented` placeholder until its own story adds behavior.

**Shared contracts** (typed Pydantic models, no behavior now; later stories preserve and extend them additively and never replace them with simplified alternatives). Cover the request, result, generation, OpenAI, and MongoDB-schema contracts. Use exactly these names:

- `QueryRequest`: `question` (1–4,000 characters), `pattern`, optional `caller_id`, optional `SemanticFilters` (`act`, `status`, `access_level`, each a list), `limit` (default 5, range 1–20), `generate_answer` (default false), `required_acts`, `chapter`. The seed resolves only its fixed local demo caller, but keeps `caller_id`.
- `QueryResult`: `pattern`, `status`, `message`, `trace`, `results`, optional `generation`, plus additive empty-by-default fields `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`. Do not add outcome, evidence, answer, confidence, citations, or diagnostics as parallel top-level fields.
- `RetrievedChunk`: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only the fields named in their own handouts: `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`, `rerank_score`, `rerank_rank`.
- `omitted_candidates` items carry `chunk_id` and `omitted_reason`.
- `GenerationResult`: `outcome` (`answered`, `insufficient_evidence`, `unavailable`, `malformed`), `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`, and the confidence fields `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`.
- `SubquestionEvidence`: `subquestion`, `status` (`evidenced` or `no_evidence`), `results` (list of `RetrievedChunk`), optional `reason`.
- `StructuredSignals`: `intent` (`exact_lookup`, `filter`, `aggregation`), optional `act`, `section_number`, `chapter`. Story 5.1 fills these in.
- `ChatCompletionRequest` (text-only): `model`, `messages` (roles `system`, `developer`, `user`, `assistant`), `stream`, `n`, and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).

**Endpoints.**

- `POST /v1/query`: validate `QueryRequest`, call the shared `run_pattern` path, return `QueryResult`. This endpoint owns the diagnostics.
- `GET /v1/models`: list the six model IDs in OpenAI format.
- `POST /v1/chat/completions`: map the selected model to the same `QueryRequest` and the same `run_pattern` path as `/v1/query`. Set the demo `caller_id` (from `WEBUI_DEMO_CALLER_ID`) and `generate_answer` on the server; never take identity, access level, or generation settings from the client. Support normal OpenAI Chat Completions JSON, and SSE with role/content/stop frames ending in `[DONE]`. Return the OpenAI-style error envelope for errors before streaming starts. Open WebUI gets only normal answer text derived from the same `QueryResult`. Do not duplicate implementations or invent custom SSE events.

Later stories render final confidence, sources, and low-confidence warnings as clearly labelled text after the answer, while keeping the full `GenerationResult` in `QueryResult.generation`.

**Open WebUI (separate client).**

- The trainer shares the bundle ZIP over the LAN. Extract it and run its setup script exactly once, before using the classroom project:
  - Windows (primary path), from the bundle root: `powershell -ExecutionPolicy Bypass -File .\setup_open_webui.ps1`
  - macOS/Linux, from the bundle root: `sh setup_open_webui.sh` (needs internet on first install).
- The script installs the pinned Open WebUI, writes course settings, starts a loopback-only service, and provisions the RAG options Filter and the Building with RAG Pipe.
- Do not hand-install Open WebUI, create accounts, edit the admin panel, change either Function, or rerun setup to reload anything. If setup fails, report its exact output and stop.
- Day-to-day start/stop/status uses the supplied `manage_open_webui` script.
- The pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`) to `/v1/chat/completions`. It never sends identity, access level, or generation settings. The seed adapter must accept exactly that request.

**Out of scope.** Retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, broad deployment, aggressive testing, multi-user authorization, a security program, an evaluation harness.

## Completion checks

Keep these light. Run and record each:

1. `uv sync` succeeds; `uv run ruff check .` is clean; `uv run pytest` runs (a minimal test is enough).
2. Start the app with no credentials set. It starts.
3. `GET /healthz` returns a safe OK.
4. One `POST /v1/query` with `pattern: semantic` returns a `QueryResult` with a `not_implemented` status and a clear message.
5. `GET /v1/models` lists exactly the six model IDs.
6. `POST /v1/chat/completions` returns the placeholder as JSON (`stream: false`) and as SSE (`stream: true`, ending in `[DONE]`).
7. Open WebUI smoke check: start the capstone API, open http://127.0.0.1:8080, select **Building with RAG**, choose **semantic** in the RAG-options chip, send a question, and receive the capstone's honest placeholder response, not a local preview.
8. `.env` is untracked; `.env.example` matches the values above; no secrets appear in code, responses, or logs.

## Handover

Fill in when done:

- **Architecture approval:** who approved, and when.
- **Files created:** list every file this story created (expected: `docs/architecture.md`, `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `src/building_with_rag/...`, `tests/...`).
- **Commands actually run:** list each command with a one-line result. Do not list commands that were not run.
- **Open WebUI result:** what was selected, what was sent, and exactly what came back.
- **Notes for later stories:** anything the next story must know. Later stories reuse the names, model IDs, and contracts above unchanged and extend them additively.
