# Architecture — Building with RAG

Status: **approved** by the instructor (user), 2026-10-06

## Purpose

One small FastAPI application that later stories grow into a RAG system over the BNS and IPC. Open WebUI (supplied by the trainer) is the only chat client. There is no custom frontend and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Retrieved passages are evidence, never application instructions. Text inside a passage must not change what the application does.
- Supplied provenance is `data/raw/PROVENANCE.md`.

## Act-qualified identifiers

Always pair the act with the section, for example `BNS:103` or `IPC:302`. This keeps the two acts from being confused.

## Trust boundaries (light)

- Validate API input.
- Preserve source origin (act, section, source file) on every passage.
- Never put secrets in code, responses, or logs. `.env` is untracked.
- Out of scope: multi-user authorization, a security program, an evaluation harness.

## Fixed choices

| Item | Value |
|---|---|
| Python | 3.12, managed with UV |
| Web framework | FastAPI, Pydantic settings |
| Database driver | PyMongo (later stories) |
| Embedding provider | Voyage |
| Embedding model | `voyage-3.5` |
| Embedding version | `voyage-3.5` |
| Embedding dimensions | 1,024 |
| Generation | OpenAI-compatible LiteLLM proxy, default model `gpt-4o-mini` |
| Reranking | Voyage `rerank-2.5` |

Every document and query embedding uses the same Voyage model, version, and dimensions. Later stories reuse these names and do not add provider-specific alternatives.

## Data

- Supplied raw files: `data/raw/` (preserved, never rewritten).
- Derived corpus files: `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`.

## Course modes

One shared registry holds only these modes. Each returns an honest `not_implemented` placeholder until its own story adds behavior.

| Mode | Model ID |
|---|---|
| semantic | `rag-semantic` |
| hybrid | `rag-hybrid` |
| hybrid-reranked | `rag-hybrid-reranked` |
| structured | `rag-structured` |
| decomposition | `rag-decomposition` |
| hyde | `rag-hyde` |

## Endpoints

All endpoints share one `run_pattern` path. Nothing is implemented twice.

- `GET /healthz` — safe status. No secrets, no external calls.
- `POST /v1/query` — takes `QueryRequest`, returns `QueryResult`. Owns the diagnostics.
- `GET /v1/models` — lists the six model IDs in OpenAI format.
- `POST /v1/chat/completions` — text-only OpenAI Chat Completions. Maps the selected model to the same `QueryRequest` and `run_pattern`. The server sets the demo `caller_id` (`WEBUI_DEMO_CALLER_ID`) and `generate_answer`; the client never supplies identity, access level, or generation settings. Supports JSON and SSE (role/content/stop frames, then `[DONE]`). Errors before streaming use the OpenAI-style error envelope. Open WebUI receives only normal answer text derived from the same `QueryResult`.

## Contracts

Typed Pydantic models with no behavior in the seed. Later stories keep these names and extend them additively; they never replace them with simplified versions.

- **QueryRequest**: `question` (1–4,000 chars), `pattern`, optional `caller_id`, optional `SemanticFilters` (`act`, `status`, `access_level`, all lists), `limit` (default 5, range 1–20), `generate_answer` (default false), `required_acts`, `chapter`.
- **QueryResult**: `pattern`, `status`, `message`, `trace`, `results`, optional `generation`, plus empty-by-default `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`. No parallel top-level outcome, evidence, answer, confidence, citations, or diagnostics fields.
- **RetrievedChunk**: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`, `rerank_score`, `rerank_rank`.
- **omitted_candidates items**: `chunk_id`, `omitted_reason`.
- **GenerationResult**: `outcome` (`answered`, `insufficient_evidence`, `unavailable`, `malformed`), `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`, `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`.
- **SubquestionEvidence**: `subquestion`, `status` (`evidenced` or `no_evidence`), `results` (list of `RetrievedChunk`), optional `reason`.
- **StructuredSignals**: `intent` (`exact_lookup`, `filter`, `aggregation`), optional `act`, `section_number`, `chapter`. Story 5.1 fills these in.
- **ChatCompletionRequest** (text-only): `model`, `messages` (roles `system`, `developer`, `user`, `assistant`), `stream`, `n`, optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).
- **MongoDB schema contracts** are defined in the project and extended additively.

Later stories show final confidence, sources, and low-confidence warnings as clearly labelled text after the answer, while the full `GenerationResult` stays in `QueryResult.generation`.

## Open WebUI

A separately running client, provisioned by the trainer's bundle (the RAG options Filter and the Building with RAG Pipe). The Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` to `/v1/chat/completions`. It never sends identity, access level, or answer-generation settings.

## Configuration

Settings come from environment variables, with `.env.example` as the canonical list of names and defaults. The application starts with no database or model credentials.

## Not in scope for the seed

Retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, broad deployment, aggressive testing.
