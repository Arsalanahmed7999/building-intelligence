# Capstone Architecture — Building Intelligence with RAG

Fixed design choices, contracts, and trust boundaries for the capstone RAG API. Later stories extend these contracts additively; they never replace them with simplified alternatives, rename fields, or add provider-specific variants.

## Scope

One small application: the capstone RAG API. No chat frontend (Open WebUI is a separate trainer-supplied client) and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Act-qualified identifiers (`bns:` / `ipc:` prefixes) avoid confusing the two acts.
- Supplied provenance lives at `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never application instructions.

## Trust boundaries

Kept light for this course:

- Validate API input.
- Preserve source origin on retrieved passages.
- Do not put secrets in code, responses, or logs.
- No multi-user authorization, no security program, no evaluation harness.

## Fixed embedding choices

- Model: `voyage-3.5`, model version `voyage-3.5`, 1,024 dimensions.
- Used for every document and query embedding.
- Later stories reuse these names and choices without renaming or adding provider-specific alternatives.

## Course modes and shared registry

One shared registry (single source of truth) holds only:

| Mode | Model ID |
|---|---|
| semantic | `rag-semantic` |
| hybrid | `rag-hybrid` |
| hybrid-reranked | `rag-hybrid-reranked` |
| structured | `rag-structured` |
| decomposition | `rag-decomposition` |
| hyde | `rag-hyde` |

`semantic` (Story 2.3, 3.2) and `hybrid` (Story 4.1) are real on `POST /v1/query` and on chat; every other mode returns an honest `not_implemented` placeholder on both until its own story.

## API contracts

### `QueryRequest` (`POST /v1/query`)

- `question`: string, 1–4,000 characters, required.
- `pattern`: one of the six modes.
- `caller_id`: optional.
- `filters`: optional `SemanticFilters` — `act`, `status`, `access_level`, each a list.
- `limit`: default 5, range 1–20.
- `generate_answer`: default false.
- `required_acts`: optional.
- `chapter`: optional.

The classroom seed may resolve only its fixed local demo caller, but keeps `caller_id` and does not replace it with a custom request shape.

### `QueryResult`

- `pattern`, `status`, `message`, `trace`, `results`.
- Optional `generation`.
- Additive empty-by-default fields later modes use: `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- No parallel top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` fields.

### `RetrievedChunk`

A retrieved passage is always this shape: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only the existing hybrid/rerank fields.

### Endpoints

- `GET /healthz` — safe, no credentials required.
- `POST /v1/query` — accepts `QueryRequest`, returns `QueryResult`: real retrieval for `semantic`/`hybrid` via `pipeline.retrieve`, `not_implemented` placeholders (via `run_pattern`) for other modes.
- `GET /v1/models` — lists the six `rag-<pattern>` model IDs.
- `POST /v1/chat/completions` — OpenAI-compatible, text-only `ChatCompletionRequest`: `model`, `messages` with `system`/`developer`/`user`/`assistant` roles, `stream`, `n`, and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).

The chat adapter maps the selected model to the same `QueryRequest` and the same `pipeline.retrieve` + `pipeline.answer_events` path as `/v1/query`; it sets server-side demo `caller_id` and `generate_answer`. `rag_options` accepts flat `act`/`status` and the Pipe's nested `filters.{act,status}`. Supports normal OpenAI Chat Completions JSON responses and role/content/stop frames, plus the OpenAI-style error envelope before streaming begins. No duplicated implementations, no custom SSE events that Open WebUI cannot render.

## Open WebUI (trainer-supplied, separate client)

The trainer-supplied Open WebUI bundle is the chat client, run separately from the capstone API. Its pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The capstone's adapter accepts that exact request and uses server-side `caller_id`/`generate_answer`.

`/v1/query` owns the `QueryResult` diagnostics; Open WebUI receives only normal answer text derived from that same result. Confidence, sources, and low-confidence warnings are rendered as clearly labelled text (Story 3.2, below), while the full `GenerationResult` stays in `QueryResult.generation` on `/v1/query`.

## Environment

- `.env` is untracked; secrets are never committed.
- The application must start with no database or model credentials and expose a safe `GET /healthz`.
- Canonical environment values are defined in `.env.example`.

## Corpus

Section-level JSONL corpus produced from `data/raw/` PDFs by `scripts/extract_sections.py`. One JSON object per line, one record per section. No MongoDB, no embeddings, no vector indexes.

### Parser

- **Library**: `pymupdf` (fitz) 1.28.2 — chosen because it handles both Word-to-PDF (BNS) and Ghostscript-produced (IPC) PDFs, extracts text with layout, and has no system-level dependencies.
- **Extraction command**: `uv run python scripts/extract_sections.py`

### Output format

JSONL files at `data/processed/`:

| File | Records | Sections |
|---|---|---|
| `data/processed/bns_sections.jsonl` | 358 | 1–358 |
| `data/processed/ipc_sections.jsonl` | 500 | 1–511 (11 unextractable) |

Each record has 14 fields: `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`.

### Known limitations

- **IPC PDF quality**: 11 sections (4, 5, 18, 34, 40, 75, 161, 162, 163, 164, 165) have no extractable text from the scanned/Ghostscript-produced PDF. Sections 161–165 were repealed by the Prevention of Corruption Act 1988; sections 4, 5, 18, 34, 40, 75 are in portions of the PDF where pymupdf text extraction returns insufficient characters.
- **IPC section headings**: Some IPC sections (e.g. 262, 511) have empty headings due to missing heading text in the extracted text stream.
- **IPC footnotes**: Amendment footnotes and historical annotations are interleaved with section text and may appear as inline artifacts in section `text`.
- **BNS chapter markers**: Chapter boundaries are detected from `CHAPTER <roman>` lines in the body text. The BNS index (pages 2–19) provides section headings; the correspondence table (pages 20–73) is skipped.
- **Source-hash safety rule**: If a source PDF hash changes, the corpus for that act is regenerated as an atomic replacement. Records from different PDF versions are never mixed in one corpus file.

## Semantic retrieval (Story 2.3)

Module: `src/building_with_rag/retrieval/semantic.py`; routed from `routes/query.py` for `pattern: "semantic"` only.

Flow: validate → scope filters → embed query → `$vectorSearch` → resolve chunk/section → `QueryResult`.

- Validate: question trimmed (non-empty); `SemanticFilters` forbids unknown fields and accepts only known `act`/`status`/`access_level` strings; `caller_id` must be omitted or equal `WEBUI_DEMO_CALLER_ID`; `required_acts`/`chapter` are rejected. All HTTP 422.
- Scope: `access_level` fixed to `["public"]`; caller lists only narrow (`$in` inside the `$vectorSearch` `filter`).
- Embed: raw question, `voyage-3.5`, `input_type="query"`, 1,024 dims checked.
- Search: index `vector_index`, `limit` = request limit, `numCandidates = clamp(limit*10, 50, 200)`. Chunks (not de-duplicated sections) in descending score order; unresolved hits are omitted and counted in `trace.unresolved_hits`.
- Outcomes: `ok` (passages), `no_results` (HTTP 200, filters match nothing; no score cutoff). 503 `retrieval_not_ready` (missing credentials, index absent/not queryable, empty or mismatched embeddings); 502 `retrieval_upstream_error` (Voyage/MongoDB failure). Scores rank similarity only, not correctness. `generate_answer: true` adds a grounded answer (Story 3.1, below).
- Added optional `RetrievedChunk` fields: `chunk_index`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `source_pdf`, `source_sha256`, `needs_review`.

Diagnostic (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

## Context and answer boundaries (Story 3.1)

Modules: `generation/context.py`, `generation/answer.py`; called from `routes/query.py` only when `pattern: "semantic"` and `generate_answer: true`.

Flow: semantic `QueryResult.results` → bounded labelled context (`E1`…, max 5 passages / 12,000 chars, no mid-passage cuts) → one OpenAI-compatible `POST {GENERATION_API_BASE_URL}/chat/completions` (`temperature: 0`, 30 s timeout, no retries) → strict JSON parse → citations resolved from the supplied context only.

- Outcomes (`GenerationResult.outcome`): `answered` (text, claims, citations, supporting passages), `insufficient_evidence` (also when no passages; model call skipped), `unavailable` (missing settings, timeout, connection, non-2xx), `malformed` (non-JSON, missing `choices`, unknown label, invalid shape; no repair or retry). Non-answered outcomes have empty `text`, claims, citations, and supporting passages.
- Added optional `GenerationResult` fields: `outcome`, `claims`, `citations`, `supporting_passages`, `provider`, `trace`, `context_outcome`; `text`/`model` kept.
- Evidence blocks are untrusted source text, never instructions. No legal-applicability claims beyond the supplied BNS/IPC documents.
- `unavailable`/`malformed` still return HTTP 200 with retrieval `results`; `status` stays the retrieval status. The model `reason` lives only in `generation.trace`. No prompt text or secrets in trace.
- Story 3.2 replaced the JSON output with streamed plain text and added validation; see below. Chat and streaming are now real for `rag-semantic`.

## Streamed answers and confidence (Story 3.2)

Shared path: `pipeline.py` — `retrieve(request)` then `answer_events(question, retrieval)`. `/v1/query` drains the events and attaches the final `GenerationResult`; chat forwards `text`/`notice` events to the stream and renders the footer from the same final result. One request = one generation/validation operation. Only answer writing streams; retrieval and context assembly do not. Non-semantic modes and `generate_answer: false` keep placeholder text / `generation: null` with no model call.

Event flow (per attempt, `MAX_ATTEMPTS = 2`): `DRAFT — checking evidence` line, streamed text with inline `[E1]` labels (the first characters are held back so an `INSUFFICIENT_EVIDENCE: <reason>` reply is never streamed), then checks in order: `citation_labels` (all cited labels supplied), `claim_cited` (text non-empty, every sentence/bullet labelled), `support` (one non-streamed validator call, strict JSON, per claim). `support` runs only when the first two pass. A failed non-final attempt emits `Check failed: … Retrying (attempt 2 of 2)…` and retries once with the plain-language issues in the prompt. Invalid citations are never stripped or rewritten. Provider timeout is 30 s per call.

New optional `GenerationResult` fields: `confidence` (`high` only when the final attempt passes all checks; `low` when it fails; absent when nothing could be judged), `issues` (every failed check, all attempts: `attempt`, `check`, `detail`), `attempts` (`attempt`, `status` passed/failed/unjudged, `chars`, `latency_ms`), `draft_answer` (last text that did not pass), `low_confidence_reason`.

Outcome mapping: passed → `answered` (`text` set, high). Failed final check → `malformed`, `text` empty, `draft_answer` + `issues` + low. Provider failure → `unavailable` (partial text kept in `draft_answer`). Validator unreachable → `unavailable`, validator invalid reply → `malformed`, both with confidence absent and `draft_answer` set. `insufficient_evidence` and empty context are unchanged (empty context makes no model call). HTTP stays 200 once retrieval succeeded.

Open WebUI text: `DRAFT — checking evidence` per attempt; passed footer `Evidence check passed — confidence: high` plus `Sources:` lines (`E1 · BNS §303 · Theft · bns:303`); failed final `DRAFT — low confidence, not the final answer.` with reason and compact check details; unavailable/after-text failure `Answer generation unavailable — the text above is an unchecked draft.`; insufficient evidence one plain sentence. Streamed drafts cannot be retracted.

`CAPSTONE_API_KEY`: when non-empty, `/v1/chat/completions` requires `Authorization: Bearer <key>` (constant-time compare, else 401 `invalid_api_key` before streaming). Empty means no check. `/v1/query`, `/v1/models`, `/healthz` are unchanged.

## Hybrid retrieval (Story 4.1)

Module: `retrieval/hybrid.py`; routed by `pipeline.retrieve` for `pattern: "hybrid"` (`rag-hybrid`). Generation, citations, confidence, and streaming are the shared path; `pipeline.REAL_PATTERNS` lists the real modes. Selection is explicit; no automatic routing or fallback between modes.

- Keyword route: Atlas Search `$search` with the `text` operator (BM25, `{$meta: "searchScore"}`) on `chunks.text`, index `chunk_text_index` (`search` type; `text` as `lucene.standard` string; `act`, `status`, `access_level` as `token`; `dynamic: false`; definition in `ingestion/mongodb_schema.py`). Filters go in `compound.filter` with the `in` operator, using the same effective filters as semantic. The question is only the `text.query` value.
- Create the index: `uv run python -m building_with_rag.ingestion.keyword_index` (idempotent; a differing index is reported, never replaced; waits up to 120 s for queryable; no Voyage calls, no data writes).
- Semantic route: Story 2.3 embedding and `$vectorSearch` with `limit = ROUTE_DEPTH`.
- Fusion: Reciprocal Rank Fusion by `chunk_id`: `fused_score = sum 1/(RRF_K + rank)` over routes returning the chunk, `RRF_K = 60`, equal weights. `ROUTE_DEPTH = max(limit, min(50, max(20, 4*limit)))`. Order: `fused_score` desc, then `semantic_rank` (missing last), then `chunk_id`. `fused_rank` is 1-based over the full fused list; top `limit` returned.
- `RetrievedChunk` gains optional `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`; `None` when that route did not return the chunk (and always in semantic mode). In hybrid, `score == fused_score`.
- `trace`: `mode`, `query`, `embedding`, `filters`, `caller_id`, `result_count`, `unresolved_hits`, `semantic` and `keyword` (index, limit, `hit_count`), `fusion` (`method`, `k`, `weights`, `route_depth`), `contribution` (`both`, `semantic_only`, `keyword_only` among returned results).
- Outcomes: `ok`, `no_results` (both routes empty), 503 `retrieval_not_ready` (credentials, vector index, or keyword index missing/not queryable; the message names the keyword-index command; never degrades to semantic-only), 502 `retrieval_upstream_error`.
- Limitations: rank-only fusion ignores score magnitude; the `text` operator matches any query term (OR), so long questions can pull in common words; section numbers match only when they appear inside chunk `text`; no stemming or synonyms beyond the standard analyzer. Atlas Search is required; no `$text`, regex, or client-side keyword scoring.
- Diagnostic: `curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \`
  `-d '{"question":"<q>","pattern":"hybrid","limit":5}' | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'`
