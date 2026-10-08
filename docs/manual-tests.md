# Manual tests

## Story 1.1 — Architecture and Project Seed

What it adds: FastAPI project seed with health, query, model-listing, and OpenAI-compatible chat endpoints — all returning honest `not_implemented` placeholders.

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Health

```bash
curl -s http://127.0.0.1:8000/healthz
```

Expected: `{"status":"ok"}`.

### List models

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

Expected: six lines: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.

### Query — each RAG mode (semantic)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic"}'
```

Expected (Story 2.3, needs `.env` credentials): `"status":"ok"` with ranked passages in `results`; see `docs/architecture.md` for the `jq` diagnostic command.

### Query — hybrid

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "criminal breach of trust", "pattern": "hybrid", "limit": 3}'
```

Expected (Story 4.1, needs the keyword index): `"status":"ok"`; see Story 4.1.

### Query — hybrid-reranked

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid-reranked`.

### Query — structured

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "structured"}'
```

Expected: `"status":"not_implemented"`, message references `structured`.

### Query — decomposition

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "decomposition"}'
```

Expected: `"status":"not_implemented"`, message references `decomposition`.

### Query — hyde

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hyde"}'
```

Expected: `"status":"not_implemented"`, message references `hyde`.

### Query — empty question (failure)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "semantic"}'
```

Expected: 422 validation error (question below min_length 1).

### Chat completions — JSON (non-streaming)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "What is theft?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content contains `not implemented yet`.

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "What is theft?"}], "stream": true}'
```

Expected: SSE `data:` frames with `delta` role then content, ending with `data: [DONE]`.

### Chat completions — invalid model (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: 400 with `"type":"invalid_request_error"`, `"code":"model_not_found"`.

### Chat completions — no user message (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "system", "content": "You are helpful."}]}'
```

Expected: 400 with `"code":"missing_user_message"`.

## Story 2.1 — Document Ingestion

What it adds: PDF extraction script that produces inspectable JSONL section-level corpora from the BNS and IPC bare act PDFs.

Prerequisite: Story 1.1 complete, `data/raw/` PDFs and `PROVENANCE.md` present, `uv sync` done.

### Run extraction

```bash
uv run python scripts/extract_sections.py
```

Expected: prints parser name/version, record counts (~358 BNS, ~500 IPC), count of `needs_review` flags, count of empty-text records. Both `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl` exist.

### Re-run (skip)

```bash
uv run python scripts/extract_sections.py
```

Expected: prints "BNS corpus up to date — skipping" and "IPC corpus up to date — skipping". No records appended or overwritten.
## Story 2.3 — Semantic Retrieval

What it adds: `/v1/query` with `pattern: "semantic"` returns ranked BNS/IPC passages via Voyage embedding + MongoDB `$vectorSearch`.

Prerequisite: Stories 2.1–2.2 data loaded, `.env` has `MONGODB_URI` and `VOYAGE_API_KEY`, API running.

### Natural-language question

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}' \
  | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

Expected: `"status":"ok"`, 1–3 results with non-increasing `score`, populated `trace`.

### No results (filters match nothing)

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "theft", "pattern": "semantic", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}' \
  | jq '{status, n: (.results | length)}'
```

Expected: HTTP 200, `"status":"no_results"`, `n` 0.

### Invalid filter (failure)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "theft", "pattern": "semantic", "filters": {"act": {"$ne": "x"}}}'
```

Expected: `422`.

### Limit out of range (failure)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "theft", "pattern": "semantic", "limit": 21}'
```

Expected: `422`.

### Missing credentials (failure)

Start the API with `VOYAGE_API_KEY=` empty, repeat the natural-language question.

Expected: HTTP 503, code `retrieval_not_ready`; not `no_results`.

### Other modes unchanged

Run the hybrid-reranked query from Story 1.1. Expected: still `not_implemented`. Chat with `rag-hybrid-reranked` still returns its placeholder (`rag-semantic` is real since Story 3.2).


## Story 3.1 — Grounded Answer Generation

What it adds: `/v1/query` with `pattern: "semantic"` and `generate_answer: true` returns a grounded, non-streaming answer in `generation` with resolved citations, or an honest `insufficient_evidence`, `unavailable`, or `malformed` outcome.

Prerequisite: Story 2.3 prerequisites, plus `.env` has `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`; API running.

### Answerable question

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}' \
  | jq '{status, g: (.generation | {outcome, model, provider, context_outcome, text: (.text[:300]), claims: [.claims[] | {t: .text[:80], e: .evidence_labels}], citations: [.citations[] | {label, chunk_id, section_id, act, heading}], trace}), ctx: [.results[] | {chunk_id, section_id, act, score}]}'
```

Expected: `"status":"ok"`, `outcome` `answered`, non-empty `text`, claims with labels, `citations` whose `chunk_id`s appear in `ctx`; `section_id` prefix matches `act`.

### Unsupported question

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the GST rate on restaurant services?", "pattern": "semantic", "limit": 5, "generate_answer": true}' \
  | jq '{status, outcome: .generation.outcome, text: .generation.text, claims: (.generation.claims | length), citations: (.generation.citations | length), n: (.results | length)}'
```

Expected: `outcome` `insufficient_evidence`, empty `text`, 0 claims and citations, `n` still > 0.

### Generation unavailable (failure)

Start the API with `GENERATION_API_KEY=` empty, repeat the unsupported-question command.

Expected: HTTP 200, `outcome` `unavailable`, empty `text`, `n` > 0.

### Generation off by default

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic", "limit": 3}' \
  | jq '{status, generation}'
```

Expected: `"status":"ok"`, `generation` null.

## Story 3.2 — Streamed Answers with Confidence

What it adds: `/v1/chat/completions` with `rag-semantic` runs the same retrieval and grounded generation as `/v1/query`, streaming a `DRAFT` answer followed by a confidence and sources footer.

Prerequisite: Story 3.1 prerequisites; API running. If `CAPSTONE_API_KEY` is set in `.env`, add `-H "Authorization: Bearer <key>"` to the chat commands.

```bash
# 1. Streamed answer
curl -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"rag-semantic","stream":true,"messages":[{"role":"user","content":"What is the punishment for theft under the BNS?"}]}' \
  | grep '^data: ' | head -c 1500

# 2. Same question via /v1/query (compare outcome and citations)
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "generate_answer": true}' \
  | jq '{status, g: (.generation | {outcome, confidence, attempts, issues, citations: [.citations[] | {label, section_id}]})}'

# 3. Unsupported question (non-streaming)
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"rag-semantic","messages":[{"role":"user","content":"What is the GST rate on restaurant services?"}]}' \
  | jq -r '.choices[0].message.content'

# 4. Wrong key (failure; API started with CAPSTONE_API_KEY set)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer wrong" \
  -d '{"model":"rag-semantic","messages":[{"role":"user","content":"theft"}]}'
```

Expected:
1. `data:` frames for `rag-semantic`: `DRAFT — checking evidence`, answer text with `[E1]` labels, then `Evidence check passed — confidence: high` and `Sources:`; ends with `data: [DONE]`.
2. `status` `ok`; `outcome` `answered`, `confidence` `high`, same citations as chat.
3. One insufficient-evidence sentence; no confidence line.
4. `401`.

## Story 4.1 — Hybrid Search

What it adds: `pattern: "hybrid"` (`rag-hybrid`) fuses Atlas Search keyword results on `chunks.text` with semantic results by reciprocal rank fusion.

Prerequisite: Stories 2.1–3.2 data and `.env`; create the keyword index once, then start the API.

```bash
# 1. Create or reuse the keyword index (name and status only)
uv run python -m building_with_rag.ingestion.keyword_index

# 2. Hybrid query (shows which route found each passage)
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"criminal breach of trust","pattern":"hybrid","limit":5}' \
  | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'

# 3. Hybrid chat (streamed)
curl -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"rag-hybrid","stream":true,"messages":[{"role":"user","content":"criminal breach of trust"}]}' | head -c 1500

# 4. Invalid scope (failure)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"theft","pattern":"hybrid","chapter":"XVII"}'
```

Expected:
1. `chunk_text_index: READY`; a second run reuses it.
2. `"status":"ok"`, at most 5 results with non-increasing `score` (equal to the fused score), `fr` equals position, each result has `sr` or `kr`; `trace` shows both routes, fusion, contribution.
3. `DRAFT`, answer, confidence/sources footer or an insufficient-evidence sentence; ends with `data: [DONE]`.
4. `422`.
