# Building with RAG

FastAPI capstone that grows into a RAG system over the BNS and IPC. See `docs/architecture.md`.

- Setup: `uv sync`; copy `.env.example` to `.env` (untracked, secrets never committed).
- Run: `uv run uvicorn building_with_rag.main:app --port 8000`
- Test: `uv run pytest`; lint: `uv run ruff check .`
- Open WebUI is a separate trainer-supplied client; use its bundle scripts only.
