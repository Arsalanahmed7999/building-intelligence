"""Create (or reuse) the Atlas Search keyword index on chunks.text.

Run: uv run python -m building_with_rag.ingestion.keyword_index
Idempotent: absent -> create; matching -> reuse; different -> report and stop (never auto-replace).
No Voyage calls, no data writes.
"""

import time

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

WAIT_SECONDS = 120
POLL_SECONDS = 5


def _find(coll):
    return next((i for i in coll.list_search_indexes() if i.get("name") == schema.KEYWORD_INDEX_NAME), None)


def definition_diffs(expected, actual, path: str = "") -> list[str]:
    """Differences where the existing definition lacks or changes an expected value."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path or 'definition'}: expected an object"]
        diffs: list[str] = []
        for key, value in expected.items():
            if key not in actual:
                diffs.append(f"{path}{key}: missing")
            else:
                diffs += definition_diffs(value, actual[key], f"{path}{key}.")
        return diffs
    return [] if expected == actual else [f"{path.rstrip('.')}: {actual!r} (expected {expected!r})"]


def ensure_keyword_index(coll) -> str:
    idx = _find(coll)
    if idx is not None:
        existing = idx.get("latestDefinition") or idx.get("definition") or {}
        diffs = definition_diffs(schema.KEYWORD_INDEX_DEFINITION, existing)
        if diffs:
            raise SystemExit(
                f"Search index '{schema.KEYWORD_INDEX_NAME}' differs: {'; '.join(diffs)}. "
                "Drop it manually in the Atlas UI, then re-run."
            )
    else:
        coll.create_search_index(
            SearchIndexModel(
                name=schema.KEYWORD_INDEX_NAME,
                type="search",
                definition=schema.KEYWORD_INDEX_DEFINITION,
            )
        )
    status = None
    for waited in range(0, WAIT_SECONDS + POLL_SECONDS, POLL_SECONDS):
        idx = _find(coll)
        status = idx.get("status") if idx else "UNKNOWN"
        if idx and (idx.get("queryable") or status == "READY"):
            return "READY"
        if waited < WAIT_SECONDS:
            time.sleep(POLL_SECONDS)
    return f"TIMEOUT (last status: {status})"


def main() -> None:
    settings = get_settings()
    if not settings.mongodb_uri:
        raise SystemExit("MONGODB_URI is not set.")
    name = settings.mongodb_test_db_name if settings.app_env == "testing" else settings.mongodb_db_name
    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    status = ensure_keyword_index(client[name][schema.CHUNKS_COLLECTION])
    print(f"{schema.KEYWORD_INDEX_NAME}: {status}")


if __name__ == "__main__":
    main()
