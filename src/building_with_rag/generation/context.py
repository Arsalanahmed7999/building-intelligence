"""Bounded, labelled evidence context built from retrieval results."""

from dataclasses import dataclass

from building_with_rag.contracts import RetrievedChunk

MAX_PASSAGES = 5
MAX_CONTEXT_CHARS = 12_000


@dataclass
class ContextEntry:
    label: str
    chunk: RetrievedChunk

    def as_dict(self) -> dict:
        c = self.chunk
        return {
            "label": self.label,
            "chunk_id": c.chunk_id,
            "section_id": c.section_id,
            "act": c.act,
            "act_label": c.act_label,
            "heading": c.heading,
            "chapter": c.chapter,
            "section_number": c.section_number,
            "status": c.status,
            "source_pdf": c.source_pdf,
            "needs_review": c.needs_review,
            "text": c.text,
        }


@dataclass
class Context:
    entries: list[ContextEntry]
    omitted: int
    chars: int

    @property
    def outcome(self) -> str:
        return "assembled" if self.entries else "empty"

    def by_label(self) -> dict[str, ContextEntry]:
        return {e.label: e for e in self.entries}


def assemble_context(results: list[RetrievedChunk]) -> Context:
    """Take passages in score order; never cut one mid-text; stop at the first budget hit."""
    entries: list[ContextEntry] = []
    chars = 0
    for chunk in results:
        if len(entries) >= MAX_PASSAGES or chars + len(chunk.text) > MAX_CONTEXT_CHARS:
            break
        entries.append(ContextEntry(label=f"E{len(entries) + 1}", chunk=chunk))
        chars += len(chunk.text)
    return Context(entries=entries, omitted=len(results) - len(entries), chars=chars)
