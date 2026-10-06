from typing import Literal

Pattern = Literal["semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde"]

MODE_REGISTRY: dict[str, str] = {
    "semantic": "rag-semantic",
    "hybrid": "rag-hybrid",
    "hybrid-reranked": "rag-hybrid-reranked",
    "structured": "rag-structured",
    "decomposition": "rag-decomposition",
    "hyde": "rag-hyde",
}

MODEL_TO_MODE: dict[str, str] = {model: mode for mode, model in MODE_REGISTRY.items()}
