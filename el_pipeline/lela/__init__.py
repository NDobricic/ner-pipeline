"""
LELA integration module for el-pipeline.

Provides LELA-style components:
- GLiNER zero-shot NER (via gliner_spacy)
- BM25 candidate generation (via bm25s)
- Dense candidate generation (via FAISS + OpenAI-compatible embeddings)
- Embedding-based reranking
- vLLM-based disambiguation
- JSONL knowledge base loader
"""

from .config import (
    NER_LABELS,
    DEFAULT_GLINER_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_RERANKER_MODEL,
    CANDIDATES_TOP_K,
    RERANKER_TOP_K,
    SPAN_OPEN,
    SPAN_CLOSE,
    NOT_AN_ENTITY,
    RETRIEVER_TASK,
    RERANKER_TASK,
)

from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    create_disambiguation_messages,
    mark_mention_in_text,
)

from .llm_pool import (
    get_sentence_transformer_instance,
    clear_sentence_transformer_instances,
    get_vllm_instance,
    clear_vllm_instances,
)

__all__ = [
    # Config
    "NER_LABELS",
    "DEFAULT_GLINER_MODEL",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBEDDER_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "CANDIDATES_TOP_K",
    "RERANKER_TOP_K",
    "SPAN_OPEN",
    "SPAN_CLOSE",
    "NOT_AN_ENTITY",
    "RETRIEVER_TASK",
    "RERANKER_TASK",
    # Prompts
    "DEFAULT_SYSTEM_PROMPT",
    "create_disambiguation_messages",
    "mark_mention_in_text",
    # LLM Pool
    "get_sentence_transformer_instance",
    "clear_sentence_transformer_instances",
    "get_vllm_instance",
    "clear_vllm_instances",
]
