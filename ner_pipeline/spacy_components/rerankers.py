"""
spaCy reranker components for the NER pipeline.

Provides factories and components for candidate reranking:
- LELAEmbedderRerankerComponent: Embedding-based cosine similarity reranking
- CrossEncoderRerankerComponent: Cross-encoder reranking
- NoOpRerankerComponent: Pass-through (no reranking)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Span

from ner_pipeline.lela.config import (
    RERANKER_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    RERANKER_TASK,
    SPAN_OPEN,
    SPAN_CLOSE,
)
from ner_pipeline.lela.llm_pool import embedder_pool

logger = logging.getLogger(__name__)


def _ensure_candidates_extension():
    """Ensure the candidates extension is registered on Span."""
    if not Span.has_extension("candidates"):
        Span.set_extension("candidates", default=[])


# ============================================================================
# LELA Embedder Reranker Component
# ============================================================================

@Language.factory(
    "ner_pipeline_lela_embedder_reranker",
    default_config={
        "model_name": DEFAULT_EMBEDDER_MODEL,
        "top_k": RERANKER_TOP_K,
        "base_url": "http://localhost",
        "port": 8000,
    },
)
def create_lela_embedder_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    base_url: str,
    port: int,
):
    """Factory for LELA embedder reranker component."""
    return LELAEmbedderRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        base_url=base_url,
        port=port,
    )


class LELAEmbedderRerankerComponent:
    """
    Embedding-based reranker component for spaCy.

    Uses OpenAI-compatible embeddings to rerank candidates by cosine similarity.
    The mention is marked in the document text with brackets for context.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = RERANKER_TOP_K,
        base_url: str = "http://localhost",
        port: int = 8000,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.base_url = base_url
        self.port = port

        _ensure_candidates_extension()

        logger.info(f"LELA embedder reranker initialized: {model_name}")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the embedding service."""
        return embedder_pool.embed(
            texts,
            model_name=self.model_name,
            base_url=self.base_url,
            port=self.port,
        )

    def _format_query(self, text: str, start: int, end: int) -> str:
        """Format query with marked mention in text."""
        marked_text = f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"
        return f"Instruct: {RERANKER_TASK}\nQuery: {marked_text}"

    def _format_candidate(self, title: str, description: str) -> str:
        """Format candidate for embedding."""
        if description:
            return f"{title}: {description}"
        return title

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        text = doc.text

        for ent in doc.ents:
            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            if len(candidates) <= self.top_k:
                continue

            # Format query and candidates
            query_text = self._format_query(text, ent.start_char, ent.end_char)
            candidate_texts = [
                self._format_candidate(title, desc)
                for title, desc in candidates
            ]

            # Embed all texts
            all_texts = [query_text] + candidate_texts
            embeddings = self._embed_texts(all_texts)
            embeddings = np.array(embeddings, dtype=np.float32)

            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)

            # Compute cosine similarities
            query_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

            # Sort by similarity and take top_k
            scored_candidates = list(zip(candidates, similarities))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored_candidates[:self.top_k]

            # Update candidates (keep as LELA format)
            ent._.candidates = [c for c, _ in top_candidates]

            logger.debug(
                f"Reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
            )

        return doc


# ============================================================================
# Cross-Encoder Reranker Component
# ============================================================================

@Language.factory(
    "ner_pipeline_cross_encoder_reranker",
    default_config={
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 10,
    },
)
def create_cross_encoder_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
):
    """Factory for cross-encoder reranker component."""
    return CrossEncoderRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
    )


class CrossEncoderRerankerComponent:
    """
    Cross-encoder reranker component for spaCy.

    Uses sentence-transformers CrossEncoder for pairwise scoring.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k

        _ensure_candidates_extension()

        # Lazy import
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )

        logger.info(f"Cross-encoder reranker initialized: {model_name}")

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        text = doc.text

        for ent in doc.ents:
            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            # Build pairs for cross-encoder
            pairs = [
                (f"{ent.text} | {text}", desc if desc else title)
                for title, desc in candidates
            ]

            # Score pairs
            scores = self.model.predict(pairs)

            # Sort and take top_k
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored_candidates[:self.top_k]

            # Update candidates
            ent._.candidates = [c for c, _ in top_candidates]

            logger.debug(
                f"Cross-encoder reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
            )

        return doc


# ============================================================================
# No-Op Reranker Component
# ============================================================================

@Language.factory(
    "ner_pipeline_noop_reranker",
    default_config={},
)
def create_noop_reranker_component(
    nlp: Language,
    name: str,
):
    """Factory for no-op reranker component."""
    return NoOpRerankerComponent(nlp=nlp)


class NoOpRerankerComponent:
    """
    No-op reranker component for spaCy.

    Passes candidates through unchanged. Use when no reranking is needed.
    """

    def __init__(self, nlp: Language):
        self.nlp = nlp
        _ensure_candidates_extension()

    def __call__(self, doc: Doc) -> Doc:
        """Pass through - no reranking."""
        return doc
