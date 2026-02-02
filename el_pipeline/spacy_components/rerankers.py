"""
spaCy reranker components for the EL pipeline.

Provides factories and components for candidate reranking:
- LELAEmbedderRerankerComponent: Embedding-based cosine similarity reranking
- CrossEncoderRerankerComponent: Cross-encoder reranking
- NoOpRerankerComponent: Pass-through (no reranking)
"""

import logging
from typing import List, Optional

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Span

from el_pipeline.lela.config import (
    RERANKER_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    RERANKER_TASK,
    SPAN_OPEN,
    SPAN_CLOSE,
)
from el_pipeline.lela.llm_pool import (
    get_sentence_transformer_instance,
    release_sentence_transformer,
)
from el_pipeline.utils import ensure_candidates_extension
from el_pipeline.types import Candidate, ProgressCallback

logger = logging.getLogger(__name__)


# ============================================================================
# LELA Embedder Reranker Component
# ============================================================================

# @Language.factory(
#     "el_pipeline_lela_embedder_reranker",
#     default_config={
#         "model_name": DEFAULT_EMBEDDER_MODEL,
#         "top_k": RERANKER_TOP_K,
#         "device": None,
#     },
# )
# def create_lela_embedder_reranker_component(
#     nlp: Language,
#     name: str,
#     model_name: str,
#     top_k: int,
#     device: Optional[str],
# ):
#     """Factory for LELA embedder reranker component."""
#     return LELAEmbedderRerankerComponent(
#         nlp=nlp,
#         model_name=model_name,
#         top_k=top_k,
#         device=device,
#     )
#
#
# class LELAEmbedderRerankerComponent:
#     """
#     Embedding-based reranker component for spaCy.
#
#     Uses SentenceTransformers to rerank candidates by cosine similarity.
#     The mention is marked in the document text with brackets for context.
#
#     Memory management: Model is loaded on-demand and released after use,
#     allowing it to be evicted if memory is needed for later stages.
#     """
#
#     def __init__(
#         self,
#         nlp: Language,
#         model_name: str = DEFAULT_EMBEDDER_MODEL,
#         top_k: int = RERANKER_TOP_K,
#         device: Optional[str] = None,
#     ):
#         self.nlp = nlp
#         self.model_name = model_name
#         self.top_k = top_k
#         self.device = device
#
#         ensure_candidates_extension()
#
#         # Model loaded on-demand in __call__, not here
#         # This allows lazy eviction when memory is needed
#
#         # Optional progress callback for fine-grained progress reporting
#         self.progress_callback: Optional[ProgressCallback] = None
#
#         logger.info(f"LELA embedder reranker initialized: {model_name}")
#
#     def _embed_texts(self, texts: List[str], model) -> np.ndarray:
#         """Embed texts using the SentenceTransformer model."""
#         return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
#
#     def _format_query(self, text: str, start: int, end: int) -> str:
#         """Format query with marked mention in text."""
#         marked_text = f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"
#         return f"Instruct: {RERANKER_TASK}\nQuery: {marked_text}"
#
#     def _format_candidate(self, candidate: Candidate, kb) -> str:
#         """Format candidate for embedding."""
#         # Get entity title from KB for display
#         entity = kb.get_entity(candidate.entity_id) if kb else None
#         title = entity.title if entity else candidate.entity_id
#         if candidate.description:
#             return f"{title}: {candidate.description}"
#         return title
#
#     def __call__(self, doc: Doc) -> Doc:
#         """Rerank candidates for all entities in the document."""
#         text = doc.text
#         entities = list(doc.ents)
#         num_entities = len(entities)
#
#         # Check if any entity needs reranking
#         needs_reranking = any(
#             len(getattr(ent._, "candidates", [])) > self.top_k
#             for ent in entities
#         )
#
#         if not needs_reranking:
#             return doc
#
#         # Report model loading
#         if self.progress_callback:
#             self.progress_callback(0.0, f"Loading reranker model ({self.model_name.split('/')[-1]})...")
#
#         # Load model for this stage (will reuse cached if available)
#         model = get_sentence_transformer_instance(self.model_name, self.device)
#
#         if self.progress_callback:
#             self.progress_callback(0.1, "Model loaded, reranking candidates...")
#
#         # Progress: 0.0-0.1 = model loading, 0.1-1.0 = processing entities
#         processing_start = 0.1
#         processing_range = 0.9
#
#         try:
#             for i, ent in enumerate(entities):
#                 # Report progress if callback is set
#                 if self.progress_callback and num_entities > 0:
#                     progress = processing_start + (i / num_entities) * processing_range
#                     ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
#                     self.progress_callback(progress, f"Reranking {i+1}/{num_entities}: {ent_text}")
#
#                 candidates = getattr(ent._, "candidates", [])
#                 if not candidates:
#                     continue
#
#                 if len(candidates) <= self.top_k:
#                     continue
#
#                 # Format query and candidates
#                 query_text = self._format_query(text, ent.start_char, ent.end_char)
#                 candidate_texts = [
#                     self._format_candidate(c, self.kb) if hasattr(self, 'kb') else
#                     f"{c.entity_id}: {c.description}" if c.description else c.entity_id
#                     for c in candidates
#                 ]
#
#                 # Embed all texts (already normalized by encode())
#                 all_texts = [query_text] + candidate_texts
#                 embeddings = self._embed_texts(all_texts, model)
#
#                 # Compute cosine similarities
#                 query_embedding = embeddings[0:1]
#                 candidate_embeddings = embeddings[1:]
#                 similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
#
#                 # Sort by similarity and take top_k
#                 scored_candidates = list(zip(candidates, similarities))
#                 scored_candidates.sort(key=lambda x: x[1], reverse=True)
#                 top_candidates = scored_candidates[:self.top_k]
#
#                 # Update candidates with new scores
#                 reranked = []
#                 reranked_scores = []
#                 for candidate, score in top_candidates:
#                     reranked.append(Candidate(
#                         entity_id=candidate.entity_id,
#                         score=float(score),
#                         description=candidate.description,
#                     ))
#                     reranked_scores.append(float(score))
#
#                 ent._.candidates = reranked
#                 ent._.candidate_scores = reranked_scores
#
#                 logger.debug(
#                     f"Reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
#                 )
#         finally:
#             # Release model - stays cached but can be evicted if memory needed
#             release_sentence_transformer(self.model_name, self.device)
#
#         # Clear progress callback after processing
#         self.progress_callback = None
#
#         return doc


# ============================================================================
# Cross-Encoder Reranker Component
# ============================================================================


@Language.factory(
    "el_pipeline_cross_encoder_reranker",
    default_config={
        "model_name": "Qwen/Qwen3-Reranker-4B-seq-cls",
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
        model_name: str = "Qwen/Qwen3-Reranker-4B-seq-cls",
        top_k: int = 10,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k

        ensure_candidates_extension()

        # Optional progress callback for fine-grained progress reporting
        self.progress_callback: Optional[ProgressCallback] = None

        # Lazy import
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )

        logger.info(f"Cross-encoder reranker initialized: {model_name}")

    def _format_query(self, text: str, start: int, end: int) -> str:
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        marked_text = (
            f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"
        )
        return f"{prefix}<Instruct>: {RERANKER_TASK}\n<Query>: {marked_text}\n"

    def _format_candidate(self, candidate: Candidate) -> str:
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        document = f"{candidate.entity_id} ({candidate.description if candidate.description else ''})"
        return f"<Document>: {document}{suffix}"

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        text = doc.text
        entities = list(doc.ents)
        num_entities = len(entities)

        for i, ent in enumerate(entities):
            # Report progress if callback is set
            if self.progress_callback and num_entities > 0:
                progress = i / num_entities
                ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                self.progress_callback(
                    progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                )

            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            # Build pairs for cross-encoder
            pairs = [
                (
                    self._format_query(text, ent.start_char, ent.end_char),
                    self._format_candidate(c),
                )
                for c in candidates
            ]

            # Score pairs
            scores = self.model.predict(pairs)

            # Sort and take top_k
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored_candidates[: self.top_k]

            # Update candidates with new scores
            reranked = []
            reranked_scores = []
            for candidate, score in top_candidates:
                reranked.append(
                    Candidate(
                        entity_id=candidate.entity_id,
                        score=float(score),
                        description=candidate.description,
                    )
                )
                reranked_scores.append(float(score))

            ent._.candidates = reranked
            ent._.candidate_scores = reranked_scores

            logger.debug(
                f"Cross-encoder reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
            )

        # Clear progress callback after processing
        self.progress_callback = None

        return doc


# ============================================================================
# No-Op Reranker Component
# ============================================================================


@Language.factory(
    "el_pipeline_noop_reranker",
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
        ensure_candidates_extension()

    def __call__(self, doc: Doc) -> Doc:
        """Pass through - no reranking."""
        return doc
