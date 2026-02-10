"""
spaCy reranker components for the EL pipeline.

Provides factories and components for candidate reranking:
- CrossEncoderRerankerComponent: Cross-encoder reranking (sentence-transformers)
- VLLMAPIClientReranker: Cross-encoder via external vLLM API
- LlamaServerReranker: Reranking via llama.cpp server
- LELAEmbedderRerankerComponent: Embedding-based cosine similarity (SentenceTransformer)
- LELAEmbedderVLLMRerankerComponent: Embedding-based cosine similarity (vLLM)
- LELACrossEncoderVLLMRerankerComponent: Cross-encoder via vLLM score() API (seq-cls)
- NoOpRerankerComponent: Pass-through (no reranking)
"""

import logging
import requests
from typing import List, Optional

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Span

from el_pipeline.lela.config import (
    RERANKER_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_VLLM_RERANKER_MODEL,
    RERANKER_TASK,
    SPAN_OPEN,
    SPAN_CLOSE,
    CROSS_ENCODER_PREFIX,
    CROSS_ENCODER_SUFFIX,
)
from el_pipeline.lela.llm_pool import (
    get_sentence_transformer_instance,
    release_sentence_transformer,
    get_vllm_instance,
    release_vllm,
)
from el_pipeline.utils import ensure_candidates_extension
from el_pipeline.types import Candidate, ProgressCallback

logger = logging.getLogger(__name__)

# Lazy imports for vLLM
_vllm = None


def _get_vllm():
    """Lazy import of vllm."""
    global _vllm
    if _vllm is None:
        try:
            import vllm

            _vllm = vllm
        except ImportError:
            raise ImportError(
                "vllm package required for vLLM reranker. "
                "Install with: pip install vllm"
            )
    return _vllm


# ============================================================================
# Cross-Encoder Reranker Component
# ============================================================================


@Language.factory(
    "el_pipeline_lela_cross_encoder_reranker",
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
# vLLM API Client Reranker Component
# ============================================================================


@Language.factory(
    "el_pipeline_lela_vllm_api_client_reranker",
    default_config={
        "top_k": 10,
        "base_url": "http://localhost",
        "port": 8000,
    },
)
def create_lela_vllm_api_client_reranker_component(
    nlp: Language,
    name: str,
    top_k: int,
    base_url: str,
    port: int,
):
    """Factory for vLLM API client reranker component."""
    return VLLMAPIClientReranker(
        nlp=nlp,
        top_k=top_k,
        base_url=base_url,
        port=port,
    )


class VLLMAPIClientReranker:
    PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    DOCUMENT_TEMPLATE = "<Document>: {doc}{suffix}"

    def __init__(
        self,
        nlp: Language,
        top_k: int = 10,
        base_url: str = "http://localhost",
        port: int = 8000,
    ):
        self.nlp = nlp
        self.top_k = top_k
        self.api_url = f"{base_url}:{port}/score"
        ensure_candidates_extension()
        logger.info(f"Using vLLM API reranker at {self.api_url}")
        self.progress_callback: Optional[ProgressCallback] = None

    @staticmethod
    def post_http_request(prompt: dict, api_url: str) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers=headers, json=prompt)
        response.raise_for_status()
        return response

    def __call__(self, doc: Doc) -> Doc:
        entities = list(doc.ents)
        num_entities = len(entities)

        for i, ent in enumerate(entities):
            if self.progress_callback and num_entities > 0:
                progress = i / num_entities
                ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                self.progress_callback(
                    progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                )

            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            query = f"{doc.text[: ent.start_char]}{SPAN_OPEN}{ent.text}{SPAN_CLOSE}{doc.text[ent.end_char:]}"
            query = self.QUERY_TEMPLATE.format(
                prefix=self.PREFIX, instruction=RERANKER_TASK, query=query
            )

            documents = [f"{c.entity_id} ({c.description or ''})" for c in candidates]
            documents = [
                self.DOCUMENT_TEMPLATE.format(doc=d, suffix=self.SUFFIX)
                for d in documents
            ]

            try:
                response = self.post_http_request(
                    prompt={
                        "text_1": query,
                        "text_2": documents,
                    },
                    api_url=self.api_url,
                ).json()

                if "data" not in response:
                    logger.error(
                        f"Reranker API response does not contain 'data' field: {response} for query: {query}"
                    )
                    # Keep original candidates if API fails
                    ent._.candidates = candidates[: self.top_k]
                    ent._.candidate_scores = [c.score for c in candidates[: self.top_k]]
                    continue

                scores = [d["score"] for d in response["data"]]
                scored_candidates = list(zip(candidates, scores))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_candidates[: self.top_k]

                reranked_candidates = [c for c, s in top_candidates]
                reranked_scores = [s for c, s in top_candidates]

                ent._.candidates = reranked_candidates
                ent._.candidate_scores = reranked_scores

            except (requests.exceptions.RequestException, ValueError) as e:
                logger.error(
                    f"Reranker API request failed for entity '{ent.text}': {e}"
                )
                # Keep original candidates on failure
                ent._.candidates = candidates[: self.top_k]
                ent._.candidate_scores = [c.score for c in candidates[: self.top_k]]
        self.progress_callback = None
        return doc


# ============================================================================
# No-Op Reranker Component
# ============================================================================


@Language.factory(
    "el_pipeline_noop_reranker",
    default_config={"top_k": 10},
)
def create_noop_reranker_component(
    nlp: Language,
    name: str,
    top_k: int = 10,
):
    """Factory for no-op reranker component."""
    return NoOpRerankerComponent(nlp=nlp, top_k=top_k)


class NoOpRerankerComponent:
    """
    No-op reranker component for spaCy.

    Truncates candidates to top_k. Use when no reranking is needed.
    """

    def __init__(self, nlp: Language, top_k: int = 10):
        self.nlp = nlp
        self.top_k = top_k
        ensure_candidates_extension()

    def __call__(self, doc: Doc) -> Doc:
        """Truncate candidates to top_k without reranking."""
        for ent in doc.ents:
            candidates = getattr(ent._, "candidates", [])
            if candidates and len(candidates) > self.top_k:
                ent._.candidates = candidates[: self.top_k]
        return doc


# ============================================================================
# Llama Server Reranker Component
# ============================================================================


@Language.factory(
    "el_pipeline_lela_llama_server_reranker",
    default_config={
        "model_name": "qwen3-reranker",
        "top_k": 10,
        "base_url": "http://localhost",
        "port": 8000,
    },
)
def create_lela_llama_server_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    base_url: str,
    port: int,
):
    """Factory for Llama Server reranker component."""
    return LlamaServerReranker(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        base_url=base_url,
        port=port,
    )


class LlamaServerReranker:
    """
    Reranker component that uses a llama.cpp server compatible with the
    OpenAI-style rerank endpoint.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str,
        top_k: int = 10,
        base_url: str = "http://localhost",
        port: int = 8000,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.api_url = f"{base_url}:{port}/v1/rerank"
        ensure_candidates_extension()
        logger.info(
            f"Using Llama Server reranker for model '{self.model_name}' at {self.api_url}"
        )
        self.progress_callback: Optional[ProgressCallback] = None

    @staticmethod
    def post_http_request(payload: dict, api_url: str) -> requests.Response:
        headers = {"User-Agent": "LELA Client", "Content-Type": "application/json"}
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response

    def __call__(self, doc: Doc) -> Doc:
        entities = list(doc.ents)
        num_entities = len(entities)

        for i, ent in enumerate(entities):
            if self.progress_callback and num_entities > 0:
                progress = i / num_entities
                ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                self.progress_callback(
                    progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                )

            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            query_text = ent.text
            document_texts = [
                f"{c.entity_id} ({c.description or ''})" for c in candidates
            ]

            try:
                response = self.post_http_request(
                    payload={
                        "model": self.model_name,
                        "query": query_text,
                        "documents": document_texts,
                        "top_n": self.top_k,
                    },
                    api_url=self.api_url,
                ).json()

                if "results" not in response:
                    logger.error(
                        f"Reranker API response does not contain 'results' field: {response} for query: {query_text}"
                    )
                    # Keep original candidates if API fails
                    ent._.candidates = candidates[: self.top_k]
                    ent._.candidate_scores = [c.score for c in candidates[: self.top_k]]
                    continue

                results = response["results"]

                reranked_candidates = []
                reranked_scores = []

                for result in results:
                    original_index = result.get("index")
                    score = result.get("relevance_score")

                    if original_index is None or score is None:
                        continue

                    if 0 <= original_index < len(candidates):
                        candidate = candidates[original_index]
                        reranked_candidates.append(
                            Candidate(
                                entity_id=candidate.entity_id,
                                score=float(score),
                                description=candidate.description,
                            )
                        )
                        reranked_scores.append(float(score))

                ent._.candidates = reranked_candidates
                ent._.candidate_scores = reranked_scores

            except (requests.exceptions.RequestException, ValueError) as e:
                logger.error(
                    f"Llama Server Reranker API request failed for entity '{ent.text}': {e}"
                )
                # Keep original candidates on failure
                ent._.candidates = candidates[: self.top_k]
                ent._.candidate_scores = [c.score for c in candidates[: self.top_k]]
        self.progress_callback = None
        return doc


# ============================================================================
# LELA Embedder Reranker Component (SentenceTransformer)
# ============================================================================


@Language.factory(
    "el_pipeline_lela_embedder_transformers_reranker",
    default_config={
        "model_name": DEFAULT_EMBEDDER_MODEL,
        "top_k": RERANKER_TOP_K,
        "device": None,
    },
)
def create_lela_embedder_transformers_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    device: Optional[str],
):
    """Factory for LELA embedder reranker component."""
    return LELAEmbedderRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        device=device,
    )


class LELAEmbedderRerankerComponent:
    """
    Embedding-based reranker component for spaCy.

    Uses SentenceTransformers to rerank candidates by cosine similarity.
    The mention is marked in the document text with brackets for context.

    Memory management: Model is loaded on-demand and released after use,
    allowing it to be evicted if memory is needed for later stages.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = RERANKER_TOP_K,
        device: Optional[str] = None,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.device = device

        ensure_candidates_extension()

        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"LELA embedder reranker initialized: {model_name}")

    def _embed_texts(self, texts: List[str], model) -> np.ndarray:
        """Embed texts using the SentenceTransformer model."""
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def _format_query(self, text: str, start: int, end: int) -> str:
        """Format query with marked mention in text."""
        marked_text = (
            f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"
        )
        return f"Instruct: {RERANKER_TASK}\nQuery: {marked_text}"

    def _format_candidate(self, candidate: Candidate) -> str:
        """Format candidate for embedding."""
        if candidate.description:
            return f"{candidate.entity_id}: {candidate.description}"
        return candidate.entity_id

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        text = doc.text
        entities = list(doc.ents)
        num_entities = len(entities)

        needs_reranking = any(
            len(getattr(ent._, "candidates", [])) > self.top_k for ent in entities
        )

        if not needs_reranking:
            return doc

        if self.progress_callback:
            self.progress_callback(
                0.0, f"Loading reranker model ({self.model_name.split('/')[-1]})..."
            )

        model, was_cached = get_sentence_transformer_instance(
            self.model_name, self.device
        )

        if self.progress_callback:
            status = "Using cached model" if was_cached else "Model loaded"
            self.progress_callback(0.1, f"{status}, reranking candidates...")

        processing_start = 0.1
        processing_range = 0.9

        try:
            for i, ent in enumerate(entities):
                if self.progress_callback and num_entities > 0:
                    progress = processing_start + (i / num_entities) * processing_range
                    ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                    self.progress_callback(
                        progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                    )

                candidates = getattr(ent._, "candidates", [])
                if not candidates or len(candidates) <= self.top_k:
                    continue

                query_text = self._format_query(text, ent.start_char, ent.end_char)
                candidate_texts = [self._format_candidate(c) for c in candidates]

                all_texts = [query_text] + candidate_texts
                embeddings = self._embed_texts(all_texts, model)

                query_embedding = embeddings[0:1]
                candidate_embeddings = embeddings[1:]
                similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

                scored_candidates = list(zip(candidates, similarities))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_candidates[: self.top_k]

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
                    f"Reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
                )
        finally:
            release_sentence_transformer(self.model_name, self.device)

        self.progress_callback = None
        return doc


# ============================================================================
# LELA Cross-Encoder vLLM Reranker Component
# ============================================================================


@Language.factory(
    "el_pipeline_lela_cross_encoder_vllm_reranker",
    default_config={
        "model_name": DEFAULT_VLLM_RERANKER_MODEL,
        "top_k": RERANKER_TOP_K,
    },
)
def create_lela_cross_encoder_vllm_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
):
    """Factory for LELA cross-encoder vLLM reranker component."""
    return LELACrossEncoderVLLMRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
    )


class LELACrossEncoderVLLMRerankerComponent:
    """
    Cross-encoder reranker using vLLM's score() API with seq-cls models.

    Uses the Qwen3-Reranker-seq-cls model variant which has a classification
    head, enabling direct use of vLLM's score() API for relevance scoring.
    Model is loaded on-demand and released after use.
    """

    QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    DOCUMENT_TEMPLATE = "<Document>: {doc}{suffix}"

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_VLLM_RERANKER_MODEL,
        top_k: int = RERANKER_TOP_K,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k

        ensure_candidates_extension()

        self.model = None
        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"LELA cross-encoder vLLM reranker initialized: {model_name}")

    def _format_query(self, text: str, start: int, end: int) -> str:
        """Format query with marked mention in text."""
        marked_text = (
            f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"
        )
        return self.QUERY_TEMPLATE.format(
            prefix=CROSS_ENCODER_PREFIX,
            instruction=RERANKER_TASK,
            query=marked_text,
        )

    def _format_document(self, candidate: Candidate) -> str:
        """Format a candidate as a document string for scoring."""
        doc_text = f"{candidate.entity_id} ({candidate.description or ''})"
        return self.DOCUMENT_TEMPLATE.format(doc=doc_text, suffix=CROSS_ENCODER_SUFFIX)

    def _ensure_model_loaded(self, progress_callback=None):
        """Load model on-demand if not already loaded."""
        if self.model is None:
            _get_vllm()

            if progress_callback:
                progress_callback(
                    0.0, f"Loading reranker model ({self.model_name.split('/')[-1]})..."
                )

            self.model, was_cached = get_vllm_instance(
                model_name=self.model_name,
                hf_overrides={
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "classifier_from_token": ["no", "yes"],
                    "is_original_qwen3_reranker": True,
                },
            )

            if progress_callback:
                status = "Using cached model" if was_cached else "Model loaded"
                progress_callback(0.1, f"{status}, reranking candidates...")

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        text = doc.text
        entities = list(doc.ents)
        num_entities = len(entities)

        needs_reranking = any(
            len(getattr(ent._, "candidates", [])) > self.top_k for ent in entities
        )

        if not needs_reranking:
            return doc

        self._ensure_model_loaded(self.progress_callback)

        try:
            # Collect all (query, document) pairs across entities for batched scoring
            all_queries = []
            all_documents = []
            # Track which entities need reranking and how many pairs each has
            work_items = []  # (entity_index, candidates, num_pairs)

            for i, ent in enumerate(entities):
                candidates = getattr(ent._, "candidates", [])
                if not candidates or len(candidates) <= self.top_k:
                    continue

                query = self._format_query(text, ent.start_char, ent.end_char)
                documents = [self._format_document(c) for c in candidates]

                work_items.append((i, candidates, len(documents)))
                all_queries.extend([query] * len(documents))
                all_documents.extend(documents)

            if all_queries:
                if self.progress_callback:
                    self.progress_callback(
                        0.2,
                        f"Scoring {len(all_queries)} pairs across {len(work_items)} entities...",
                    )

                # Single batched score call using vLLM's N -> N pattern
                outputs = self.model.score(all_queries, all_documents)
                all_scores = [out.outputs.score for out in outputs]

                # Split scores back per entity and apply top_k selection
                offset = 0
                for ent_idx, candidates, num_pairs in work_items:
                    ent = entities[ent_idx]
                    scores = all_scores[offset : offset + num_pairs]
                    offset += num_pairs

                    scored_candidates = list(zip(candidates, scores))
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = scored_candidates[: self.top_k]

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
                        f"Cross-encoder (vLLM) reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
                    )
        finally:
            release_vllm(self.model_name)
            self.model = None  # Drop reference so pool eviction can free GPU memory

        self.progress_callback = None
        return doc


# ============================================================================
# LELA Embedder vLLM Reranker Component
# ============================================================================


@Language.factory(
    "el_pipeline_lela_embedder_vllm_reranker",
    default_config={
        "model_name": DEFAULT_EMBEDDER_MODEL,
        "top_k": RERANKER_TOP_K,
    },
)
def create_lela_embedder_vllm_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
):
    """Factory for LELA embedder vLLM reranker component."""
    return LELAEmbedderVLLMRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
    )


class LELAEmbedderVLLMRerankerComponent:
    """
    Embedding-based reranker using vLLM's .encode() API.

    Same functionality as LELAEmbedderRerankerComponent but using vLLM
    instead of SentenceTransformers for faster inference.
    Model is loaded on-demand and released after use.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = RERANKER_TOP_K,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k

        ensure_candidates_extension()

        self.model = None
        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"LELA embedder vLLM reranker initialized: {model_name}")

    def _format_query(self, text: str, start: int, end: int) -> str:
        """Format query with marked mention in text."""
        marked_text = (
            f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"
        )
        return f"Instruct: {RERANKER_TASK}\nQuery: {marked_text}"

    def _format_candidate(self, candidate: Candidate) -> str:
        """Format candidate for embedding."""
        if candidate.description:
            return f"{candidate.entity_id}: {candidate.description}"
        return candidate.entity_id

    def _ensure_model_loaded(self, progress_callback=None):
        """Load model on-demand if not already loaded."""
        if self.model is None:
            _get_vllm()

            if progress_callback:
                progress_callback(
                    0.0, f"Loading reranker model ({self.model_name.split('/')[-1]})..."
                )

            self.model, was_cached = get_vllm_instance(
                model_name=self.model_name,
                convert="embed",
            )

            if progress_callback:
                status = "Using cached model" if was_cached else "Model loaded"
                progress_callback(0.1, f"{status}, reranking candidates...")

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        text = doc.text
        entities = list(doc.ents)
        num_entities = len(entities)

        needs_reranking = any(
            len(getattr(ent._, "candidates", [])) > self.top_k for ent in entities
        )

        if not needs_reranking:
            return doc

        self._ensure_model_loaded(self.progress_callback)

        processing_start = 0.1
        processing_range = 0.9

        try:
            for i, ent in enumerate(entities):
                if self.progress_callback and num_entities > 0:
                    progress = processing_start + (i / num_entities) * processing_range
                    ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                    self.progress_callback(
                        progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                    )

                candidates = getattr(ent._, "candidates", [])
                if not candidates or len(candidates) <= self.top_k:
                    continue

                query_text = self._format_query(text, ent.start_char, ent.end_char)
                candidate_texts = [self._format_candidate(c) for c in candidates]

                all_texts = [query_text] + candidate_texts
                outputs = self.model.encode(all_texts)

                embeddings = np.array([out.outputs.embedding for out in outputs])
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms

                query_embedding = embeddings[0:1]
                candidate_embeddings = embeddings[1:]
                similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

                scored_candidates = list(zip(candidates, similarities))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_candidates[: self.top_k]

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
                    f"Embedder (vLLM) reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
                )
        finally:
            release_vllm(self.model_name, convert="embed")
            self.model = None  # Drop reference so pool eviction can free GPU memory

        self.progress_callback = None
        return doc
