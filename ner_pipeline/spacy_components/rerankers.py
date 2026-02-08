"""
spaCy reranker components for the NER pipeline.

Provides factories and components for candidate reranking:
- CrossEncoderRerankerComponent: Cross-encoder reranking
- NoOpRerankerComponent: Pass-through (no reranking)
"""

import logging
import requests
from typing import List, Optional

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
from ner_pipeline.lela.llm_pool import (
    get_sentence_transformer_instance,
    release_sentence_transformer,
)
from ner_pipeline.utils import ensure_candidates_extension
from ner_pipeline.types import Candidate, ProgressCallback

logger = logging.getLogger(__name__)


# ============================================================================
# Cross-Encoder Reranker Component
# ============================================================================


@Language.factory(
    "ner_pipeline_cross_encoder_reranker",
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
    "ner_pipeline_vllm_api_client_reranker",
    default_config={
        "model_name": "Qwen/Qwen3-Reranker-4B-seq-cls",
        "top_k": 10,
        "base_url": "http://localhost",
        "port": 8000,
    },
)
def create_vllm_api_client_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    base_url: str,
    port: int,
):
    """Factory for vLLM API client reranker component."""
    return VLLMAPIClientReranker(
        nlp=nlp,
        model_name=model_name,
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
        model_name: str,
        top_k: int = 10,
        base_url: str = "http://localhost",
        port: int = 8000,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.api_url = f"{base_url}:{port}/score"
        ensure_candidates_extension()
        logger.info(
            f"Using vLLM API reranker for model '{self.model_name}' at {self.api_url}"
        )

    @staticmethod
    def post_http_request(prompt: dict, api_url: str) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers=headers, json=prompt)
        response.raise_for_status()
        return response

    def __call__(self, doc: Doc) -> Doc:
        for ent in doc.ents:
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
                        "model": self.model_name,
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
        return doc


# ============================================================================
# No-Op Reranker Component
# ============================================================================


@Language.factory(
    "ner_pipeline_noop_reranker",
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
    "ner_pipeline_llama_server_reranker",
    default_config={
        "model_name": "qwen3-reranker",
        "top_k": 10,
        "base_url": "http://localhost",
        "port": 8002,
    },
)
def create_llama_server_reranker_component(
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
        port: int = 8002,
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
