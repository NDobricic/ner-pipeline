"""
NER Pipeline with spaCy integration.

This module provides the main NERPipeline class that orchestrates
document processing using spaCy's pipeline architecture.
"""

import hashlib
import json
import os
import pickle
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Import spacy_components to register factories
from ner_pipeline import spacy_components  # noqa: F401

# Keep loader and KB registration
from ner_pipeline import loaders as _loaders_pkg  # noqa: F401
from ner_pipeline import knowledge_bases as _kb_pkg  # noqa: F401

from .config import ComponentConfig, PipelineConfig
from .registry import (
    knowledge_bases,
    loaders,
)
from .types import Candidate, Document, ProgressCallback

# Component name mapping from config names to spaCy factory names
NER_COMPONENT_MAP = {
    "lela_gliner": "ner_pipeline_lela_gliner",
    "simple": "ner_pipeline_simple",
    "gliner": "ner_pipeline_gliner",
    "transformers": "ner_pipeline_transformers",
    "spacy": None,  # Use built-in spaCy NER
}

CANDIDATES_COMPONENT_MAP = {
    "lela_bm25": "ner_pipeline_lela_bm25_candidates",
    "lela_dense": "ner_pipeline_lela_dense_candidates",
    "fuzzy": "ner_pipeline_fuzzy_candidates",
    "bm25": "ner_pipeline_bm25_candidates",
}

RERANKER_COMPONENT_MAP = {
    "lela_embedder": "ner_pipeline_lela_embedder_reranker",
    "cross_encoder": "ner_pipeline_cross_encoder_reranker",
    "none": "ner_pipeline_noop_reranker",
}

DISAMBIGUATOR_COMPONENT_MAP = {
    "lela_vllm": "ner_pipeline_lela_vllm_disambiguator",
    "lela_transformers": "ner_pipeline_lela_transformers_disambiguator",
    "first": "ner_pipeline_first_disambiguator",
    "popularity": "ner_pipeline_popularity_disambiguator",
}


class NERPipeline:
    """
    Orchestrates the modular NER pipeline using spaCy.

    The pipeline uses spaCy's component system for NER, candidate generation,
    reranking, and disambiguation, while keeping document loaders and
    knowledge bases as custom registry components.
    """

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cancel_event = cancel_event

        def report(progress: float, desc: str):
            if progress_callback:
                progress_callback(progress, desc)

        def check_cancelled():
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Pipeline initialization cancelled")

        report(0.0, "Loading knowledge base...")
        check_cancelled()

        if config.knowledge_base is None:
            from ner_pipeline.knowledge_bases.yago_downloader import ensure_yago_kb

            path = ensure_yago_kb()
            config.knowledge_base = ComponentConfig(
                name="custom", params={"path": path}
            )

        self.kb = None
        if config.knowledge_base:
            config.knowledge_base.params["cache_dir"] = str(self.cache_dir)
            # Pass cancel_event and progress_callback to KB if it supports them
            config.knowledge_base.params["cancel_event"] = cancel_event
            config.knowledge_base.params["progress_callback"] = progress_callback
            kb_factory = knowledge_bases.get(config.knowledge_base.name)
            # Filter params to only those the factory accepts
            import inspect

            sig = inspect.signature(kb_factory)
            valid_params = {
                k: v
                for k, v in config.knowledge_base.params.items()
                if k in sig.parameters
            }
            self.kb = kb_factory(**valid_params)

        check_cancelled()
        report(0.15, "Initializing document loader...")
        loader_factory = loaders.get(config.loader.name)
        self.loader = loader_factory(**config.loader.params)

        check_cancelled()
        # Build spaCy pipeline
        report(0.2, "Building spaCy pipeline...")
        self.nlp = self._build_nlp_pipeline(config, progress_callback, cancel_event)

    def _build_nlp_pipeline(
        self,
        config: PipelineConfig,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Language:
        """
        Build spaCy pipeline from configuration.

        Args:
            config: Pipeline configuration
            progress_callback: Optional callback for progress reporting
            cancel_event: Optional event to check for cancellation

        Returns:
            Configured spaCy Language instance
        """

        def report(progress: float, desc: str):
            if progress_callback:
                progress_callback(progress, desc)

        def check_cancelled():
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Pipeline initialization cancelled")

        ner_name = config.ner.name
        ner_params = dict(config.ner.params)

        if ner_name == "spacy":
            # Use spaCy's built-in NER with a pretrained model
            model_name = ner_params.pop("model", "en_core_web_sm")
            nlp = spacy.load(model_name)
            check_cancelled()
            # Add a filter to set context on existing entities after NER runs.
            # We add it 'after' the 'ner' component to ensure entities are present.
            if "ner" in nlp.pipe_names:
                nlp.add_pipe("ner_pipeline_ner_filter", after="ner")
            else:
                # If the loaded model has no NER, we still add the filter
                # in case another component adds entities.
                nlp.add_pipe("ner_pipeline_ner_filter")
        else:
            # For custom NER components, start with a blank model
            nlp = spacy.blank("en")
            factory_name = NER_COMPONENT_MAP.get(ner_name)
            if factory_name is None:
                raise ValueError(f"Unknown NER component: {ner_name}")
            nlp.add_pipe(factory_name, config=ner_params)

        # Add candidate generation component
        check_cancelled()
        report(
            0.45, f"Loading candidate generator ({config.candidate_generator.name})..."
        )
        cand_name = config.candidate_generator.name
        cand_params = dict(config.candidate_generator.params)
        factory_name = CANDIDATES_COMPONENT_MAP.get(cand_name)
        if factory_name is None:
            raise ValueError(f"Unknown candidate generator: {cand_name}")
        cand_component = nlp.add_pipe(factory_name, config=cand_params)

        # Initialize with KB if needed
        if hasattr(cand_component, "initialize") and self.kb is not None:
            cand_component.initialize(self.kb, cache_dir=self.cache_dir)

        check_cancelled()
        # Add reranker component
        if config.reranker and config.reranker.name != "none":
            report(0.6, f"Loading reranker ({config.reranker.name})...")
            rerank_name = config.reranker.name
            rerank_params = dict(config.reranker.params)
            factory_name = RERANKER_COMPONENT_MAP.get(rerank_name)
            if factory_name is None:
                raise ValueError(f"Unknown reranker: {rerank_name}")
            nlp.add_pipe(factory_name, config=rerank_params)
        else:
            # Always add the noop reranker to enforce top_k truncation
            rerank_params = dict(config.reranker.params) if config.reranker else {}
            nlp.add_pipe("ner_pipeline_noop_reranker", config=rerank_params)

        check_cancelled()
        # Add disambiguator component (optional)
        if config.disambiguator:
            report(0.75, f"Loading disambiguator ({config.disambiguator.name})...")
            disamb_name = config.disambiguator.name
            disamb_params = dict(config.disambiguator.params)
            factory_name = DISAMBIGUATOR_COMPONENT_MAP.get(disamb_name)
            if factory_name is None:
                raise ValueError(f"Unknown disambiguator: {disamb_name}")
            disamb_component = nlp.add_pipe(factory_name, config=disamb_params)

            # Initialize with KB if needed
            if hasattr(disamb_component, "initialize") and self.kb is not None:
                disamb_component.initialize(self.kb)

        check_cancelled()
        report(1.0, "Pipeline initialization complete")
        return nlp

    def _cache_key(self, path: str) -> str:
        """Generate cache key for a file path."""
        stat = os.stat(path)
        raw = f"{path}-{stat.st_mtime}-{stat.st_size}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _load_with_cache(self, path: str) -> Iterator[Document]:
        """Load documents from path with caching."""
        key = self._cache_key(path)
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            for doc in cached:
                yield doc
            return

        docs = list(self.loader.load(path))
        with cache_file.open("wb") as f:
            pickle.dump(docs, f)
        for doc in docs:
            yield doc

    def _serialize_doc(self, spacy_doc: Doc, source_doc: Document) -> Dict:
        """
        Serialize spaCy Doc to output format.

        Args:
            spacy_doc: Processed spaCy Doc
            source_doc: Original source Document

        Returns:
            Dict with entities and metadata
        """
        entities = []

        for ent in spacy_doc.ents:
            # Get candidates (List[Candidate])
            candidates = getattr(ent._, "candidates", [])
            # Get candidate scores if available
            candidate_scores = getattr(ent._, "candidate_scores", [])

            # Get resolved entity
            resolved_entity = getattr(ent._, "resolved_entity", None)

            # Get context
            context = getattr(ent._, "context", None)

            # Compute linking confidence:
            # - If entity resolved: use the score for the resolved candidate
            # - If not resolved: None (indicates NIL / no match found in KB)
            linking_confidence = None
            if resolved_entity and candidates:
                # Find the score for the resolved entity by matching entity_id
                for i, c in enumerate(candidates):
                    if c.entity_id == resolved_entity.id and i < len(candidate_scores):
                        linking_confidence = candidate_scores[i]
                        break
                # Fallback: use top score if we can't find exact match
                if linking_confidence is None and candidate_scores:
                    linking_confidence = candidate_scores[0]

            entity_dict = {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "context": context,
                "entity_id": resolved_entity.id if resolved_entity else None,
                "entity_title": resolved_entity.title if resolved_entity else None,
                "entity_description": (
                    resolved_entity.description if resolved_entity else None
                ),
                "linking_confidence": linking_confidence,
                "candidates": [
                    {
                        "entity_id": c.entity_id,
                        "score": c.score,
                        "description": c.description,
                    }
                    for c in candidates
                ],
            }
            entities.append(entity_dict)

        return {
            "id": source_doc.id,
            "text": source_doc.text,
            "entities": entities,
            "meta": source_doc.meta,
        }

    def process_document(self, doc: Document) -> Dict:
        """
        Process a single document through the pipeline.

        Args:
            doc: Document to process

        Returns:
            Dict with extracted entities and metadata
        """
        # Run through spaCy pipeline
        spacy_doc = self.nlp(doc.text)

        # Serialize to output format
        result = self._serialize_doc(spacy_doc, doc)

        # Normalize linking confidence across entities in this document
        self._normalize_linking_confidence([result])

        return result

    def process_document_with_progress(
        self,
        doc: Document,
        progress_callback: Optional[ProgressCallback] = None,
        base_progress: float = 0.0,
        progress_range: float = 1.0,
    ) -> Dict:
        """
        Process a single document with progress callbacks for each stage.

        Args:
            doc: Document to process
            progress_callback: Optional callback(progress: float, description: str)
            base_progress: Starting progress value (0.0-1.0)
            progress_range: How much progress this processing represents (0.0-1.0)

        Returns:
            Dict with extracted entities and metadata
        """

        def report(local_progress: float, desc: str):
            if progress_callback:
                actual_progress = base_progress + local_progress * progress_range
                progress_callback(actual_progress, desc)

        # Create spaCy doc from text
        report(0.0, "Tokenizing document...")
        spacy_doc = self.nlp.make_doc(doc.text)

        # Get pipeline components
        components = list(self.nlp.pipeline)
        if not components:
            return self._serialize_doc(spacy_doc, doc)

        # Calculate progress per stage
        # Reserve some progress for entity-level processing
        num_components = len(components)
        component_progress = (
            0.9 / num_components
        )  # 90% for components, 10% for serialization

        current_progress = 0.0

        for i, (name, component) in enumerate(components):
            stage_name = self._get_stage_description(name)
            report(current_progress, f"{stage_name}...")

            # Set up component-level progress callback if the component supports it
            if hasattr(
                component, "progress_callback"
            ) and self._is_entity_processing_component(name):

                def make_component_callback(
                    base_prog: float, comp_prog: float, stage: str
                ):
                    def callback(local_progress: float, desc: str):
                        actual_progress = base_prog + local_progress * comp_prog
                        report(actual_progress, f"{stage}: {desc}")

                    return callback

                component.progress_callback = make_component_callback(
                    current_progress, component_progress, stage_name
                )

            # Run the component
            spacy_doc = component(spacy_doc)

            current_progress += component_progress

            # Report entity count after NER component
            if self._is_ner_component(name) and hasattr(spacy_doc, "ents"):
                num_ents = len(spacy_doc.ents)
                report(
                    current_progress,
                    f"{stage_name} complete: found {num_ents} entities",
                )

        import logging
        import sys

        logger = logging.getLogger(__name__)

        logger.info(
            "process_document_with_progress: all components done, serializing..."
        )
        sys.stderr.flush()
        report(0.95, "Serializing results...")
        result = self._serialize_doc(spacy_doc, doc)
        logger.info("process_document_with_progress: serialization done")
        sys.stderr.flush()

        # Normalize linking confidence across entities in this document
        logger.info("process_document_with_progress: normalizing confidence...")
        sys.stderr.flush()
        self._normalize_linking_confidence([result])
        logger.info("process_document_with_progress: normalization done")
        sys.stderr.flush()

        logger.info("process_document_with_progress: calling final report(1.0)...")
        sys.stderr.flush()
        report(1.0, "Document processing complete")
        logger.info("=== process_document_with_progress RETURNING ===")
        sys.stderr.flush()
        return result

    def _get_stage_description(self, component_name: str) -> str:
        """Get human-readable description for a component name."""
        descriptions = {
            "ner_pipeline_lela_gliner": "NER (GLiNER)",
            "ner_pipeline_simple": "NER (regex)",
            "ner_pipeline_gliner": "NER (GLiNER)",
            "ner_pipeline_transformers": "NER (Transformers)",
            "ner_pipeline_ner_filter": "NER context extraction",
            "ner": "NER (spaCy)",
            "ner_pipeline_lela_bm25_candidates": "Candidate generation (BM25)",
            "ner_pipeline_lela_dense_candidates": "Candidate generation (dense)",
            "ner_pipeline_fuzzy_candidates": "Candidate generation (fuzzy)",
            "ner_pipeline_bm25_candidates": "Candidate generation (BM25)",
            "ner_pipeline_lela_embedder_reranker": "Reranking (embedder)",
            "ner_pipeline_cross_encoder_reranker": "Reranking (cross-encoder)",
            "ner_pipeline_noop_reranker": "Reranking (pass-through)",
            "ner_pipeline_lela_vllm_disambiguator": "Disambiguation (LLM)",
            "ner_pipeline_lela_transformers_disambiguator": "Disambiguation (LLM)",
            "ner_pipeline_first_disambiguator": "Disambiguation",
            "ner_pipeline_popularity_disambiguator": "Disambiguation",
        }
        return descriptions.get(component_name, component_name)

    def _is_ner_component(self, component_name: str) -> bool:
        """Check if component is a NER component."""
        return component_name in (
            "ner_pipeline_lela_gliner",
            "ner_pipeline_simple",
            "ner_pipeline_gliner",
            "ner_pipeline_transformers",
            "ner",
            "ner_pipeline_ner_filter",
        )

    def _is_entity_processing_component(self, component_name: str) -> bool:
        """Check if component processes entities (candidates, reranking, disambiguation)."""
        return component_name in (
            "ner_pipeline_lela_bm25_candidates",
            "ner_pipeline_lela_dense_candidates",
            "ner_pipeline_fuzzy_candidates",
            "ner_pipeline_bm25_candidates",
            "ner_pipeline_lela_embedder_reranker",
            "ner_pipeline_cross_encoder_reranker",
            "ner_pipeline_lela_vllm_disambiguator",
            "ner_pipeline_lela_transformers_disambiguator",
            "ner_pipeline_first_disambiguator",
            "ner_pipeline_popularity_disambiguator",
        )

    def _normalize_linking_confidence(self, results: List[Dict]) -> None:
        """
        Normalize linking_confidence values across all entities in the batch.

        Uses robust min-max normalization (IQR-based) to handle outliers.
        Adds a 'linking_confidence_normalized' field to each entity dict.

        Args:
            results: List of document result dicts (modified in place)
        """
        # Collect all non-None linking_confidence values
        scores: List[float] = []
        for doc_result in results:
            for entity in doc_result.get("entities", []):
                conf = entity.get("linking_confidence")
                if conf is not None:
                    scores.append(float(conf))

        if not scores:
            # No scores to normalize, set all to None
            for doc_result in results:
                for entity in doc_result.get("entities", []):
                    entity["linking_confidence_normalized"] = None
            return

        scores_arr = np.array(scores)

        # Handle edge case: all scores identical
        if np.all(scores_arr == scores_arr[0]):
            # All same value -> normalize to 1.0 (or 0.5 if you prefer)
            for doc_result in results:
                for entity in doc_result.get("entities", []):
                    if entity.get("linking_confidence") is not None:
                        entity["linking_confidence_normalized"] = 1.0
                    else:
                        entity["linking_confidence_normalized"] = None
            return

        # Compute IQR-based bounds for robust normalization
        q1, q3 = np.percentile(scores_arr, [25, 75])
        iqr = q3 - q1

        if iqr == 0:
            # Very little spread, fall back to simple min-max
            lower = scores_arr.min()
            upper = scores_arr.max()
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            # Ensure bounds cover at least the actual data range
            lower = min(lower, scores_arr.min())
            upper = max(upper, scores_arr.max())

        # Apply normalization to each entity
        for doc_result in results:
            for entity in doc_result.get("entities", []):
                conf = entity.get("linking_confidence")
                if conf is not None:
                    clipped = np.clip(float(conf), lower, upper)
                    normalized = (clipped - lower) / (upper - lower)
                    entity["linking_confidence_normalized"] = float(normalized)
                else:
                    entity["linking_confidence_normalized"] = None

    def run(
        self, paths: Iterable[str], output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process multiple files through the pipeline.

        Args:
            paths: Iterable of file paths to process
            output_path: Optional path for JSONL output

        Returns:
            List of result dicts
        """
        results: List[Dict] = []

        for path in paths:
            for doc in self._load_with_cache(path):
                result = self.process_document(doc)
                results.append(result)

        # Normalize linking confidence across the entire batch
        self._normalize_linking_confidence(results)

        # Write to output file after normalization
        if output_path:
            with Path(output_path).open("w", encoding="utf-8") as writer:
                for result in results:
                    writer.write(json.dumps(result) + "\n")

        return results
