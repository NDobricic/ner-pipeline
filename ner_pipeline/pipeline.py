"""
NER Pipeline with spaCy integration.

This module provides the main NERPipeline class that orchestrates
document processing using spaCy's pipeline architecture.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Import spacy_components to register factories
from ner_pipeline import spacy_components  # noqa: F401

# Keep loader and KB registration
from ner_pipeline import loaders as _loaders_pkg  # noqa: F401
from ner_pipeline import knowledge_bases as _kb_pkg  # noqa: F401

from .config import PipelineConfig
from .registry import (
    knowledge_bases,
    loaders,
)
from .types import Document, tuples_to_candidates


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

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize knowledge base (kept as custom registry)
        self.kb = None
        if config.knowledge_base:
            kb_factory = knowledge_bases.get(config.knowledge_base.name)
            self.kb = kb_factory(**config.knowledge_base.params)

        # Initialize loader (kept as custom registry)
        loader_factory = loaders.get(config.loader.name)
        self.loader = loader_factory(**config.loader.params)

        # Build spaCy pipeline
        self.nlp = self._build_nlp_pipeline(config)

    def _build_nlp_pipeline(self, config: PipelineConfig) -> Language:
        """
        Build spaCy pipeline from configuration.

        Args:
            config: Pipeline configuration

        Returns:
            Configured spaCy Language instance
        """
        # Start with blank English model
        nlp = spacy.blank("en")

        # Add NER component
        ner_name = config.ner.name
        ner_params = dict(config.ner.params)

        if ner_name == "spacy":
            # Use spaCy's built-in NER with a pretrained model
            model_name = ner_params.pop("model", "en_core_web_sm")
            spacy_nlp = spacy.load(model_name)
            # Copy NER component
            if "ner" in spacy_nlp.pipe_names:
                nlp.add_pipe("ner", source=spacy_nlp)
            # Add filter to set context
            nlp.add_pipe("ner_pipeline_ner_filter")
        else:
            factory_name = NER_COMPONENT_MAP.get(ner_name)
            if factory_name is None:
                raise ValueError(f"Unknown NER component: {ner_name}")
            nlp.add_pipe(factory_name, config=ner_params)

        # Add candidate generation component
        cand_name = config.candidate_generator.name
        cand_params = dict(config.candidate_generator.params)
        factory_name = CANDIDATES_COMPONENT_MAP.get(cand_name)
        if factory_name is None:
            raise ValueError(f"Unknown candidate generator: {cand_name}")
        cand_component = nlp.add_pipe(factory_name, config=cand_params)

        # Initialize with KB if needed
        if hasattr(cand_component, "initialize") and self.kb is not None:
            cand_component.initialize(self.kb)

        # Add reranker component (optional)
        if config.reranker and config.reranker.name != "none":
            rerank_name = config.reranker.name
            rerank_params = dict(config.reranker.params)
            factory_name = RERANKER_COMPONENT_MAP.get(rerank_name)
            if factory_name is None:
                raise ValueError(f"Unknown reranker: {rerank_name}")
            nlp.add_pipe(factory_name, config=rerank_params)

        # Add disambiguator component (optional)
        if config.disambiguator:
            disamb_name = config.disambiguator.name
            disamb_params = dict(config.disambiguator.params)
            factory_name = DISAMBIGUATOR_COMPONENT_MAP.get(disamb_name)
            if factory_name is None:
                raise ValueError(f"Unknown disambiguator: {disamb_name}")
            disamb_component = nlp.add_pipe(factory_name, config=disamb_params)

            # Initialize with KB if needed
            if hasattr(disamb_component, "initialize") and self.kb is not None:
                disamb_component.initialize(self.kb)

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
            # Get candidates (LELA format: List[Tuple[str, str]])
            candidates_tuples = getattr(ent._, "candidates", [])
            # Convert to Candidate objects for output
            candidates = tuples_to_candidates(candidates_tuples)

            # Get resolved entity
            resolved_entity = getattr(ent._, "resolved_entity", None)

            # Get context
            context = getattr(ent._, "context", None)

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
        return self._serialize_doc(spacy_doc, doc)

    def run(self, paths: Iterable[str], output_path: Optional[str] = None) -> List[Dict]:
        """
        Process multiple files through the pipeline.

        Args:
            paths: Iterable of file paths to process
            output_path: Optional path for JSONL output

        Returns:
            List of result dicts
        """
        results: List[Dict] = []
        writer = None
        if output_path:
            writer = Path(output_path).open("w", encoding="utf-8")

        try:
            for path in paths:
                for doc in self._load_with_cache(path):
                    result = self.process_document(doc)
                    if writer:
                        writer.write(json.dumps(result) + "\n")
                    results.append(result)
        finally:
            if writer:
                writer.close()

        return results
