"""
Modular NER pipeline package.

This package provides a configurable NER (Named Entity Recognition) pipeline
using spaCy's component architecture for NER, candidate generation, reranking,
and entity disambiguation.
"""

__all__ = [
    "PipelineConfig",
    "NERPipeline",
]

__version__ = "0.2.0"

# Import spacy_components to register factories with spaCy
from ner_pipeline import spacy_components  # noqa: F401

from .config import PipelineConfig  # noqa: E402
from .pipeline import NERPipeline  # noqa: E402
