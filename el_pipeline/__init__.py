"""
Modular EL pipeline package.

This package provides a configurable NER (Named Entity Recognition) pipeline
using spaCy's component architecture for NER, candidate generation, reranking,
and entity disambiguation.
"""

__all__ = [
    "PipelineConfig",
    "ELPipeline",
]

__version__ = "0.2.0"

# Import spacy_components to register factories with spaCy
from el_pipeline import spacy_components  # noqa: F401

from .config import PipelineConfig  # noqa: E402
from .pipeline import ELPipeline  # noqa: E402
