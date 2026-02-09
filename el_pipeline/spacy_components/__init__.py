"""
spaCy components for EL pipeline.

This module provides spaCy-compatible pipeline components for:
- NER (named entity recognition)
- Candidate generation
- Reranking
- Disambiguation

Import this module to register all factories with spaCy.
"""

from . import ner
from . import candidates
from . import rerankers
from . import disambiguators

__all__ = [
    "ner",
    "candidates",
    "rerankers",
    "disambiguators",
]
