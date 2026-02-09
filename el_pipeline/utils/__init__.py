"""
Shared utilities for the EL pipeline.
"""

from el_pipeline.utils.spans import filter_spans
from el_pipeline.utils.extensions import (
    ensure_candidates_extension,
    ensure_resolved_entity_extension,
    ensure_context_extension,
)

__all__ = [
    "filter_spans",
    "ensure_candidates_extension",
    "ensure_resolved_entity_extension",
    "ensure_context_extension",
]
