"""
spaCy extension management utilities.

Provides functions to safely register custom extensions on Span objects.
"""

from spacy.tokens import Span


def ensure_context_extension() -> None:
    """Ensure the context extension is registered on Span."""
    if not Span.has_extension("context"):
        Span.set_extension("context", default=None)


def ensure_candidates_extension() -> None:
    """Ensure the candidates extensions are registered on Span."""
    if not Span.has_extension("candidates"):
        Span.set_extension("candidates", default=[])
    if not Span.has_extension("candidate_scores"):
        Span.set_extension("candidate_scores", default=[])


def ensure_resolved_entity_extension() -> None:
    """Ensure the resolved_entity extension is registered on Span."""
    if not Span.has_extension("resolved_entity"):
        Span.set_extension("resolved_entity", default=None)
