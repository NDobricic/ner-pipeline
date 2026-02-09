"""
Span filtering utilities for NER components.
"""

from typing import List

from spacy.tokens import Span


def filter_spans(spans: List[Span]) -> List[Span]:
    """
    Filter overlapping spans, keeping the longest.

    When spans overlap, the longest span is kept and shorter
    overlapping spans are discarded.

    Args:
        spans: List of spaCy Span objects

    Returns:
        Filtered list of non-overlapping spans, sorted by start position
    """
    if not spans:
        return []

    # Sort by length (descending) then by start position
    sorted_spans = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))

    result = []
    seen_tokens = set()

    for span in sorted_spans:
        span_tokens = set(range(span.start, span.end))
        if not span_tokens & seen_tokens:
            result.append(span)
            seen_tokens.update(span_tokens)

    # Sort by start position for final result
    return sorted(result, key=lambda s: s.start)
