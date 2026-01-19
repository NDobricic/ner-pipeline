"""
spaCy NER components for the NER pipeline.

Provides factories and components for various NER implementations:
- LELAGLiNERComponent: Zero-shot GLiNER NER
- SimpleNERComponent: Regex-based NER
- GLiNERComponent: Standard GLiNER wrapper
- TransformersNERComponent: HuggingFace NER
- NERFilterComponent: Post-filter for spaCy's built-in NER
"""

import logging
import re
from typing import List, Optional, Callable

from spacy.language import Language
from spacy.tokens import Doc, Span

from ner_pipeline.context import extract_context
from ner_pipeline.lela.config import (
    DEFAULT_GLINER_MODEL,
    NER_LABELS,
)

logger = logging.getLogger(__name__)

# Lazy imports
_GLiNER = None


def _get_gliner():
    """Lazy import of GLiNER."""
    global _GLiNER
    if _GLiNER is None:
        try:
            from gliner import GLiNER
            _GLiNER = GLiNER
        except ImportError:
            raise ImportError(
                "gliner package required for GLiNER NER. "
                "Install with: pip install gliner"
            )
    return _GLiNER


# ============================================================================
# LELA GLiNER NER Component
# ============================================================================

@Language.factory(
    "ner_pipeline_lela_gliner",
    default_config={
        "model_name": DEFAULT_GLINER_MODEL,
        "labels": list(NER_LABELS),
        "threshold": 0.5,
        "context_mode": "sentence",
    },
)
def create_lela_gliner_component(
    nlp: Language,
    name: str,
    model_name: str,
    labels: List[str],
    threshold: float,
    context_mode: str,
):
    """Factory for LELA GLiNER NER component."""
    return LELAGLiNERComponent(
        nlp=nlp,
        model_name=model_name,
        labels=labels,
        threshold=threshold,
        context_mode=context_mode,
    )


class LELAGLiNERComponent:
    """
    Zero-shot GLiNER NER component for spaCy.

    Uses the GLiNER library for zero-shot named entity recognition.
    Entities are added to doc.ents with context stored in span._.context.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_GLINER_MODEL,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        context_mode: str = "sentence",
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.labels = labels if labels is not None else list(NER_LABELS)
        self.threshold = threshold
        self.context_mode = context_mode

        # Register custom extension for context
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        GLiNER = _get_gliner()
        logger.info(f"Loading LELA GLiNER model: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)
        logger.info(f"LELA GLiNER loaded with labels: {self.labels}")

    def __call__(self, doc: Doc) -> Doc:
        """Process document and add entities."""
        text = doc.text
        if not text or not text.strip():
            return doc

        predictions = self.model.predict_entities(
            text,
            labels=self.labels,
            threshold=self.threshold,
        )

        spans = []
        for pred in predictions:
            start_char = pred["start"]
            end_char = pred["end"]

            # Find token boundaries
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is None:
                continue

            # Create span with label
            label = pred.get("label", "ENT")
            new_span = Span(doc, span.start, span.end, label=label)

            # Store context
            context = extract_context(text, start_char, end_char, mode=self.context_mode)
            new_span._.context = context

            spans.append(new_span)

        # Filter overlapping spans (keep longest)
        doc.ents = self._filter_spans(spans)

        logger.debug(f"Extracted {len(doc.ents)} entities from document")
        return doc

    def _filter_spans(self, spans: List[Span]) -> List[Span]:
        """Filter overlapping spans, keeping the longest."""
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


# ============================================================================
# Simple Regex NER Component
# ============================================================================

@Language.factory(
    "ner_pipeline_simple",
    default_config={
        "min_len": 3,
        "context_mode": "sentence",
    },
)
def create_simple_ner_component(
    nlp: Language,
    name: str,
    min_len: int,
    context_mode: str,
):
    """Factory for simple regex NER component."""
    return SimpleNERComponent(nlp=nlp, min_len=min_len, context_mode=context_mode)


class SimpleNERComponent:
    """
    Lightweight regex-based NER component for spaCy.

    Extracts capitalized multi-word sequences as entities.
    Useful for quick tests without external dependencies.
    """

    def __init__(
        self,
        nlp: Language,
        min_len: int = 3,
        context_mode: str = "sentence",
    ):
        self.nlp = nlp
        self.pattern = re.compile(
            r"\b([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*)\b"
        )
        self.min_len = min_len
        self.context_mode = context_mode

        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

    def __call__(self, doc: Doc) -> Doc:
        """Process document and add entities."""
        text = doc.text
        spans = []

        for match in self.pattern.finditer(text):
            match_text = match.group(1)
            if len(match_text) < self.min_len:
                continue

            start_char = match.start(1)
            end_char = match.end(1)

            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is None:
                continue

            new_span = Span(doc, span.start, span.end, label="ENT")
            context = extract_context(text, start_char, end_char, mode=self.context_mode)
            new_span._.context = context
            spans.append(new_span)

        doc.ents = self._filter_spans(spans)
        return doc

    def _filter_spans(self, spans: List[Span]) -> List[Span]:
        """Filter overlapping spans, keeping the longest."""
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))
        result = []
        seen_tokens = set()

        for span in sorted_spans:
            span_tokens = set(range(span.start, span.end))
            if not span_tokens & seen_tokens:
                result.append(span)
                seen_tokens.update(span_tokens)

        return sorted(result, key=lambda s: s.start)


# ============================================================================
# Standard GLiNER Component (non-LELA)
# ============================================================================

@Language.factory(
    "ner_pipeline_gliner",
    default_config={
        "model_name": "urchade/gliner_base",
        "labels": ["person", "organization", "location"],
        "threshold": 0.5,
        "context_mode": "sentence",
    },
)
def create_gliner_component(
    nlp: Language,
    name: str,
    model_name: str,
    labels: List[str],
    threshold: float,
    context_mode: str,
):
    """Factory for standard GLiNER component."""
    return GLiNERComponent(
        nlp=nlp,
        model_name=model_name,
        labels=labels,
        threshold=threshold,
        context_mode=context_mode,
    )


class GLiNERComponent:
    """
    Standard GLiNER NER component for spaCy.

    Similar to LELAGLiNERComponent but uses different defaults.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = "urchade/gliner_base",
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        context_mode: str = "sentence",
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.labels = labels or ["person", "organization", "location"]
        self.threshold = threshold
        self.context_mode = context_mode

        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        GLiNER = _get_gliner()
        logger.info(f"Loading GLiNER model: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)

    def __call__(self, doc: Doc) -> Doc:
        """Process document and add entities."""
        text = doc.text
        if not text or not text.strip():
            return doc

        predictions = self.model.predict_entities(
            text,
            labels=self.labels,
            threshold=self.threshold,
        )

        spans = []
        for pred in predictions:
            start_char = pred["start"]
            end_char = pred["end"]

            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is None:
                continue

            label = pred.get("label", "ENT")
            new_span = Span(doc, span.start, span.end, label=label)
            context = extract_context(text, start_char, end_char, mode=self.context_mode)
            new_span._.context = context
            spans.append(new_span)

        doc.ents = self._filter_spans(spans)
        return doc

    def _filter_spans(self, spans: List[Span]) -> List[Span]:
        """Filter overlapping spans."""
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))
        result = []
        seen_tokens = set()

        for span in sorted_spans:
            span_tokens = set(range(span.start, span.end))
            if not span_tokens & seen_tokens:
                result.append(span)
                seen_tokens.update(span_tokens)

        return sorted(result, key=lambda s: s.start)


# ============================================================================
# Transformers NER Component
# ============================================================================

@Language.factory(
    "ner_pipeline_transformers",
    default_config={
        "model_name": "dslim/bert-base-NER",
        "context_mode": "sentence",
        "aggregation_strategy": "simple",
    },
)
def create_transformers_ner_component(
    nlp: Language,
    name: str,
    model_name: str,
    context_mode: str,
    aggregation_strategy: str,
):
    """Factory for Transformers NER component."""
    return TransformersNERComponent(
        nlp=nlp,
        model_name=model_name,
        context_mode=context_mode,
        aggregation_strategy=aggregation_strategy,
    )


class TransformersNERComponent:
    """
    HuggingFace Transformers NER component for spaCy.

    Uses the transformers pipeline for token classification.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = "dslim/bert-base-NER",
        context_mode: str = "sentence",
        aggregation_strategy: str = "simple",
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.context_mode = context_mode
        self.aggregation_strategy = aggregation_strategy

        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        # Lazy import transformers
        try:
            from transformers import pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy=aggregation_strategy,
            )
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        logger.info(f"Loaded Transformers NER model: {model_name}")

    def __call__(self, doc: Doc) -> Doc:
        """Process document and add entities."""
        text = doc.text
        if not text or not text.strip():
            return doc

        predictions = self.ner_pipeline(text)

        spans = []
        for pred in predictions:
            start_char = pred["start"]
            end_char = pred["end"]

            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is None:
                continue

            # Clean label (remove B-/I- prefixes)
            label = pred.get("entity_group", pred.get("entity", "ENT"))
            if label.startswith(("B-", "I-")):
                label = label[2:]

            new_span = Span(doc, span.start, span.end, label=label)
            context = extract_context(text, start_char, end_char, mode=self.context_mode)
            new_span._.context = context
            spans.append(new_span)

        doc.ents = self._filter_spans(spans)
        return doc

    def _filter_spans(self, spans: List[Span]) -> List[Span]:
        """Filter overlapping spans."""
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))
        result = []
        seen_tokens = set()

        for span in sorted_spans:
            span_tokens = set(range(span.start, span.end))
            if not span_tokens & seen_tokens:
                result.append(span)
                seen_tokens.update(span_tokens)

        return sorted(result, key=lambda s: s.start)


# ============================================================================
# NER Filter Component (for spaCy's built-in NER)
# ============================================================================

@Language.component("ner_pipeline_ner_filter")
def ner_filter_component(doc: Doc) -> Doc:
    """
    Post-filter for spaCy's built-in NER.

    Adds context extension to existing entities from spaCy's NER.
    Use this after spaCy's built-in 'ner' component.
    """
    if not Span.has_extension("context"):
        Span.set_extension("context", default=None)

    text = doc.text
    for ent in doc.ents:
        context = extract_context(text, ent.start_char, ent.end_char, mode="sentence")
        ent._.context = context

    return doc
