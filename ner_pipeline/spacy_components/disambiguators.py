"""
spaCy disambiguator components for the NER pipeline.

Provides factories and components for entity disambiguation:
- LELAvLLMDisambiguatorComponent: vLLM-based LLM disambiguation
- FirstDisambiguatorComponent: Select first candidate
- PopularityDisambiguatorComponent: Select by highest score
"""

import logging
import re
from collections import Counter
from typing import List, Optional, Tuple

from spacy.language import Language
from spacy.tokens import Doc, Span

from ner_pipeline.knowledge_bases.base import KnowledgeBase
from ner_pipeline.lela.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_GENERATION_CONFIG,
    SPAN_OPEN,
    SPAN_CLOSE,
)
from ner_pipeline.lela.prompts import (
    create_disambiguation_messages,
    DEFAULT_SYSTEM_PROMPT,
)
from ner_pipeline.lela.llm_pool import get_vllm_instance

logger = logging.getLogger(__name__)

# Lazy imports
_vllm = None
_SamplingParams = None


def _get_vllm():
    """Lazy import of vllm."""
    global _vllm, _SamplingParams
    if _vllm is None:
        try:
            import vllm
            from vllm import SamplingParams
            _vllm = vllm
            _SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "vllm package required for LLM disambiguation. "
                "Install with: pip install vllm"
            )
    return _vllm, _SamplingParams


def _ensure_extensions():
    """Ensure required extensions are registered on Span."""
    if not Span.has_extension("candidates"):
        Span.set_extension("candidates", default=[])
    if not Span.has_extension("resolved_entity"):
        Span.set_extension("resolved_entity", default=None)


# ============================================================================
# LELA vLLM Disambiguator Component
# ============================================================================

@Language.factory(
    "ner_pipeline_lela_vllm_disambiguator",
    default_config={
        "model_name": DEFAULT_LLM_MODEL,
        "tensor_parallel_size": DEFAULT_TENSOR_PARALLEL_SIZE,
        "max_model_len": DEFAULT_MAX_MODEL_LEN,
        "add_none_candidate": False,
        "add_descriptions": True,
        "disable_thinking": False,
        "system_prompt": None,
        "generation_config": None,
        "self_consistency_k": 1,
    },
)
def create_lela_vllm_disambiguator_component(
    nlp: Language,
    name: str,
    model_name: str,
    tensor_parallel_size: int,
    max_model_len: Optional[int],
    add_none_candidate: bool,
    add_descriptions: bool,
    disable_thinking: bool,
    system_prompt: Optional[str],
    generation_config: Optional[dict],
    self_consistency_k: int,
):
    """Factory for LELA vLLM disambiguator component."""
    return LELAvLLMDisambiguatorComponent(
        nlp=nlp,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        add_none_candidate=add_none_candidate,
        add_descriptions=add_descriptions,
        disable_thinking=disable_thinking,
        system_prompt=system_prompt,
        generation_config=generation_config,
        self_consistency_k=self_consistency_k,
    )


class LELAvLLMDisambiguatorComponent:
    """
    vLLM-based entity disambiguator component for spaCy.

    Uses vLLM for fast batched LLM inference to select the best entity.
    Sets span.kb_id_ to the selected entity ID and span._.resolved_entity
    to the full entity object.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_LLM_MODEL,
        tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
        max_model_len: Optional[int] = DEFAULT_MAX_MODEL_LEN,
        add_none_candidate: bool = False,
        add_descriptions: bool = True,
        disable_thinking: bool = False,
        system_prompt: Optional[str] = None,
        generation_config: Optional[dict] = None,
        self_consistency_k: int = 1,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.add_none_candidate = add_none_candidate
        self.add_descriptions = add_descriptions
        self.disable_thinking = disable_thinking
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.self_consistency_k = self_consistency_k

        _ensure_extensions()

        # Get vLLM and SamplingParams
        vllm, SamplingParams = _get_vllm()

        # Get or create LLM instance
        self.llm = get_vllm_instance(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

        # Set up sampling parameters
        gen_config = generation_config or DEFAULT_GENERATION_CONFIG
        sampling_config = {**gen_config, "n": self_consistency_k}
        self.sampling_params = SamplingParams(**sampling_config)

        # Initialize lazily
        self.kb = None

        logger.info(f"LELA vLLM disambiguator initialized: {model_name}")

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        self.kb = kb

    @staticmethod
    def _parse_output(output: str) -> int:
        """Parse LLM output to extract answer index."""
        match = re.search(r'"?answer"?:\s*(\d+)', output)
        if match:
            return int(match.group(1))
        logger.debug(f"Unexpected output format: {output}")
        return 0

    def _apply_self_consistency(self, outputs: list) -> int:
        """Apply self-consistency voting over multiple outputs."""
        if self.self_consistency_k == 1:
            return self._parse_output(outputs[0].text)
        answers = [self._parse_output(o.text) for o in outputs]
        return Counter(answers).most_common(1)[0][0]

    def _mark_mention(self, text: str, start: int, end: int) -> str:
        """Mark mention in text with brackets."""
        return f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"

    def __call__(self, doc: Doc) -> Doc:
        """Disambiguate all entities in the document."""
        if self.kb is None:
            logger.warning("vLLM disambiguator not initialized - call initialize(kb) first")
            return doc

        text = doc.text

        for ent in doc.ents:
            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            # If only one candidate and no none option, select it directly
            if len(candidates) == 1 and not self.add_none_candidate:
                title = candidates[0][0]
                entity = self.kb.get_entity(title)
                if entity:
                    ent._.resolved_entity = entity
                continue

            # Mark mention in text
            marked_text = self._mark_mention(text, ent.start_char, ent.end_char)

            # Create messages for LLM
            messages = create_disambiguation_messages(
                marked_text=marked_text,
                candidates=candidates,
                system_prompt=self.system_prompt,
                add_none_candidate=self.add_none_candidate,
                add_descriptions=self.add_descriptions,
            )

            # Get chat template kwargs
            chat_kwargs = {}
            if self.disable_thinking:
                chat_kwargs["enable_thinking"] = False

            try:
                responses = self.llm.chat(
                    [messages],
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    chat_template_kwargs=chat_kwargs if chat_kwargs else {},
                )
                response = responses[0] if responses else None
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                continue

            if response is None:
                continue

            try:
                answer = self._apply_self_consistency(response.outputs)

                # Answer 0 means "None" if add_none_candidate is True
                if answer == 0:
                    continue

                if 0 < answer <= len(candidates):
                    selected_title = candidates[answer - 1][0]
                    entity = self.kb.get_entity(selected_title)
                    if entity:
                        ent._.resolved_entity = entity
                else:
                    logger.debug(f"Answer {answer} out of range for {len(candidates)} candidates")

            except Exception as e:
                logger.error(f"Error processing LLM response: {e}")
                continue

        return doc


# ============================================================================
# First Candidate Disambiguator Component
# ============================================================================

@Language.factory(
    "ner_pipeline_first_disambiguator",
    default_config={},
)
def create_first_disambiguator_component(
    nlp: Language,
    name: str,
):
    """Factory for first candidate disambiguator component."""
    return FirstDisambiguatorComponent(nlp=nlp)


class FirstDisambiguatorComponent:
    """
    First candidate disambiguator component for spaCy.

    Simply selects the first candidate in the list.
    """

    def __init__(self, nlp: Language):
        self.nlp = nlp
        self.kb = None

        _ensure_extensions()

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        self.kb = kb

    def __call__(self, doc: Doc) -> Doc:
        """Select first candidate for all entities."""
        if self.kb is None:
            logger.warning("First disambiguator not initialized - call initialize(kb) first")
            return doc

        for ent in doc.ents:
            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            # Select first candidate
            title = candidates[0][0]
            entity = self.kb.get_entity(title)
            if entity:
                ent._.resolved_entity = entity

        return doc


# ============================================================================
# Popularity Disambiguator Component
# ============================================================================

@Language.factory(
    "ner_pipeline_popularity_disambiguator",
    default_config={},
)
def create_popularity_disambiguator_component(
    nlp: Language,
    name: str,
):
    """Factory for popularity disambiguator component."""
    return PopularityDisambiguatorComponent(nlp=nlp)


class PopularityDisambiguatorComponent:
    """
    Popularity-based disambiguator component for spaCy.

    Selects the candidate with the highest score.
    Since candidates in LELA format don't have scores, this uses position
    (first candidate is assumed to have highest score from retrieval).
    """

    def __init__(self, nlp: Language):
        self.nlp = nlp
        self.kb = None

        _ensure_extensions()

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        self.kb = kb

    def __call__(self, doc: Doc) -> Doc:
        """Select best candidate for all entities."""
        if self.kb is None:
            logger.warning("Popularity disambiguator not initialized - call initialize(kb) first")
            return doc

        for ent in doc.ents:
            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            # In LELA format, candidates are already sorted by score
            # So first candidate has highest score
            title = candidates[0][0]
            entity = self.kb.get_entity(title)
            if entity:
                ent._.resolved_entity = entity

        return doc
