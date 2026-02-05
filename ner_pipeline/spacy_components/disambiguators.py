"""
spaCy disambiguator components for the NER pipeline.

Provides factories and components for entity disambiguation:
- LELAvLLMDisambiguatorComponent: vLLM-based LLM disambiguation (all candidates at once)
- FirstDisambiguatorComponent: Select first candidate
- PopularityDisambiguatorComponent: Select by highest score
"""

import logging
import re
from collections import Counter
from typing import List, Optional

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
from ner_pipeline.lela.llm_pool import get_vllm_instance, release_vllm
from ner_pipeline.utils import ensure_candidates_extension, ensure_resolved_entity_extension
from ner_pipeline.types import Candidate, ProgressCallback

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
    ensure_candidates_extension()
    ensure_resolved_entity_extension()


# ============================================================================
# LELA vLLM Disambiguator Component
# ============================================================================

@Language.factory(
    "ner_pipeline_lela_vllm_disambiguator",
    default_config={
        "model_name": DEFAULT_LLM_MODEL,
        "tensor_parallel_size": DEFAULT_TENSOR_PARALLEL_SIZE,
        "max_model_len": DEFAULT_MAX_MODEL_LEN,
        "add_none_candidate": True,  # Enable NIL handling by default
        "add_descriptions": True,
        "disable_thinking": True,  # Disable thinking by default for faster responses
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
    
    Memory management: LLM is loaded on-demand and released after use,
    allowing previous stage models to be evicted to free memory.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_LLM_MODEL,
        tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
        max_model_len: Optional[int] = DEFAULT_MAX_MODEL_LEN,
        add_none_candidate: bool = False,
        add_descriptions: bool = True,
        disable_thinking: bool = True,  # Default True for faster responses
        system_prompt: Optional[str] = None,
        generation_config: Optional[dict] = None,
        self_consistency_k: int = 1,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.add_none_candidate = add_none_candidate
        self.add_descriptions = add_descriptions
        self.disable_thinking = disable_thinking
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.self_consistency_k = self_consistency_k
        self.generation_config = generation_config or DEFAULT_GENERATION_CONFIG

        _ensure_extensions()

        # LLM loaded on-demand in __call__, not here
        # This allows lazy eviction of previous stage models
        self.llm = None
        self.sampling_params = None

        # Initialize lazily
        self.kb = None
        
        # Optional progress callback for fine-grained progress reporting
        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"LELA vLLM disambiguator initialized: {model_name}")

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        self.kb = kb

    @staticmethod
    def _parse_output(output: str) -> int:
        """Parse LLM output to extract answer index.
        
        Handles multiple formats:
        - "answer": 3  (standard format)
        - answer: 3
        - 3  (just a number, common with /no_think mode)
        """
        # Try standard format first: "answer": N or answer: N
        match = re.search(r'"?answer"?\s*:\s*(\d+)', output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Try to find any standalone number (for /no_think mode which may output just "3")
        # Look for a number that's either at the start/end or surrounded by whitespace/punctuation
        match = re.search(r'(?:^|\s)(\d+)(?:\s|$|\.)', output.strip())
        if match:
            return int(match.group(1))
        
        # Last resort: find any digit
        match = re.search(r'(\d+)', output)
        if match:
            return int(match.group(1))
        
        logger.debug(f"Could not parse answer from output: {output}")
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

    def _ensure_llm_loaded(self, progress_callback=None):
        """Load LLM on-demand if not already loaded."""
        if self.llm is None:
            vllm, SamplingParams = _get_vllm()
            
            if progress_callback:
                progress_callback(0.0, f"Loading LLM model ({self.model_name.split('/')[-1]})...")
            
            self.llm, was_cached = get_vllm_instance(
                model_name=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
            )
            sampling_config = {**self.generation_config, "n": self.self_consistency_k}
            self.sampling_params = SamplingParams(**sampling_config)
            
            if progress_callback:
                status = "Using cached LLM" if was_cached else "LLM loaded"
                progress_callback(0.1, f"{status}, starting disambiguation...")

    def __call__(self, doc: Doc) -> Doc:
        """Disambiguate all entities in the document."""
        if self.kb is None:
            logger.warning("vLLM disambiguator not initialized - call initialize(kb) first")
            return doc

        text = doc.text
        entities = list(doc.ents)
        num_entities = len(entities)
        
        if num_entities == 0:
            return doc

        # Load LLM on-demand (may evict unused models to free memory)
        self._ensure_llm_loaded(self.progress_callback)

        # Progress: 0.0-0.1 = model loading, 0.1-1.0 = processing entities
        processing_start = 0.1
        processing_range = 0.9

        for i, ent in enumerate(entities):
            ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
            base_progress = processing_start + (i / num_entities) * processing_range
            entity_progress_range = processing_range / num_entities
            
            def report_entity_progress(sub_progress: float, sub_desc: str):
                if self.progress_callback and num_entities > 0:
                    progress = base_progress + sub_progress * entity_progress_range
                    self.progress_callback(progress, f"Entity {i+1}/{num_entities} ({ent_text}): {sub_desc}")
            
            report_entity_progress(0.0, "checking candidates")
            
            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            # If only one candidate and no none option, select it directly
            if len(candidates) == 1 and not self.add_none_candidate:
                report_entity_progress(0.5, "single candidate, selecting")
                entity = self.kb.get_entity(candidates[0].entity_id)
                if entity:
                    ent._.resolved_entity = entity
                continue

            report_entity_progress(0.1, f"preparing prompt ({len(candidates)} candidates)")

            # Mark mention in text
            marked_text = self._mark_mention(text, ent.start_char, ent.end_char)

            # Create messages for LLM
            messages = create_disambiguation_messages(
                marked_text=marked_text,
                candidates=candidates,
                kb=self.kb,
                system_prompt=self.system_prompt,
                add_none_candidate=self.add_none_candidate,
                add_descriptions=self.add_descriptions,
                disable_thinking=self.disable_thinking,
            )

            report_entity_progress(0.2, "calling LLM...")

            try:
                # Apply chat template manually for more control
                tokenizer = self.llm.get_tokenizer()
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for msg in messages:
                    logger.debug(f"[{msg['role']}] {msg['content']}")
                logger.debug(f"Formatted prompt for '{ent.text}':\n{prompt}")

                responses = self.llm.generate(
                    [prompt],
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                response = responses[0] if responses else None
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                continue

            if response is None:
                continue

            report_entity_progress(0.9, "parsing LLM response")

            try:
                # Log the raw LLM output for debugging
                raw_output = response.outputs[0].text if response.outputs else ""
                logger.debug(f"LLM raw output for '{ent.text}': {raw_output}")
                
                answer = self._apply_self_consistency(response.outputs)
                logger.debug(f"Parsed answer: {answer} (from {len(candidates)} candidates)")

                # Answer 0 means "None" if add_none_candidate is True, or parsing failed
                if answer == 0:
                    logger.debug(f"Skipping entity '{ent.text}': answer was 0 (none or parse failure)")
                    continue

                if 0 < answer <= len(candidates):
                    selected = candidates[answer - 1]
                    logger.debug(f"Selected candidate: '{selected.entity_id}'")
                    entity = self.kb.get_entity(selected.entity_id)
                    if entity:
                        ent._.resolved_entity = entity
                        logger.debug(f"Resolved '{ent.text}' to '{entity.title}' (id: {entity.id})")
                    else:
                        logger.warning(f"Entity not found in KB: '{selected.entity_id}'")
                else:
                    logger.debug(f"Answer {answer} out of range for {len(candidates)} candidates")

            except Exception as e:
                logger.error(f"Error processing LLM response: {e}")
                continue
        
        # Release LLM - stays cached but can be evicted if memory needed
        release_vllm(self.model_name, self.tensor_parallel_size)
        
        # Clear progress callback after processing
        self.progress_callback = None

        return doc


# ============================================================================
# LELA Transformers Disambiguator Component (for older GPUs)
# ============================================================================

@Language.factory(
    "ner_pipeline_lela_transformers_disambiguator",
    default_config={
        "model_name": DEFAULT_LLM_MODEL,
        "add_none_candidate": True,  # Enable NIL handling by default
        "add_descriptions": True,
        "disable_thinking": True,
        "system_prompt": None,
        "generation_config": None,
    },
)
def create_lela_transformers_disambiguator_component(
    nlp: Language,
    name: str,
    model_name: str,
    add_none_candidate: bool,
    add_descriptions: bool,
    disable_thinking: bool,
    system_prompt: Optional[str],
    generation_config: Optional[dict],
):
    """Factory for LELA transformers disambiguator component."""
    return LELATransformersDisambiguatorComponent(
        nlp=nlp,
        model_name=model_name,
        add_none_candidate=add_none_candidate,
        add_descriptions=add_descriptions,
        disable_thinking=disable_thinking,
        system_prompt=system_prompt,
        generation_config=generation_config,
    )


class LELATransformersDisambiguatorComponent:
    """
    Transformers-based entity disambiguator component for spaCy.

    Alternative to vLLM for older GPUs (like P100) where vLLM has issues.
    Uses HuggingFace transformers directly for inference.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_LLM_MODEL,
        add_none_candidate: bool = False,
        add_descriptions: bool = True,
        disable_thinking: bool = True,
        system_prompt: Optional[str] = None,
        generation_config: Optional[dict] = None,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.add_none_candidate = add_none_candidate
        self.add_descriptions = add_descriptions
        self.disable_thinking = disable_thinking
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.generation_config = generation_config or DEFAULT_GENERATION_CONFIG

        _ensure_extensions()

        # Lazy load model and tokenizer
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading transformers model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")
        self.model.eval()

        self.kb = None
        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"LELA transformers disambiguator initialized: {model_name}")

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        self.kb = kb

    @staticmethod
    def _parse_output(output: str) -> int:
        """Parse LLM output to extract answer index.

        Handles Qwen3 thinking mode by extracting content after </think> tag.
        """
        # Handle Qwen3 thinking mode - extract content after </think>
        if "</think>" in output:
            output = output.split("</think>")[-1].strip()

        # Try standard format: "answer": N or answer: N
        match = re.search(r'"?answer"?\s*:\s*(\d+)', output, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Try standalone number
        match = re.search(r'(?:^|\s)(\d+)(?:\s|$|\.)', output.strip())
        if match:
            return int(match.group(1))

        # Last resort: find any digit
        match = re.search(r'(\d+)', output)
        if match:
            return int(match.group(1))

        logger.debug(f"Could not parse answer from output: {output}")
        return 0

    def _mark_mention(self, text: str, start: int, end: int) -> str:
        """Mark mention in text with brackets."""
        return f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"

    def __call__(self, doc: Doc) -> Doc:
        """Disambiguate all entities in the document."""
        import torch

        if self.kb is None:
            logger.warning("Transformers disambiguator not initialized - call initialize(kb) first")
            return doc

        text = doc.text
        entities = list(doc.ents)
        num_entities = len(entities)

        for i, ent in enumerate(entities):
            ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
            base_progress = i / num_entities
            entity_progress_range = 1.0 / num_entities

            def report_entity_progress(sub_progress: float, sub_desc: str):
                if self.progress_callback and num_entities > 0:
                    progress = base_progress + sub_progress * entity_progress_range
                    self.progress_callback(progress, f"Entity {i+1}/{num_entities} ({ent_text}): {sub_desc}")

            report_entity_progress(0.0, "checking candidates")

            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            if len(candidates) == 1 and not self.add_none_candidate:
                report_entity_progress(0.5, "single candidate, selecting")
                entity = self.kb.get_entity(candidates[0].entity_id)
                if entity:
                    ent._.resolved_entity = entity
                continue

            report_entity_progress(0.1, f"preparing prompt ({len(candidates)} candidates)")

            marked_text = self._mark_mention(text, ent.start_char, ent.end_char)
            messages = create_disambiguation_messages(
                marked_text=marked_text,
                candidates=candidates,
                kb=self.kb,
                system_prompt=self.system_prompt,
                add_none_candidate=self.add_none_candidate,
                add_descriptions=self.add_descriptions,
                disable_thinking=self.disable_thinking,
            )

            report_entity_progress(0.2, "calling LLM...")

            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for msg in messages:
                    logger.debug(f"[{msg['role']}] {msg['content']}")
                logger.debug(f"Formatted prompt for '{ent.text}':\n{prompt}")

                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.generation_config.get("max_tokens", 2048),
                        temperature=self.generation_config.get("temperature", 0.1),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                # Extract only the generated part (remove the prompt)
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                logger.debug(f"LLM raw output for '{ent.text}': {raw_output}")

            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                continue

            report_entity_progress(0.9, "parsing LLM response")

            try:
                answer = self._parse_output(raw_output)
                logger.debug(f"Parsed answer: {answer} (from {len(candidates)} candidates)")

                if answer == 0:
                    logger.debug(f"Skipping entity '{ent.text}': answer was 0 (none or parse failure)")
                    continue

                if 0 < answer <= len(candidates):
                    selected = candidates[answer - 1]
                    logger.debug(f"Selected candidate: '{selected.entity_id}'")
                    entity = self.kb.get_entity(selected.entity_id)
                    if entity:
                        ent._.resolved_entity = entity
                        logger.debug(f"Resolved '{ent.text}' to '{entity.title}' (id: {entity.id})")
                    else:
                        logger.warning(f"Entity not found in KB: '{selected.entity_id}'")
                else:
                    logger.debug(f"Answer {answer} out of range for {len(candidates)} candidates")

            except Exception as e:
                logger.error(f"Error processing LLM response: {e}")
                continue

        self.progress_callback = None
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
        
        # Optional progress callback for fine-grained progress reporting
        self.progress_callback: Optional[ProgressCallback] = None

        _ensure_extensions()

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        self.kb = kb

    def __call__(self, doc: Doc) -> Doc:
        """Select first candidate for all entities."""
        if self.kb is None:
            logger.warning("First disambiguator not initialized - call initialize(kb) first")
            return doc

        entities = list(doc.ents)
        num_entities = len(entities)

        for i, ent in enumerate(entities):
            # Report progress if callback is set
            if self.progress_callback and num_entities > 0:
                progress = i / num_entities
                ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                self.progress_callback(progress, f"Disambiguating entity {i+1}/{num_entities}: {ent_text}")
            
            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            # Select first candidate by entity_id
            entity = self.kb.get_entity(candidates[0].entity_id)
            if entity:
                ent._.resolved_entity = entity

        # Clear progress callback after processing
        self.progress_callback = None

        return doc



