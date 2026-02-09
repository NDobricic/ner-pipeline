"""Prompt templates for LELA disambiguation."""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from el_pipeline.knowledge_bases.base import KnowledgeBase
    from el_pipeline.types import Candidate

DEFAULT_SYSTEM_PROMPT = f"""You are an expert designed to disambiguate entities in text, taking into account the overall context and a list of entity candidates.
You are provided with an input text that includes a full contextual narrative, the mention enclosed between the '[' and ']' characters, and a list of candidates, each preceded by an index number.
Your task is to determine the most appropriate entity from the candidates based on the context and candidate entity descriptions.
Please show your choice in the answer field with only the choice index number, e.g., "answer": 3."""


def create_disambiguation_messages(
    marked_text: str,
    candidates: List["Candidate"],
    kb: Optional["KnowledgeBase"] = None,
    system_prompt: Optional[str] = None,
    query_prompt: Optional[str] = None,
    add_none_candidate: bool = True,
    add_descriptions: bool = True,
    disable_thinking: bool = False,
) -> List[dict]:
    """
    Create message list for LLM disambiguation.

    Args:
        marked_text: Text with mention marked using [brackets]
        candidates: List of Candidate objects
        kb: Optional knowledge base for looking up entity titles
        system_prompt: Optional custom system prompt
        query_prompt: Optional additional query context
        add_none_candidate: Whether to include "None" option
        add_descriptions: Whether to include entity descriptions
        disable_thinking: Whether to use a simpler prompt that asks for just a number

    Returns:
        List of message dicts for chat API
    """
    messages = []

    # Use appropriate system prompt
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    messages.append({"role": "system", "content": final_system_prompt})

    if query_prompt:
        messages.append({"role": "user", "content": query_prompt})

    # Build candidate list string
    none_option = "0. None of the listed candidates\n" if add_none_candidate else ""

    candidate_lines = []
    for i, candidate in enumerate(candidates):
        # Get entity title from KB if available, otherwise use entity_id
        if kb:
            entity = kb.get_entity(candidate.entity_id)
            title = entity.title if entity else candidate.entity_id
        else:
            title = candidate.entity_id

        if add_descriptions and candidate.description:
            candidate_lines.append(f"{i + 1}. {title} - {candidate.description}")
        else:
            candidate_lines.append(f"{i + 1}. {title}")

    candidate_str = none_option + "\n".join(candidate_lines)

    user_message = (
        f"Input text: {marked_text}\nList of candidate entities:\n{candidate_str}"
    )

    # Add /no_think soft switch for Qwen3 models when thinking is disabled
    # This is a soft switch that Qwen3 recognizes to skip chain-of-thought
    if disable_thinking:
        user_message += " /no_think"

    messages.append({"role": "user", "content": user_message})

    return messages


def mark_mention_in_text(
    text: str,
    start: int,
    end: int,
    open_marker: str = "[",
    close_marker: str = "]",
) -> str:
    """
    Mark a mention in text with brackets.

    Args:
        text: Full text
        start: Mention start offset
        end: Mention end offset
        open_marker: Opening bracket character
        close_marker: Closing bracket character

    Returns:
        Text with mention marked
    """
    return f"{text[:start]}{open_marker}{text[start:end]}{close_marker}{text[end:]}"
