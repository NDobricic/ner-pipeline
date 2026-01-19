from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Document:
    """Single document item."""

    id: Optional[str]
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Mention:
    """Raw mention detected by NER."""

    start: int
    end: int
    text: str
    label: Optional[str] = None
    context: Optional[str] = None


@dataclass
class Candidate:
    """Candidate entity reference."""

    entity_id: str
    score: Optional[float] = None
    description: Optional[str] = None


@dataclass
class Entity:
    """Entity record from a knowledge base."""

    id: str
    title: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedMention:
    """Mention resolved to a KB entity."""

    mention: Mention
    entity: Optional[Entity]
    candidates: List[Candidate] = field(default_factory=list)


# ============================================================================
# Conversion Utilities for LELA Format
# ============================================================================

def candidates_to_tuples(candidates: List[Candidate]) -> List[Tuple[str, str]]:
    """
    Convert Candidate list to LELA tuple format.

    LELA format uses (title, description) tuples for candidates.

    Args:
        candidates: List of Candidate objects

    Returns:
        List of (entity_id/title, description) tuples
    """
    return [(c.entity_id, c.description or "") for c in candidates]


def tuples_to_candidates(tuples: List[Tuple[str, str]]) -> List[Candidate]:
    """
    Convert LELA tuple format to Candidate list.

    Args:
        tuples: List of (title, description) tuples

    Returns:
        List of Candidate objects
    """
    return [
        Candidate(entity_id=title, score=None, description=description)
        for title, description in tuples
    ]
