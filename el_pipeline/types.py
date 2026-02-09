from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Type alias for progress callbacks used throughout the pipeline
ProgressCallback = Callable[[float, str], None]


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
