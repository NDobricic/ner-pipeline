from typing import Iterable, List, Optional, Protocol

from el_pipeline.types import Entity


class KnowledgeBase(Protocol):
    """Abstract knowledge base interface."""

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        ...

    def search(self, query: str, top_k: int = 10) -> List[Entity]:
        ...

    def all_entities(self) -> Iterable[Entity]:
        ...

