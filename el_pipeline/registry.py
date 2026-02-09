"""
Component registry for EL pipeline.

Provides pluggable component registration for document loaders
and knowledge bases. NER, candidate generation, reranking, and
disambiguation components are spaCy factories (see spacy_components/).
"""

from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """Simple registry to keep components pluggable."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._registry:
                raise ValueError(f"Component '{name}' already registered.")
            self._registry[name] = factory
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(f"Component '{name}' not found.") from exc

    def available(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._registry)


# Registries for all pipeline components
loaders = ComponentRegistry()
knowledge_bases = ComponentRegistry()
ner_models = ComponentRegistry()
candidate_generators = ComponentRegistry()
rerankers = ComponentRegistry()
disambiguators = ComponentRegistry()