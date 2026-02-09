from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ComponentConfig:
    """Generic component configuration."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    loader: ComponentConfig
    ner: ComponentConfig
    candidate_generator: ComponentConfig
    reranker: Optional[ComponentConfig] = None
    disambiguator: Optional[ComponentConfig] = None
    knowledge_base: Optional[ComponentConfig] = None
    cache_dir: str = ".ner_cache"
    batch_size: int = 1

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PipelineConfig":
        def build(section: str) -> Optional[ComponentConfig]:
            if section not in data or data[section] is None:
                return None
            entry = data[section]
            return ComponentConfig(name=entry["name"], params=entry.get("params", {}))

        return PipelineConfig(
            loader=build("loader"),  # type: ignore[arg-type]
            ner=build("ner"),  # type: ignore[arg-type]
            candidate_generator=build("candidate_generator"),  # type: ignore[arg-type]
            reranker=build("reranker"),
            disambiguator=build("disambiguator"),
            knowledge_base=build("knowledge_base"),
            cache_dir=data.get("cache_dir", ".ner_cache"),
            batch_size=data.get("batch_size", 1),
        )

